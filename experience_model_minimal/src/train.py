"""Train the Experience Model using LoRA fine-tuning.

This is the core training script that fine-tunes a base LLM to predict
state transitions: (state, action) → (next_state, reward, done)

Usage:
    python src/train.py \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --output models/experience_model \
        --epochs 3

Key concepts:
    - LoRA: Low-rank adapters for efficient fine-tuning (~1% of params)
    - QLoRA: 4-bit quantization + LoRA for lower memory usage
    - SFTTrainer: Supervised fine-tuning using HuggingFace TRL
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import yaml

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Enable TF32 for faster matmuls on Ampere+ GPUs (A40, A100, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Sample Generation Callback (Critical for Monitoring)
# =============================================================================

class SampleGenerationCallback(TrainerCallback):
    """Generate samples at regular intervals to monitor training progress.
    
    This callback is CRITICAL for understanding how training is progressing.
    It generates predictions on fixed validation samples and saves them,
    allowing you to see qualitative improvements over time.
    """
    
    def __init__(self, tokenizer, eval_dataset, output_dir: Path, 
                 sample_every: int = 100, num_samples: int = 3):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.sample_every = sample_every
        self.num_samples = num_samples
        self.samples_file = self.output_dir / "training_samples.jsonl"
        self.sample_indices = None
        
    def _get_sample_indices(self):
        """Get fixed sample indices for consistent comparison across steps."""
        if self.sample_indices is None:
            dataset_size = len(self.eval_dataset) if self.eval_dataset else 0
            if dataset_size > 0:
                random.seed(42)  # Deterministic for reproducibility
                self.sample_indices = random.sample(
                    range(dataset_size), 
                    min(self.num_samples, dataset_size)
                )
            else:
                self.sample_indices = []
        return self.sample_indices
    
    def _extract_from_messages(self, messages: list, role: str) -> str:
        """Extract content from messages by role."""
        for msg in messages:
            if msg.get("role") == role:
                return msg.get("content", "")
        return ""
    
    def _generate_samples(self, model, step: int):
        """Generate samples and save to file."""
        if not self.eval_dataset or not self._get_sample_indices():
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        
        model.eval()
        for idx in self._get_sample_indices():
            example = self.eval_dataset[idx]
            messages = example.get("messages", [])
            
            prompt = self._extract_from_messages(messages, "user")
            ground_truth = self._extract_from_messages(messages, "assistant")
            
            if not prompt:
                continue
            
            # Format for generation
            chat_messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                input_text, return_tensors="pt", 
                truncation=True, max_length=1024
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            try:
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=True,
                        )
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
            except Exception as e:
                generated = f"[Generation error: {e}]"
            
            sample = {
                "step": step,
                "sample_idx": idx,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "ground_truth": ground_truth,
                "generated": generated,
            }
            samples.append(sample)
            
            # Log to console
            logger.info(f"\n{'='*60}")
            logger.info(f"SAMPLE @ Step {step} (idx={idx})")
            logger.info(f"{'='*60}")
            logger.info(f"GROUND TRUTH: {ground_truth[:200]}...")
            logger.info(f"GENERATED: {generated[:200]}...")
        
        # Append to JSONL file
        with open(self.samples_file, "a") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        model.train()
        logger.info(f"Saved {len(samples)} samples to {self.samples_file}")
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Generate samples before any training."""
        logger.info("Generating initial samples (step 0)...")
        self._generate_samples(model, step=0)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate samples at regular intervals."""
        if state.global_step > 0 and state.global_step % self.sample_every == 0:
            self._generate_samples(model, step=state.global_step)
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Generate final samples."""
        logger.info(f"Generating final samples (step {state.global_step})...")
        self._generate_samples(model, step=state.global_step)


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        },
        "training": {
            "epochs": 3,
            "batch_size": 2,  # Reduced for memory
            "gradient_accumulation": 8,  # Increased to maintain effective batch size
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "lr_scheduler": "cosine",
            "max_seq_length": 1024,
            "seed": 42,
        },
        "optimization": {
            "use_4bit": True,
            "gradient_checkpointing": True,  # Critical for memory
            "bf16": True,
        },
        "checkpointing": {
            "save_steps": 200,
            "save_total_limit": 2,
            "early_stopping_patience": 2,
        },
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_tokenizer(model_name: str):
    """Load and configure tokenizer."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def load_model(model_name: str, use_4bit: bool = True):
    """Load base model with optional quantization.
    
    Args:
        model_name: HuggingFace model identifier
        use_4bit: Whether to use 4-bit quantization (QLoRA)
    
    Returns:
        Loaded model ready for LoRA
    """
    logger.info(f"Loading model: {model_name}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    # Enable Flash Attention 2 if available
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    except ImportError:
        logger.info("Flash Attention 2 not available")
    
    # 4-bit quantization for lower memory usage
    if use_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model


def apply_lora(model, lora_config: dict):
    """Apply LoRA adapters to the model.
    
    LoRA (Low-Rank Adaptation) adds small trainable matrices to specific
    layers, enabling efficient fine-tuning with ~1% of parameters.
    
    Args:
        model: Base model
        lora_config: LoRA configuration dict
    
    Returns:
        Model with LoRA adapters
    """
    logger.info(f"Applying LoRA (rank={lora_config['rank']}, alpha={lora_config['alpha']})")
    
    config = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model


# =============================================================================
# Data Loading
# =============================================================================

def load_training_data(train_path: Path, val_path: Path):
    """Load training and validation datasets."""
    logger.info(f"Loading training data from {train_path}")
    
    data_files = {"train": str(train_path)}
    if val_path and val_path.exists():
        data_files["validation"] = str(val_path)
        logger.info(f"Loading validation data from {val_path}")
    
    dataset = load_dataset("json", data_files=data_files)
    
    logger.info(f"Loaded {len(dataset['train'])} training samples")
    if "validation" in dataset:
        logger.info(f"Loaded {len(dataset['validation'])} validation samples")
    
    return dataset


# =============================================================================
# Training
# =============================================================================

def train(
    model,
    tokenizer,
    dataset,
    output_dir: Path,
    training_config: dict,
    optimization_config: dict,
    checkpointing_config: dict,
    callbacks: list = None,
    max_steps: int = -1,
):
    """Run the training loop.
    
    Uses HuggingFace TRL's SFTTrainer for supervised fine-tuning.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        dataset: Dataset with 'train' and optionally 'validation' splits
        output_dir: Where to save checkpoints and final model
        training_config: Training hyperparameters
        optimization_config: Memory/precision settings
        checkpointing_config: Save/early stopping settings
        callbacks: Optional list of TrainerCallbacks
        max_steps: Max training steps (-1 for full epochs)
    """
    logger.info("Setting up training")
    
    # Determine eval strategy
    has_validation = "validation" in dataset
    eval_strategy = "epoch" if has_validation and max_steps < 0 else "steps" if has_validation else "no"
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        
        # Training hyperparameters
        num_train_epochs=training_config["epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler"],
        seed=training_config["seed"],
        
        # Precision
        bf16=optimization_config["bf16"],
        
        # Memory optimization
        gradient_checkpointing=optimization_config["gradient_checkpointing"],
        
        # Checkpointing
        save_strategy="steps",
        save_steps=checkpointing_config["save_steps"],
        save_total_limit=checkpointing_config["save_total_limit"],
        
        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=500 if eval_strategy == "steps" else None,
        
        # Logging
        logging_steps=10,
        report_to="none",
        
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",  # Faster optimizer (~10%)
        remove_unused_columns=True,
    )
    
    # Handle early stopping with load_best_model
    if callbacks is None:
        callbacks = []
    
    if has_validation and any(isinstance(cb, EarlyStoppingCallback) for cb in callbacks):
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"  Epochs: {training_config['epochs']}")
    logger.info(f"  Batch size: {training_config['batch_size']}")
    logger.info(f"  Gradient accumulation: {training_config['gradient_accumulation']}")
    logger.info(f"  Effective batch size: {training_config['batch_size'] * training_config['gradient_accumulation']}")
    logger.info(f"  Learning rate: {training_config['learning_rate']}")
    
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save LoRA adapter separately
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")
    
    return trainer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Experience Model with LoRA")
    
    # Data
    parser.add_argument("--train-data", type=Path, required=True, help="Training data JSONL")
    parser.add_argument("--val-data", type=Path, default=None, help="Validation data JSONL")
    parser.add_argument("--output", type=Path, default=Path("models/experience_model"), help="Output directory")
    
    # Model
    parser.add_argument("--base-model", type=str, default=None, help="Base model to fine-tune")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="Config file")
    
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha")
    
    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Max sequence length")
    
    # Optimization
    parser.add_argument("--use-4bit", action="store_true", default=None, help="Use 4-bit quantization")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for ~20%% speedup (not with 4-bit)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    
    # Monitoring
    parser.add_argument("--sample-every", type=int, default=0, help="Generate samples every N steps (0=disabled)")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate at each checkpoint")
    
    # Other
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    if args.config.exists():
        file_config = load_config(args.config)
        # Deep merge
        for key in file_config:
            if isinstance(file_config[key], dict) and key in config:
                config[key].update(file_config[key])
            else:
                config[key] = file_config[key]
    
    # Override with CLI args
    if args.base_model:
        config["base_model"] = args.base_model
    if args.lora_rank:
        config["lora"]["rank"] = args.lora_rank
    if args.lora_alpha:
        config["lora"]["alpha"] = args.lora_alpha
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.gradient_accumulation:
        config["training"]["gradient_accumulation"] = args.gradient_accumulation
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_seq_length:
        config["training"]["max_seq_length"] = args.max_seq_length
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.use_4bit:
        config["optimization"]["use_4bit"] = True
    if args.no_4bit:
        config["optimization"]["use_4bit"] = False
    if args.gradient_checkpointing:
        config["optimization"]["gradient_checkpointing"] = True
    
    # Validate inputs
    if not args.train_data.exists():
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Run prepare_data.py first to create training data.")
        return 1
    
    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load components
    tokenizer = load_tokenizer(config["base_model"])
    model = load_model(config["base_model"], use_4bit=config["optimization"]["use_4bit"])
    model = apply_lora(model, config["lora"])
    
    # Optional torch.compile for speedup
    if args.compile:
        if config["optimization"]["use_4bit"]:
            logger.warning("torch.compile not compatible with 4-bit quantization - skipping")
        else:
            logger.info("Compiling model with torch.compile (first few steps will be slower)")
            model = torch.compile(model)
    
    dataset = load_training_data(args.train_data, args.val_data)
    
    # Build callbacks
    callbacks = []
    
    # Early stopping
    if "validation" in dataset and config["checkpointing"].get("early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["checkpointing"]["early_stopping_patience"]
            )
        )
        logger.info(f"Early stopping enabled (patience={config['checkpointing']['early_stopping_patience']})")
    
    # Sample generation for monitoring
    if args.sample_every > 0 and "validation" in dataset:
        callbacks.append(
            SampleGenerationCallback(
                tokenizer=tokenizer,
                eval_dataset=dataset["validation"],
                output_dir=args.output,
                sample_every=args.sample_every,
                num_samples=args.num_samples,
            )
        )
        logger.info(f"Sample generation enabled every {args.sample_every} steps")
    
    # Train
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        training_config=config["training"],
        optimization_config=config["optimization"],
        checkpointing_config=config["checkpointing"],
        callbacks=callbacks,
        max_steps=args.max_steps,
    )
    
    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())
