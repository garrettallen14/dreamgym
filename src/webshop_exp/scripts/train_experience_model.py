"""Train the Experience Model using LoRA fine-tuning.

Usage:
    uv run train-experience-model --base-model Qwen/Qwen2.5-3B-Instruct --output models/experience_model

Optimized based on research insights:
- Gradient checkpointing OFF by default (A40 48GB has headroom)
- Flash Attention 2 enabled when available
- Early stopping on eval loss
- Tokenizer parallelism disabled to avoid fork warnings
"""

import argparse
import logging
import os
from pathlib import Path

# Suppress tokenizer parallelism warning (minimal throughput impact per benchmarks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Enable TF32 for faster matmuls on Ampere+ GPUs (A40, A100, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import json
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default base models in order of preference
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]


def format_chat_template(example: dict, tokenizer) -> str:
    """Format example using the model's chat template."""
    messages = example.get("messages", [])
    if not messages:
        # Fallback for raw prompt/completion format
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class SampleGenerationCallback(TrainerCallback):
    """Generate samples at regular intervals to monitor training progress."""
    
    def __init__(self, tokenizer, eval_dataset, output_dir: Path, sample_every: int = 100, num_samples: int = 3):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.sample_every = sample_every
        self.num_samples = num_samples
        self.samples_file = output_dir / "samples.jsonl"
        self.sample_indices = None
        
    def _get_sample_indices(self):
        """Get fixed sample indices for consistent comparison."""
        if self.sample_indices is None:
            dataset_size = len(self.eval_dataset) if self.eval_dataset else 0
            if dataset_size > 0:
                random.seed(42)  # Deterministic samples
                self.sample_indices = random.sample(range(dataset_size), min(self.num_samples, dataset_size))
            else:
                self.sample_indices = []
        return self.sample_indices
    
    def _extract_prompt_from_messages(self, messages: list) -> str:
        """Extract user prompt from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def _extract_ground_truth(self, messages: list) -> str:
        """Extract assistant response (ground truth) from messages."""
        for msg in messages:
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    
    def _generate_samples(self, model, step: int):
        """Generate samples and save to file."""
        if not self.eval_dataset or not self._get_sample_indices():
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        
        model.eval()
        with torch.no_grad():
            for idx in self._get_sample_indices():
                example = self.eval_dataset[idx]
                messages = example.get("messages", [])
                
                prompt = self._extract_prompt_from_messages(messages)
                ground_truth = self._extract_ground_truth(messages)
                
                if not prompt:
                    continue
                
                # Format as chat for generation
                chat_messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate with inference mode and proper dtype handling
                try:
                    with torch.inference_mode():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=256,
                                do_sample=False,  # Greedy for speed
                                pad_token_id=self.tokenizer.pad_token_id,
                                use_cache=True,
                            )
                    generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
                logger.info(f"PROMPT: {sample['prompt'][:200]}...")
                logger.info(f"GROUND TRUTH: {ground_truth[:300]}...")
                logger.info(f"GENERATED: {generated[:300]}...")
        
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
            logger.info(f"Generating samples at step {state.global_step}...")
            self._generate_samples(model, step=state.global_step)
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Generate final samples."""
        logger.info(f"Generating final samples (step {state.global_step})...")
        self._generate_samples(model, step=state.global_step)


def load_training_data(train_path: Path, val_path: Path):
    """Load training and validation datasets."""
    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    
    dataset = load_dataset("json", data_files=data_files)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train experience model with LoRA")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/experience_train_hf.jsonl"),
        help="Training data JSONL file",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/experience_val_hf.jsonl"),
        help="Validation data JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/experience_model"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="webshop-experience-model",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves memory, ~20%% slower). Off by default for A40.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disabled). Stop if eval loss doesn't improve for N evals.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=100,
        help="Generate samples every N steps (0 = disabled). Shows model progress vs ground truth.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to generate at each checkpoint.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for ~20%% speedup (first few steps slower due to compilation)",
    )
    
    args = parser.parse_args()
    
    # Check training data exists
    if not args.train_data.exists():
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Run prepare-experience-data --hf-format first.")
        return 1
    
    # Setup wandb
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
    logger.info(f"Loading model from {args.base_model}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    # Enable Flash Attention 2 if available (2x speedup)
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2 for faster training")
    except ImportError:
        logger.info("Flash Attention 2 not available, using default attention")
    
    if args.use_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    
    # Setup LoRA
    logger.info(f"Setting up LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Optional torch.compile for speedup (not compatible with quantized models)
    if args.compile:
        if args.use_4bit:
            logger.warning("torch.compile not compatible with quantized models - skipping")
        else:
            logger.info("Compiling model with torch.compile (first few steps will be slower)")
            model = torch.compile(model)
    
    # Load dataset
    logger.info("Loading training data")
    dataset = load_training_data(args.train_data, args.val_data)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",  # Save periodically
        save_steps=200,         # Save every 200 steps (~40min)
        save_total_limit=2,     # Keep only 2 checkpoints to save disk space
        eval_strategy="epoch" if "validation" in dataset and args.max_steps < 0 else "steps" if "validation" in dataset else "no",
        eval_steps=500,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"experience-model-{args.base_model.split('/')[-1]}",
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        # Speedups
        optim="adamw_torch_fused",  # Fused optimizer (~10% faster)
    )
    
    # Early stopping callback if enabled
    callbacks = []
    if args.early_stopping_patience > 0 and "validation" in dataset:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
        logger.info(f"Early stopping enabled with patience={args.early_stopping_patience}")
    
    # Sample generation callback for monitoring progress
    if args.sample_every > 0 and "validation" in dataset:
        sample_callback = SampleGenerationCallback(
            tokenizer=tokenizer,
            eval_dataset=dataset["validation"],
            output_dir=args.output,
            sample_every=args.sample_every,
            num_samples=args.num_samples,
        )
        callbacks.append(sample_callback)
        logger.info(f"Sample generation enabled every {args.sample_every} steps ({args.num_samples} samples)")
    
    # Create trainer (TRL 0.25+ API uses `processing_class` instead of `tokenizer`)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    # Also save the LoRA adapter separately
    adapter_path = args.output / "adapter"
    model.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")
    
    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())
