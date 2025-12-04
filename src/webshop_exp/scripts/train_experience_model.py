"""Train the Experience Model using LoRA fine-tuning.

Usage:
    uv run train-experience-model --base-model Qwen/Qwen2.5-3B-Instruct --output models/experience_model
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
        save_strategy="epoch" if args.max_steps < 0 else "steps",
        save_steps=500,
        eval_strategy="epoch" if "validation" in dataset and args.max_steps < 0 else "steps" if "validation" in dataset else "no",
        eval_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"experience-model-{args.base_model.split('/')[-1]}",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )
    
    # Create trainer (TRL 0.25+ API uses `processing_class` instead of `tokenizer`)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
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
