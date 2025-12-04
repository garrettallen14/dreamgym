"""Train the agent using LoRA fine-tuning.

Usage:
    uv run train-agent --data real --output models/agent_real
    uv run train-agent --data synthetic --output models/agent_synthetic
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
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train agent with LoRA")
    parser.add_argument(
        "--method",
        type=str,
        choices=["sft", "dpo"],
        default="sft",
        help="Training method: 'sft' (behavior cloning) or 'dpo' (preference learning)",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["real", "synthetic", "custom"],
        default="real",
        help="Data source: 'real', 'synthetic', or 'custom'",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Custom training data path (when --data=custom)",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Custom validation data path",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
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
        default="webshop-agent",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    
    args = parser.parse_args()
    
    # Determine data paths
    if args.data == "custom":
        if args.train_data is None:
            logger.error("--train-data required when using --data=custom")
            return 1
        train_path = args.train_data
        val_path = args.val_data
    else:
        # Use standard paths
        data_dir = Path("data")
        train_path = data_dir / f"agent_train_{args.data}.jsonl"
        val_path = data_dir / f"agent_val_{args.data}.jsonl"
        
        # Try HF format first
        train_hf_path = data_dir / f"agent_train_{args.data}_hf.jsonl"
        if train_hf_path.exists():
            train_path = train_hf_path
            val_path = data_dir / f"agent_val_{args.data}_hf.jsonl"
    
    # Set default output
    if args.output is None:
        args.output = Path(f"models/agent_{args.data}")
    
    # Check training data exists
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run prepare-agent-data first to create the training data.")
        return 1
    
    logger.info(f"Training agent on {args.data} data")
    logger.info(f"  Train data: {train_path}")
    logger.info(f"  Val data: {val_path if val_path and val_path.exists() else 'None'}")
    logger.info(f"  Output: {args.output}")
    
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
    data_files = {"train": str(train_path)}
    if val_path and val_path.exists():
        data_files["validation"] = str(val_path)
    
    dataset = load_dataset("json", data_files=data_files)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if "validation" in dataset else "no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"agent-{args.data}-{args.base_model.split('/')[-1]}",
    )
    
    # Create trainer
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
    
    # Save LoRA adapter separately
    adapter_path = args.output / "adapter"
    model.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")
    
    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())
