"""Train the agent using GRPO (Group Relative Policy Optimization).

This is the proper DreamGym approach - using RL in the synthetic environment.

Usage:
    uv run python -m webshop_exp.scripts.train_agent_rl \
        --experience-model models/experiments/yolo_final/adapter \
        --method grpo \
        --output models/agent_grpo
"""

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SyntheticRolloutEnv:
    """Synthetic environment using the experience model for RL training."""
    
    def __init__(
        self,
        experience_model,
        tokenizer,
        instructions: List[str],
        max_new_tokens: int = 128,
    ):
        self.model = experience_model
        self.tokenizer = tokenizer
        self.instructions = instructions
        self.max_new_tokens = max_new_tokens
        
        self.current_instruction = None
        self.current_state = None
        self.step_count = 0
        self.history = []
    
    def reset(self, instruction: Optional[str] = None) -> str:
        """Reset environment with a task instruction."""
        if instruction is None:
            instruction = random.choice(self.instructions)
        
        self.current_instruction = instruction
        self.step_count = 0
        self.history = []
        
        # Initial state is search page
        self.current_state = f"""Instruction:
{instruction}

[Search]
Enter a search query to find products matching your requirements.

Available actions: search[query]"""
        
        return self.current_state
    
    def step(self, action: str) -> tuple:
        """Take action and get next state, reward, done from experience model."""
        self.step_count += 1
        
        # Format prompt for experience model
        prompt = f"""Task: {self.current_instruction}

Current State:
{self.current_state}

Action taken: {action}

Predict the next state and reward."""
        
        # Generate prediction
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)
        
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Parse output
        next_state, reward, done = self._parse_output(generated)
        
        self.history.append({
            "state": self.current_state,
            "action": action,
            "reward": reward,
        })
        self.current_state = next_state
        
        return next_state, reward, done, {"raw_output": generated}
    
    def _parse_output(self, output: str) -> tuple:
        """Parse model output to extract next_state, reward, done."""
        next_state = ""
        reward = 0.0
        done = False
        
        # Extract next state
        state_match = re.search(r"Next State:\s*(.*?)(?=\nReward:|$)", output, re.DOTALL)
        if state_match:
            next_state = state_match.group(1).strip()
        else:
            next_state = output.strip()
        
        # Extract reward
        reward_match = re.search(r"Reward:\s*([\d.]+)", output)
        if reward_match:
            try:
                reward = float(reward_match.group(1))
            except ValueError:
                pass
        
        # Extract done
        done_match = re.search(r"Done:\s*(True|False)", output, re.IGNORECASE)
        if done_match:
            done = done_match.group(1).lower() == "true"
        
        # Also check for purchase complete
        if "purchase complete" in output.lower():
            done = True
        
        return next_state, reward, done
    
    def get_trajectory_reward(self) -> float:
        """Get total reward from trajectory."""
        if self.history:
            return self.history[-1]["reward"]
        return 0.0


def load_experience_model(model_path: Path, base_model: str = None):
    """Load the trained experience model."""
    from peft import PeftModel
    
    # Find base model from adapter config
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists() and base_model is None:
        with open(adapter_config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path")
    
    if base_model is None:
        base_model = "Qwen/Qwen2.5-3B-Instruct"
    
    logger.info(f"Loading experience model base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer


def generate_rl_training_data(
    env: SyntheticRolloutEnv,
    agent_model,
    agent_tokenizer,
    num_episodes: int = 100,
    max_steps: int = 8,
) -> List[dict]:
    """Generate training data by rolling out agent in synthetic env."""
    
    training_data = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        episode_data = []
        
        while not done and step < max_steps:
            # Agent predicts action
            action = predict_action(agent_model, agent_tokenizer, env.current_instruction, state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            episode_data.append({
                "instruction": env.current_instruction,
                "state": state,
                "action": action,
                "reward": reward,
                "done": done,
            })
            
            state = next_state
            step += 1
        
        # Get trajectory reward
        traj_reward = env.get_trajectory_reward()
        
        # Add trajectory reward to all steps (for GRPO)
        for data in episode_data:
            data["trajectory_reward"] = traj_reward
            training_data.append(data)
    
    return training_data


def predict_action(model, tokenizer, instruction: str, state: str) -> str:
    """Predict action given instruction and state."""
    prompt = f"""Task: {instruction}

Current observation:
{state}

What action should you take?"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    # Extract action
    action_match = re.search(r"(search\[.+?\]|click\[.+?\])", generated, re.IGNORECASE)
    if action_match:
        return action_match.group(1)
    return generated.strip().split("\n")[0]


def create_preference_pairs(training_data: List[dict]) -> Dataset:
    """Create preference pairs for GRPO from trajectory data.
    
    GRPO learns from groups of responses with different rewards.
    We group by (instruction, state) and rank by reward.
    """
    from collections import defaultdict
    
    # Group by instruction + state
    groups = defaultdict(list)
    for item in training_data:
        key = (item["instruction"], item["state"][:200])  # Truncate state for grouping
        groups[key].append(item)
    
    # Create preference pairs
    pairs = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        
        # Sort by reward (descending)
        items_sorted = sorted(items, key=lambda x: x["trajectory_reward"], reverse=True)
        
        # Take best and worst
        best = items_sorted[0]
        worst = items_sorted[-1]
        
        if best["trajectory_reward"] > worst["trajectory_reward"]:
            pairs.append({
                "prompt": f"Task: {best['instruction']}\n\nCurrent observation:\n{best['state']}\n\nWhat action should you take?",
                "chosen": best["action"],
                "rejected": worst["action"],
            })
    
    return Dataset.from_list(pairs)


def train_sft(
    model,
    tokenizer,
    train_data: List[dict],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """Train agent using SFT on the trajectory data."""
    
    # Convert to HF format
    hf_data = []
    for item in train_data:
        prompt = f"Task: {item['instruction']}\n\nCurrent observation:\n{item['state']}\n\nWhat action should you take?"
        hf_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": item["action"]},
            ]
        })
    
    dataset = Dataset.from_list(hf_data)
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return trainer


def train_grpo(
    model,
    tokenizer,
    preference_data: Dataset,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
):
    """Train agent using GRPO on preference pairs."""
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        logger.warning("GRPOTrainer not available in this TRL version, falling back to DPO")
        return train_dpo(model, tokenizer, preference_data, output_dir, epochs, batch_size, learning_rate)
    
    config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=preference_data,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return trainer


def train_dpo(
    model,
    tokenizer,
    preference_data: Dataset,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
):
    """Train agent using DPO on preference pairs (fallback if GRPO unavailable)."""
    from trl import DPOConfig, DPOTrainer
    
    config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        beta=0.1,  # KL penalty coefficient
    )
    
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=preference_data,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return trainer


def load_instructions(path: Optional[Path]) -> List[str]:
    """Load instructions from file or use defaults."""
    if path and path.exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return [
        "Find me a black leather wallet under $50 with RFID blocking",
        "I need a laptop stand for my desk that is adjustable",
        "Search for wireless bluetooth headphones under $100",
    ]


def main():
    parser = argparse.ArgumentParser(description="Train agent with RL (GRPO/DPO)")
    parser.add_argument("--experience-model", type=Path, required=True, help="Path to experience model adapter")
    parser.add_argument("--method", choices=["sft", "grpo", "dpo"], default="sft", help="Training method")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("models/agent_rl"))
    parser.add_argument("--instructions", type=Path, default=None)
    parser.add_argument("--num-episodes", type=int, default=500, help="Number of rollout episodes")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load experience model for synthetic environment
    logger.info("Loading experience model...")
    exp_model, exp_tokenizer = load_experience_model(args.experience_model)
    
    # Load instructions
    instructions = load_instructions(args.instructions)
    logger.info(f"Loaded {len(instructions)} instructions")
    
    # Create synthetic environment
    env = SyntheticRolloutEnv(
        experience_model=exp_model,
        tokenizer=exp_tokenizer,
        instructions=instructions,
    )
    
    # Load agent model (separate from experience model)
    logger.info(f"Loading agent model: {args.base_model}")
    agent_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if agent_tokenizer.pad_token is None:
        agent_tokenizer.pad_token = agent_tokenizer.eos_token
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    agent_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    
    # Setup LoRA for agent
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    agent_model = get_peft_model(agent_model, lora_config)
    agent_model.print_trainable_parameters()
    
    # Generate rollout data
    logger.info(f"Generating {args.num_episodes} rollout episodes in synthetic env...")
    training_data = generate_rl_training_data(
        env=env,
        agent_model=agent_model,
        agent_tokenizer=agent_tokenizer,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )
    logger.info(f"Generated {len(training_data)} training samples")
    
    # Train based on method
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.method == "sft":
        logger.info("Training with SFT (behavior cloning)...")
        trainer = train_sft(
            agent_model, agent_tokenizer, training_data,
            args.output, args.epochs, args.batch_size, args.learning_rate
        )
    else:
        # Create preference pairs for GRPO/DPO
        logger.info("Creating preference pairs...")
        preference_data = create_preference_pairs(training_data)
        logger.info(f"Created {len(preference_data)} preference pairs")
        
        if len(preference_data) < 10:
            logger.warning("Not enough preference pairs, falling back to SFT")
            trainer = train_sft(
                agent_model, agent_tokenizer, training_data,
                args.output, args.epochs, args.batch_size, args.learning_rate
            )
        elif args.method == "grpo":
            logger.info("Training with GRPO...")
            trainer = train_grpo(
                agent_model, agent_tokenizer, preference_data,
                args.output, args.epochs, args.batch_size, args.learning_rate
            )
        else:  # dpo
            logger.info("Training with DPO...")
            trainer = train_dpo(
                agent_model, agent_tokenizer, preference_data,
                args.output, args.epochs, args.batch_size, args.learning_rate
            )
    
    # Save
    logger.info(f"Saving model to {args.output}")
    trainer.save_model()
    agent_tokenizer.save_pretrained(args.output)
    
    # Save adapter
    adapter_path = args.output / "adapter"
    agent_model.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")
    
    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())
