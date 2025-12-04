"""Fast synthetic trajectory generation using vLLM batched inference.

10-50x faster than HuggingFace generate().

Usage:
    uv run python -m webshop_exp.scripts.generate_synthetic_fast \
        --model models/experiments/yolo_final/adapter \
        --num-trajectories 1000
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from webshop_exp.types import Trajectory, Transition, save_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vllm_model(adapter_path: Path, base_model: str = None):
    """Load model with vLLM for fast batched inference."""
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    
    # Find base model from adapter config
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists() and base_model is None:
        with open(adapter_config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")
    
    if base_model is None:
        base_model = "Qwen/Qwen2.5-3B-Instruct"
    
    logger.info(f"Loading vLLM with base model: {base_model}")
    logger.info(f"LoRA adapter: {adapter_path}")
    
    # Load vLLM with LoRA support
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    
    # Create LoRA request
    lora_request = LoRARequest("experience_model", 1, str(adapter_path))
    
    return llm, lora_request


def batch_generate(
    llm,
    lora_request,
    prompts: List[str],
    max_tokens: int = 128,
) -> List[str]:
    """Generate completions for a batch of prompts."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,  # Greedy
        stop=["\n\nTask:", "\n\nCurrent State:"],  # Stop at next example
    )
    
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    return [output.outputs[0].text for output in outputs]


def format_prompt(instruction: str, state: str, action: str) -> str:
    """Format prompt for experience model."""
    return f"""Task: {instruction}

Current State:
{state}

Action taken: {action}

Predict the next state and reward."""


def parse_output(output: str) -> Tuple[str, float, bool]:
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
        if reward == 0:
            reward = 0.5  # Assume some reward for completion
    
    return next_state, reward, done


def extract_keywords(instruction: str) -> List[str]:
    """Extract search keywords from instruction."""
    stop_words = {
        "find", "me", "a", "an", "the", "with", "that", "is", "are", "for",
        "and", "or", "under", "over", "less", "more", "than", "please", "i",
        "want", "need", "looking", "search", "get", "buy", "purchase",
    }
    words = instruction.lower().split()
    return [w for w in words if w not in stop_words and len(w) > 2]


def select_action(instruction: str, state: str, step: int) -> str:
    """Simple heuristic policy to select actions."""
    keywords = extract_keywords(instruction)
    
    # First step: search
    if step == 0:
        query = " ".join(keywords[:4]) if keywords else "product"
        return f"search[{query}]"
    
    # Look for buy action
    if step > 2 and ("product page" in state.lower() or "buy now" in state.lower()):
        if random.random() > 0.3:  # 70% chance to buy if on product page
            return "click[Buy Now]"
    
    # Click on products
    product_ids = re.findall(r"\[([A-Z0-9]{10,})\]", state)
    if product_ids:
        return f"click[{random.choice(product_ids)}]"
    
    # Default: search again with different keywords
    if keywords:
        query = " ".join(random.sample(keywords, min(3, len(keywords))))
        return f"search[{query}]"
    
    return "search[product]"


def generate_initial_state(instruction: str) -> str:
    """Generate template initial state."""
    return f"""Instruction:
{instruction}

[Search]
Enter a search query to find products matching your requirements.

Available actions: search[query]"""


def generate_trajectories_batched(
    llm,
    lora_request,
    instructions: List[str],
    num_trajectories: int,
    max_steps: int = 6,
    batch_size: int = 32,
) -> List[Trajectory]:
    """Generate trajectories with batched inference."""
    
    trajectories = []
    
    # Process in batches
    for batch_start in tqdm(range(0, num_trajectories, batch_size), desc="Generating batches"):
        batch_end = min(batch_start + batch_size, num_trajectories)
        current_batch_size = batch_end - batch_start
        
        # Initialize batch of trajectories
        batch_instructions = [random.choice(instructions) for _ in range(current_batch_size)]
        batch_states = [generate_initial_state(inst) for inst in batch_instructions]
        batch_transitions = [[] for _ in range(current_batch_size)]
        batch_done = [False] * current_batch_size
        
        # Run steps
        for step in range(max_steps):
            # Collect prompts for active trajectories
            active_indices = [i for i in range(current_batch_size) if not batch_done[i]]
            
            if not active_indices:
                break
            
            # Select actions
            actions = [
                select_action(batch_instructions[i], batch_states[i], step)
                for i in active_indices
            ]
            
            # Format prompts
            prompts = [
                format_prompt(batch_instructions[i], batch_states[i], actions[j])
                for j, i in enumerate(active_indices)
            ]
            
            # Batch generate
            outputs = batch_generate(llm, lora_request, prompts)
            
            # Process outputs
            for j, i in enumerate(active_indices):
                next_state, reward, done = parse_output(outputs[j])
                
                # Record transition
                batch_transitions[i].append(Transition(
                    task_instruction=batch_instructions[i],
                    state=batch_states[i],
                    action=actions[j],
                    next_state=next_state,
                    reward=reward,
                    done=done,
                ))
                
                batch_states[i] = next_state
                batch_done[i] = done
        
        # Create trajectory objects
        for i in range(current_batch_size):
            if batch_transitions[i]:
                total_reward = batch_transitions[i][-1].reward
                trajectories.append(Trajectory(
                    instruction=batch_instructions[i],
                    transitions=batch_transitions[i],
                    total_reward=total_reward,
                    metadata={"synthetic": True, "num_steps": len(batch_transitions[i])},
                ))
    
    return trajectories


def load_instructions(path: Path) -> List[str]:
    """Load instructions from file."""
    if path and path.exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    
    # Default instructions
    return [
        "Find me a black leather wallet under $50 with RFID blocking",
        "I need a laptop stand for my desk that is adjustable",
        "Search for wireless bluetooth headphones with noise cancellation under $100",
        "Find a red dress for a formal event, size medium",
        "I'm looking for a stainless steel water bottle, 32 oz",
    ]


def main():
    parser = argparse.ArgumentParser(description="Fast synthetic generation with vLLM")
    parser.add_argument("--model", type=Path, required=True, help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--instructions", type=Path, default=None)
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_trajectories.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load model
    llm, lora_request = load_vllm_model(args.model, args.base_model)
    
    # Load instructions
    instructions = load_instructions(args.instructions)
    logger.info(f"Loaded {len(instructions)} instructions")
    
    # Generate
    logger.info(f"Generating {args.num_trajectories} trajectories (batch_size={args.batch_size})")
    trajectories = generate_trajectories_batched(
        llm=llm,
        lora_request=lora_request,
        instructions=instructions,
        num_trajectories=args.num_trajectories,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories(trajectories, str(args.output))
    
    # Stats
    total_transitions = sum(len(t.transitions) for t in trajectories)
    avg_steps = total_transitions / len(trajectories) if trajectories else 0
    avg_reward = sum(t.total_reward for t in trajectories) / len(trajectories) if trajectories else 0
    
    logger.info(f"Generated {len(trajectories)} trajectories")
    logger.info(f"Total transitions: {total_transitions}")
    logger.info(f"Avg steps: {avg_steps:.1f}")
    logger.info(f"Avg reward: {avg_reward:.3f}")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
