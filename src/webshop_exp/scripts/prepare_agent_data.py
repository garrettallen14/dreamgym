"""Prepare training data for the agent.

Converts trajectories into action prediction training samples.

Usage:
    uv run prepare-agent-data --input data/trajectories.jsonl --output data/agent_train_real.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from webshop_exp.types import AgentTrainingSample, Trajectory, load_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_agent_prompt(instruction: str, state: str, available_actions: list[str] = None) -> str:
    """Format input prompt for agent."""
    prompt = f"""Task: {instruction}

Current observation:
{state}"""
    
    if available_actions:
        prompt += f"\n\nAvailable actions: {', '.join(available_actions)}"
    
    prompt += "\n\nWhat action should you take?"
    
    return prompt


def trajectory_to_samples(
    trajectory: Trajectory,
    include_available_actions: bool = False,
) -> list[AgentTrainingSample]:
    """Convert a trajectory into agent training samples."""
    samples = []
    
    for transition in trajectory.transitions:
        # Skip transitions with empty states or actions
        if not transition.state or not transition.action:
            continue
        
        # Optionally extract available actions from state
        available_actions = None
        if include_available_actions:
            available_actions = extract_available_actions(transition.state)
        
        prompt = format_agent_prompt(
            transition.task_instruction,
            transition.state,
            available_actions,
        )
        
        samples.append(AgentTrainingSample(
            prompt=prompt,
            completion=transition.action,
        ))
    
    return samples


def extract_available_actions(state: str) -> list[str]:
    """Extract available actions from state observation."""
    import re
    
    actions = []
    
    # Look for explicit available actions line
    match = re.search(r"Available actions?:\s*(.+?)(?:\n|$)", state, re.IGNORECASE)
    if match:
        actions_str = match.group(1)
        # Split by comma or common separators
        actions = [a.strip() for a in re.split(r"[,;]", actions_str) if a.strip()]
    
    return actions


def prepare_agent_data(
    trajectories: list[Trajectory],
    val_split: float = 0.1,
    max_prompt_length: int = 2000,
    include_available_actions: bool = False,
    filter_by_reward: float = None,
) -> tuple[list[AgentTrainingSample], list[AgentTrainingSample]]:
    """Prepare training and validation datasets."""
    
    # Optionally filter by reward
    if filter_by_reward is not None:
        trajectories = [t for t in trajectories if t.total_reward >= filter_by_reward]
        logger.info(f"Filtered to {len(trajectories)} trajectories with reward >= {filter_by_reward}")
    
    # Convert all trajectories to samples
    all_samples = []
    for traj in tqdm(trajectories, desc="Converting trajectories"):
        samples = trajectory_to_samples(traj, include_available_actions)
        
        # Filter out samples with overly long prompts
        for sample in samples:
            if len(sample.prompt) <= max_prompt_length:
                all_samples.append(sample)
    
    logger.info(f"Created {len(all_samples)} training samples")
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_split))
    
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    return train_samples, val_samples


def save_samples(samples: list[AgentTrainingSample], path: Path) -> None:
    """Save samples to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def save_hf_format(samples: list[AgentTrainingSample], path: Path) -> None:
    """Save samples in HuggingFace-compatible format for SFT training."""
    with open(path, "w") as f:
        for sample in samples:
            hf_sample = {
                "messages": [
                    {"role": "user", "content": sample.prompt},
                    {"role": "assistant", "content": sample.completion},
                ]
            }
            f.write(json.dumps(hf_sample) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare agent training data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/trajectories.jsonl"),
        help="Input trajectories JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/agent_train_real.jsonl"),
        help="Output training file",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Output validation file (default: derived from --output)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set fraction",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=2000,
        help="Maximum prompt text length",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Minimum trajectory reward to include",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include available actions in prompt",
    )
    parser.add_argument(
        "--hf-format",
        action="store_true",
        help="Save in HuggingFace chat format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Derive validation output path if not specified
    if args.val_output is None:
        stem = args.output.stem.replace("_train", "")
        args.val_output = args.output.parent / f"{stem}_val.jsonl"
    
    # Load trajectories
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    logger.info(f"Loading trajectories from {args.input}")
    trajectories = load_trajectories(str(args.input))
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Prepare data
    train_samples, val_samples = prepare_agent_data(
        trajectories,
        val_split=args.val_split,
        max_prompt_length=args.max_prompt_length,
        include_available_actions=args.include_actions,
        filter_by_reward=args.min_reward,
    )
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if args.hf_format:
        save_hf_format(train_samples, args.output)
        save_hf_format(val_samples, args.val_output)
    else:
        save_samples(train_samples, args.output)
        save_samples(val_samples, args.val_output)
    
    logger.info(f"Saved {len(train_samples)} training samples to {args.output}")
    logger.info(f"Saved {len(val_samples)} validation samples to {args.val_output}")
    
    # Print sample
    if train_samples:
        print("\n### Sample Training Example ###")
        sample = train_samples[0]
        print(f"PROMPT:\n{sample.prompt[:500]}...")
        print(f"\nCOMPLETION: {sample.completion}")
    
    return 0


if __name__ == "__main__":
    exit(main())
