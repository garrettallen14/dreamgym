"""Prepare training data for the Experience Model.

Converts trajectory data into the chat format used for LLM fine-tuning.

Usage:
    python src/prepare_data.py --input data/trajectories.jsonl --output-dir data/
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.data_types import Trajectory, TrainingSample, load_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

def format_prompt(instruction: str, state: str, action: str) -> str:
    """Format the input prompt for the Experience Model.
    
    This is what the model sees as input. It contains:
    - The task/goal
    - Current state observation
    - Action being taken
    - Instruction to predict the outcome
    """
    return f"""Task: {instruction}

Current State:
{state}

Action taken: {action}

Predict the next state and reward."""


def format_completion(next_state: str, reward: float, done: bool) -> str:
    """Format the expected completion (output) for the Experience Model.
    
    This is what the model should learn to generate:
    - The resulting next state
    - The reward received
    - Whether the episode ended
    """
    return f"""Next State:
{next_state}

Reward: {reward}
Done: {done}"""


# =============================================================================
# Data Conversion
# =============================================================================

def transition_to_sample(transition) -> TrainingSample:
    """Convert a single transition to a training sample."""
    prompt = format_prompt(
        transition.task_instruction,
        transition.state,
        transition.action,
    )
    completion = format_completion(
        transition.next_state,
        transition.reward,
        transition.done,
    )
    return TrainingSample(prompt=prompt, completion=completion)


def trajectory_to_samples(trajectory: Trajectory) -> List[TrainingSample]:
    """Convert a trajectory into training samples.
    
    Each transition in the trajectory becomes one training sample.
    """
    samples = []
    for transition in trajectory.transitions:
        samples.append(transition_to_sample(transition))
    return samples


def prepare_data(
    trajectories: List[Trajectory],
    val_split: float = 0.1,
    max_prompt_length: int = 4000,
    seed: int = 42,
) -> Tuple[List[TrainingSample], List[TrainingSample]]:
    """Convert trajectories to training samples and split into train/val.
    
    Args:
        trajectories: List of trajectory objects
        val_split: Fraction of data to use for validation
        max_prompt_length: Filter out samples with prompts longer than this
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_samples, val_samples)
    """
    random.seed(seed)
    
    # Convert all trajectories to samples
    all_samples = []
    for traj in tqdm(trajectories, desc="Converting trajectories"):
        samples = trajectory_to_samples(traj)
        
        # Filter out samples with overly long prompts
        for sample in samples:
            if len(sample.prompt) <= max_prompt_length:
                all_samples.append(sample)
    
    logger.info(f"Created {len(all_samples)} training samples from {len(trajectories)} trajectories")
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_split))
    
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")
    
    return train_samples, val_samples


# =============================================================================
# Save Functions
# =============================================================================

def save_samples_chat_format(samples: List[TrainingSample], path: Path) -> None:
    """Save samples in HuggingFace chat format for SFTTrainer.
    
    Format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_chat_format()) + "\n")


def save_samples_raw(samples: List[TrainingSample], path: Path) -> None:
    """Save samples in raw prompt/completion format."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare Experience Model training data")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input trajectories JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for training files",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=4000,
        help="Maximum prompt length in characters (default: 4000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Check input exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Load trajectories
    logger.info(f"Loading trajectories from {args.input}")
    trajectories = load_trajectories(str(args.input))
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Prepare data
    train_samples, val_samples = prepare_data(
        trajectories,
        val_split=args.val_split,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
    )
    
    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    
    save_samples_chat_format(train_samples, train_path)
    save_samples_chat_format(val_samples, val_path)
    
    logger.info(f"Saved {len(train_samples)} training samples to {train_path}")
    logger.info(f"Saved {len(val_samples)} validation samples to {val_path}")
    
    # Print example
    if train_samples:
        print("\n" + "=" * 60)
        print("EXAMPLE TRAINING SAMPLE")
        print("=" * 60)
        sample = train_samples[0]
        print(f"PROMPT:\n{sample.prompt[:500]}...")
        print(f"\nCOMPLETION:\n{sample.completion[:500]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())
