"""Prepare training data for the Experience Model.

Converts trajectories into input-output pairs for LLM fine-tuning.

Usage:
    uv run prepare-experience-data --input data/trajectories.jsonl --output data/experience_train.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from webshop_exp.types import ExperienceModelSample, Trajectory, load_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_experience_prompt(instruction: str, state: str, action: str) -> str:
    """Format input prompt for experience model."""
    return f"""Task: {instruction}

Current State:
{state}

Action taken: {action}

Predict the next state and reward."""


def format_experience_completion(next_state: str, reward: float, done: bool) -> str:
    """Format output completion for experience model."""
    return f"""Next State:
{next_state}

Reward: {reward}
Done: {done}"""


def trajectory_to_samples(trajectory: Trajectory) -> list[ExperienceModelSample]:
    """Convert a trajectory into experience model training samples."""
    samples = []
    
    for transition in trajectory.transitions:
        prompt = format_experience_prompt(
            transition.task_instruction,
            transition.state,
            transition.action,
        )
        completion = format_experience_completion(
            transition.next_state,
            transition.reward,
            transition.done,
        )
        samples.append(ExperienceModelSample(prompt=prompt, completion=completion))
    
    return samples


def prepare_experience_data(
    trajectories: list[Trajectory],
    val_split: float = 0.1,
    max_state_length: int = 2000,
) -> tuple[list[ExperienceModelSample], list[ExperienceModelSample]]:
    """Prepare training and validation datasets."""
    
    # Convert all trajectories to samples
    all_samples = []
    for traj in tqdm(trajectories, desc="Converting trajectories"):
        samples = trajectory_to_samples(traj)
        
        # Filter out samples with overly long states
        for sample in samples:
            if len(sample.prompt) <= max_state_length * 2:
                all_samples.append(sample)
    
    logger.info(f"Created {len(all_samples)} training samples")
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_split))
    
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    return train_samples, val_samples


def save_samples(samples: list[ExperienceModelSample], path: Path) -> None:
    """Save samples to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def save_hf_format(samples: list[ExperienceModelSample], path: Path) -> None:
    """Save samples in HuggingFace-compatible format for SFT training."""
    with open(path, "w") as f:
        for sample in samples:
            # Format for SFTTrainer with instruction-following format
            hf_sample = {
                "messages": [
                    {"role": "user", "content": sample.prompt},
                    {"role": "assistant", "content": sample.completion},
                ]
            }
            f.write(json.dumps(hf_sample) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare experience model training data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/trajectories.jsonl"),
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
        help="Validation set fraction",
    )
    parser.add_argument(
        "--max-state-length",
        type=int,
        default=2000,
        help="Maximum state text length",
    )
    parser.add_argument(
        "--hf-format",
        action="store_true",
        help="Also save in HuggingFace chat format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load trajectories
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("Run extract-trajectories first to create the trajectories file.")
        return 1
    
    logger.info(f"Loading trajectories from {args.input}")
    trajectories = load_trajectories(str(args.input))
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Prepare data
    train_samples, val_samples = prepare_experience_data(
        trajectories,
        val_split=args.val_split,
        max_state_length=args.max_state_length,
    )
    
    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = args.output_dir / "experience_train.jsonl"
    val_path = args.output_dir / "experience_val.jsonl"
    
    save_samples(train_samples, train_path)
    save_samples(val_samples, val_path)
    
    logger.info(f"Saved {len(train_samples)} training samples to {train_path}")
    logger.info(f"Saved {len(val_samples)} validation samples to {val_path}")
    
    # Also save HuggingFace format if requested
    if args.hf_format:
        train_hf_path = args.output_dir / "experience_train_hf.jsonl"
        val_hf_path = args.output_dir / "experience_val_hf.jsonl"
        
        save_hf_format(train_samples, train_hf_path)
        save_hf_format(val_samples, val_hf_path)
        
        logger.info(f"Saved HuggingFace format to {train_hf_path} and {val_hf_path}")
    
    # Print sample
    if train_samples:
        print("\n### Sample Training Example ###")
        sample = train_samples[0]
        print(f"PROMPT:\n{sample.prompt[:500]}...")
        print(f"\nCOMPLETION:\n{sample.completion[:500]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())
