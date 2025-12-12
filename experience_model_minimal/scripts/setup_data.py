#!/usr/bin/env python3
"""Setup script to copy sample data from the main dreamgym repository.

Usage:
    python scripts/setup_data.py
    
    # Or with custom paths:
    python scripts/setup_data.py --source ../data --num-samples 1000
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Setup sample data for training")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../data"),
        help="Source data directory (from main dreamgym repo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to copy (None = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Try to find source files
    train_source = args.source / "experience_train_hf.jsonl"
    val_source = args.source / "experience_val_hf.jsonl"
    traj_source = args.source / "trajectories.jsonl"
    
    if train_source.exists():
        print(f"Found training data: {train_source}")
        copy_jsonl(train_source, args.output / "train.jsonl", args.num_samples)
    else:
        print(f"Training data not found at {train_source}")
        print("Creating example data instead...")
        create_example_data(args.output / "train.jsonl", 100)
    
    if val_source.exists():
        print(f"Found validation data: {val_source}")
        num_val = args.num_samples // 10 if args.num_samples else None
        copy_jsonl(val_source, args.output / "val.jsonl", num_val)
    else:
        print(f"Validation data not found at {val_source}")
        create_example_data(args.output / "val.jsonl", 20)
    
    if traj_source.exists():
        print(f"Found trajectories: {traj_source}")
        num_traj = args.num_samples // 5 if args.num_samples else None
        copy_jsonl(traj_source, args.output / "trajectories.jsonl", num_traj)
    
    print(f"\nData setup complete in {args.output}/")


def copy_jsonl(source: Path, dest: Path, num_samples: int = None):
    """Copy JSONL file, optionally sampling."""
    lines = []
    with open(source) as f:
        for line in f:
            if line.strip():
                lines.append(line)
    
    if num_samples and num_samples < len(lines):
        lines = random.sample(lines, num_samples)
    
    with open(dest, "w") as f:
        f.writelines(lines)
    
    print(f"  Copied {len(lines)} samples to {dest}")


def create_example_data(dest: Path, num_samples: int):
    """Create example training data for testing."""
    examples = [
        {
            "instruction": "Find a red leather wallet under $50",
            "state": "[Search Page]\nEnter a search query to find products.",
            "action": "search[red leather wallet]",
            "next_state": "[Search Results]\nProducts:\n- [B001] Red Leather Wallet - $35\n- [B002] Brown Wallet - $25",
            "reward": 0.0,
            "done": False,
        },
        {
            "instruction": "Find a red leather wallet under $50",
            "state": "[Search Results]\nProducts:\n- [B001] Red Leather Wallet - $35",
            "action": "click[B001]",
            "next_state": "[Product Page: B001]\nRed Leather Wallet\nPrice: $35\nOptions: color=red",
            "reward": 0.0,
            "done": False,
        },
        {
            "instruction": "Find a red leather wallet under $50",
            "state": "[Product Page: B001]\nRed Leather Wallet\nPrice: $35",
            "action": "click[Buy Now]",
            "next_state": "[Purchase Complete]\nProduct: B001\nPrice: $35\nReward: 0.95",
            "reward": 0.95,
            "done": True,
        },
    ]
    
    samples = []
    for _ in range(num_samples):
        ex = random.choice(examples)
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Task: {ex['instruction']}\n\nCurrent State:\n{ex['state']}\n\nAction taken: {ex['action']}\n\nPredict the next state and reward."
                },
                {
                    "role": "assistant", 
                    "content": f"Next State:\n{ex['next_state']}\n\nReward: {ex['reward']}\nDone: {ex['done']}"
                }
            ]
        }
        samples.append(sample)
    
    with open(dest, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"  Created {len(samples)} example samples in {dest}")


if __name__ == "__main__":
    main()
