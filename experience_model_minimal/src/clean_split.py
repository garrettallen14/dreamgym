"""Clean train/val split by removing detected leakage.

Applies fixes based on analyze_split.py recommendations:
1. Remove exact duplicates from validation
2. Optionally: Re-split by instruction to eliminate task leakage
3. Optionally: Re-split by entity to eliminate product leakage

Usage:
    # Remove duplicates only
    python src/clean_split.py --train data/train.jsonl --val data/val.jsonl

    # Resplit by instruction (recommended for experience models)
    python src/clean_split.py --train data/train.jsonl --val data/val.jsonl --resplit-by instruction

    # Resplit by entity (products)
    python src/clean_split.py --train data/train.jsonl --val data/val.jsonl --resplit-by entity
"""

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Handle both module and script execution
try:
    from src.analyze_split import extract_entities, extract_instruction, load_samples, Sample
except ImportError:
    from analyze_split import extract_entities, extract_instruction, load_samples, Sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def remove_duplicates(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[List[Sample], List[Sample], int]:
    """Remove exact duplicates from validation set.
    
    Returns:
        (train_samples, cleaned_val_samples, removed_count)
    """
    train_hashes = {s.full_hash for s in train_samples}
    
    cleaned_val = []
    removed = 0
    
    for sample in val_samples:
        if sample.full_hash in train_hashes:
            removed += 1
        else:
            cleaned_val.append(sample)
    
    return train_samples, cleaned_val, removed


def remove_prompt_duplicates(
    train_samples: List[Sample],
    val_samples: List[Sample],
) -> Tuple[List[Sample], List[Sample], int]:
    """Remove samples with duplicate prompts from validation set.
    
    Returns:
        (train_samples, cleaned_val_samples, removed_count)
    """
    train_prompt_hashes = {s.prompt_hash for s in train_samples}
    
    cleaned_val = []
    removed = 0
    
    for sample in val_samples:
        if sample.prompt_hash in train_prompt_hashes:
            removed += 1
        else:
            cleaned_val.append(sample)
    
    return train_samples, cleaned_val, removed


def resplit_by_instruction(
    train_samples: List[Sample],
    val_samples: List[Sample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Re-split data ensuring no instruction overlap.
    
    Groups samples by instruction, then splits groups.
    This prevents task memorization leakage.
    
    Returns:
        (new_train_samples, new_val_samples)
    """
    random.seed(seed)
    
    # Combine all samples
    all_samples = train_samples + val_samples
    
    # Group by instruction
    by_instruction: Dict[str, List[Sample]] = defaultdict(list)
    for sample in all_samples:
        by_instruction[sample.instruction].append(sample)
    
    # Shuffle instruction groups
    instructions = list(by_instruction.keys())
    random.shuffle(instructions)
    
    # Split groups (not individual samples)
    total_samples = len(all_samples)
    target_val = int(total_samples * val_ratio)
    
    new_train = []
    new_val = []
    val_count = 0
    
    for instruction in instructions:
        samples = by_instruction[instruction]
        
        if val_count < target_val:
            new_val.extend(samples)
            val_count += len(samples)
        else:
            new_train.extend(samples)
    
    logger.info(f"Re-split by instruction: {len(new_train)} train, {len(new_val)} val")
    logger.info(f"Train instructions: {len(set(s.instruction for s in new_train))}")
    logger.info(f"Val instructions: {len(set(s.instruction for s in new_val))}")
    
    return new_train, new_val


def resplit_by_entity(
    train_samples: List[Sample],
    val_samples: List[Sample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Re-split data ensuring no entity (ASIN) overlap.
    
    Groups samples by primary entity, then splits groups.
    This prevents product-specific memorization.
    
    Returns:
        (new_train_samples, new_val_samples)
    """
    random.seed(seed)
    
    # Combine all samples
    all_samples = train_samples + val_samples
    
    # Group by primary entity (first ASIN found)
    by_entity: Dict[str, List[Sample]] = defaultdict(list)
    no_entity = []
    
    for sample in all_samples:
        if sample.entities:
            primary_entity = sorted(sample.entities)[0]  # Deterministic selection
            by_entity[primary_entity].append(sample)
        else:
            no_entity.append(sample)
    
    # Shuffle entity groups
    entities = list(by_entity.keys())
    random.shuffle(entities)
    
    # Split groups
    total_samples = len(all_samples)
    target_val = int(total_samples * val_ratio)
    
    new_train = []
    new_val = []
    val_count = 0
    
    for entity in entities:
        samples = by_entity[entity]
        
        if val_count < target_val:
            new_val.extend(samples)
            val_count += len(samples)
        else:
            new_train.extend(samples)
    
    # Add samples without entities to train
    new_train.extend(no_entity)
    
    logger.info(f"Re-split by entity: {len(new_train)} train, {len(new_val)} val")
    
    return new_train, new_val


def save_samples(samples: List[Sample], path: Path) -> None:
    """Save samples to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.raw) + "\n")
    logger.info(f"Saved {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean train/val split to reduce overfitting risk"
    )
    
    parser.add_argument(
        "--train", type=Path, required=True,
        help="Path to training data (JSONL)"
    )
    parser.add_argument(
        "--val", type=Path, required=True,
        help="Path to validation data (JSONL)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same as input with _clean suffix)"
    )
    parser.add_argument(
        "--resplit-by", choices=["instruction", "entity", "none"], default="none",
        help="Re-split strategy to use"
    )
    parser.add_argument(
        "--remove-prompt-duplicates", action="store_true",
        help="Also remove samples with duplicate prompts (more aggressive)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Target validation ratio for re-splitting (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for re-splitting"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading training data: {args.train}")
    train_samples = load_samples(args.train)
    logger.info(f"Loaded {len(train_samples)} training samples")
    
    logger.info(f"Loading validation data: {args.val}")
    val_samples = load_samples(args.val)
    logger.info(f"Loaded {len(val_samples)} validation samples")
    
    # Apply cleaning
    if args.resplit_by == "instruction":
        logger.info("Re-splitting by instruction...")
        train_samples, val_samples = resplit_by_instruction(
            train_samples, val_samples, args.val_ratio, args.seed
        )
    elif args.resplit_by == "entity":
        logger.info("Re-splitting by entity...")
        train_samples, val_samples = resplit_by_entity(
            train_samples, val_samples, args.val_ratio, args.seed
        )
    else:
        # Just remove duplicates
        logger.info("Removing exact duplicates...")
        train_samples, val_samples, removed = remove_duplicates(
            train_samples, val_samples
        )
        logger.info(f"Removed {removed} exact duplicates from validation")
        
        if args.remove_prompt_duplicates:
            logger.info("Removing prompt duplicates...")
            train_samples, val_samples, removed = remove_prompt_duplicates(
                train_samples, val_samples
            )
            logger.info(f"Removed {removed} prompt duplicates from validation")
    
    # Determine output paths
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.train.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_out = output_dir / "train_clean.jsonl"
    val_out = output_dir / "val_clean.jsonl"
    
    # Save
    save_samples(train_samples, train_out)
    save_samples(val_samples, val_out)
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  Original: {len(train_samples)} train, {len(val_samples)} val")
    print(f"  Output:   {train_out}")
    print(f"            {val_out}")
    print("\nRun analyze_split.py on cleaned data to verify improvement.")
    
    return 0


if __name__ == "__main__":
    exit(main())
