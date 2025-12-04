#!/usr/bin/env python3
"""Run hyperparameter sweep experiments.

Usage:
    # Run quick test first
    uv run python experiments/run_sweep.py --quick-test
    
    # Run a specific sweep
    uv run python experiments/run_sweep.py --sweep lora_rank_sweep
    
    # Run all sweeps in order
    uv run python experiments/run_sweep.py --all
    
    # Run single variation
    uv run python experiments/run_sweep.py --sweep lora_rank_sweep --variation rank_16
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(defaults: dict, variation: dict, output_dir: Path) -> list[str]:
    """Build training command from config."""
    # Merge defaults with variation (variation overrides)
    config = {**defaults, **variation}
    
    cmd = [
        "uv", "run", "train-experience-model",
        "--base-model", config["base_model"],
        "--train-data", config["train_data"],
        "--val-data", config["val_data"],
        "--output", str(output_dir),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--gradient-accumulation", str(config["gradient_accumulation"]),
        "--learning-rate", str(config["learning_rate"]),
        "--lora-rank", str(config["lora_rank"]),
        "--lora-alpha", str(config["lora_alpha"]),
    ]
    
    if config.get("max_seq_length"):
        cmd.extend(["--max-seq-length", str(config["max_seq_length"])])
    
    if config.get("max_steps") and config["max_steps"] > 0:
        cmd.extend(["--max-steps", str(config["max_steps"])])
    
    if config.get("seed"):
        cmd.extend(["--seed", str(config["seed"])])
    
    if config.get("use_4bit", False):
        cmd.append("--use-4bit")
    
    if config.get("wandb_project"):
        cmd.extend(["--wandb-project", config["wandb_project"]])
    else:
        cmd.append("--no-wandb")
    
    return cmd


def run_experiment(
    name: str,
    defaults: dict,
    variation: dict,
    output_base: Path,
    dry_run: bool = False,
) -> dict:
    """Run a single experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"{name}_{timestamp}"
    
    cmd = build_command(defaults, variation, output_dir)
    
    logger.info(f"Running experiment: {name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        logger.info("[DRY RUN] Would execute above command")
        return {"name": name, "status": "dry_run", "output_dir": str(output_dir)}
    
    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump({
            "name": name,
            "defaults": defaults,
            "variation": variation,
            "command": cmd,
            "timestamp": timestamp,
        }, f, indent=2)
    
    # Run training
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        status = "success"
        error = None
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = str(e)
        logger.error(f"Experiment {name} failed: {error}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save results
    results = {
        "name": name,
        "status": status,
        "output_dir": str(output_dir),
        "duration_seconds": duration,
        "error": error,
    }
    
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_sweep(
    sweep_name: str,
    sweep_config: dict,
    defaults: dict,
    output_base: Path,
    variation_filter: str = None,
    dry_run: bool = False,
) -> list[dict]:
    """Run all variations in a sweep."""
    logger.info(f"Running sweep: {sweep_name}")
    logger.info(f"Description: {sweep_config.get('description', 'N/A')}")
    
    results = []
    variations = sweep_config.get("variations", [])
    
    for var in variations:
        var_name = var.get("name", "unnamed")
        
        if variation_filter and var_name != variation_filter:
            continue
        
        full_name = f"{sweep_name}__{var_name}"
        result = run_experiment(
            name=full_name,
            defaults=defaults,
            variation=var,
            output_base=output_base,
            dry_run=dry_run,
        )
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        help="Name of sweep to run",
    )
    parser.add_argument(
        "--variation",
        type=str,
        help="Specific variation within sweep",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all sweeps in priority order",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick validation test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Override output base directory",
    )
    
    args = parser.parse_args()
    
    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    output_base = args.output_base or Path(defaults.get("output_base", "models/experiments"))
    
    all_results = []
    
    # Quick test
    if args.quick_test:
        quick_config = config.get("quick_test", {})
        result = run_experiment(
            name="quick_test",
            defaults=defaults,
            variation=quick_config,
            output_base=output_base,
            dry_run=args.dry_run,
        )
        all_results.append(result)
    
    # Single sweep
    elif args.sweep:
        sweeps = config.get("sweeps", {})
        if args.sweep not in sweeps:
            logger.error(f"Sweep not found: {args.sweep}")
            logger.info(f"Available sweeps: {list(sweeps.keys())}")
            return 1
        
        results = run_sweep(
            sweep_name=args.sweep,
            sweep_config=sweeps[args.sweep],
            defaults=defaults,
            output_base=output_base,
            variation_filter=args.variation,
            dry_run=args.dry_run,
        )
        all_results.extend(results)
    
    # All sweeps
    elif args.all:
        sweeps = config.get("sweeps", {})
        # Sort by priority
        sorted_sweeps = sorted(
            sweeps.items(),
            key=lambda x: x[1].get("priority", 999),
        )
        
        for sweep_name, sweep_config in sorted_sweeps:
            results = run_sweep(
                sweep_name=sweep_name,
                sweep_config=sweep_config,
                defaults=defaults,
                output_base=output_base,
                dry_run=args.dry_run,
            )
            all_results.extend(results)
    
    else:
        parser.print_help()
        return 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    for result in all_results:
        status = result.get("status", "unknown")
        name = result.get("name", "unnamed")
        duration = result.get("duration_seconds", 0)
        
        status_icon = "✓" if status == "success" else "✗" if status == "failed" else "○"
        logger.info(f"{status_icon} {name}: {status} ({duration/60:.1f} min)")
    
    # Save summary
    summary_file = output_base / f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
