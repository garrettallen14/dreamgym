#!/usr/bin/env python3
"""Analyze experiment results and generate comparison report.

Usage:
    uv run python experiments/analyze_results.py --results-dir models/experiments
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_experiment_results(results_dir: Path) -> list[dict]:
    """Load all experiment results from directory."""
    results = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        config_file = exp_dir / "experiment_config.json"
        results_file = exp_dir / "experiment_results.json"
        trainer_state = exp_dir / "trainer_state.json"
        
        if not config_file.exists():
            continue
        
        with open(config_file) as f:
            config = json.load(f)
        
        experiment = {
            "name": config.get("name", exp_dir.name),
            "dir": str(exp_dir),
            **config.get("variation", {}),
        }
        
        # Add training results if available
        if results_file.exists():
            with open(results_file) as f:
                exp_results = json.load(f)
                experiment["status"] = exp_results.get("status")
                experiment["duration_min"] = exp_results.get("duration_seconds", 0) / 60
        
        # Add trainer metrics if available
        if trainer_state.exists():
            with open(trainer_state) as f:
                state = json.load(f)
                experiment["train_loss"] = state.get("log_history", [{}])[-1].get("train_loss")
                experiment["eval_loss"] = state.get("log_history", [{}])[-1].get("eval_loss")
                experiment["total_steps"] = state.get("global_step")
        
        # Check for eval results in log history
        all_logs = exp_dir / "all_results.json"
        if all_logs.exists():
            with open(all_logs) as f:
                log_data = json.load(f)
                experiment["final_eval_loss"] = log_data.get("eval_loss")
        
        results.append(experiment)
    
    return results


def generate_report(results: list[dict], output_path: Path = None):
    """Generate comparison report from results."""
    if not results:
        logger.warning("No results found")
        return
    
    df = pd.DataFrame(results)
    
    # Sort by eval_loss if available, else by name
    if "eval_loss" in df.columns:
        df = df.sort_values("eval_loss", ascending=True)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    
    # Key columns to show
    key_cols = [
        "name", "status", "lora_rank", "learning_rate", 
        "batch_size", "gradient_accumulation",
        "train_loss", "eval_loss", "duration_min"
    ]
    display_cols = [c for c in key_cols if c in df.columns]
    
    print("\n### All Experiments ###")
    print(df[display_cols].to_string(index=False))
    
    # Best by eval loss
    if "eval_loss" in df.columns and df["eval_loss"].notna().any():
        best = df.loc[df["eval_loss"].idxmin()]
        print("\n### Best Configuration (lowest eval_loss) ###")
        print(f"Name: {best['name']}")
        print(f"Eval Loss: {best['eval_loss']:.4f}")
        print(f"Config:")
        for col in ["lora_rank", "lora_alpha", "learning_rate", "batch_size", "gradient_accumulation"]:
            if col in best:
                print(f"  {col}: {best[col]}")
    
    # Save report
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Report saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("models/experiments"),
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for report",
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        return 1
    
    results = load_experiment_results(args.results_dir)
    logger.info(f"Loaded {len(results)} experiment results")
    
    output_path = args.output or args.results_dir / "comparison_report.csv"
    generate_report(results, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
