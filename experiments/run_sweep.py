#!/usr/bin/env python3
"""Run hyperparameter sweep experiments with full persistence.

All data is saved to disk regardless of dashboard/terminal state:
- experiments/results/sweep_results.json   (master results tracker)
- models/experiments/<name>/               (per-experiment data)
    - experiment_config.json
    - experiment_results.json  
    - training.log
    - trainer_state.json (from HF Trainer)

Usage:
    # Always run quick-test first!
    uv run python experiments/run_sweep.py --quick-test
    
    # Run specific sweep
    uv run python experiments/run_sweep.py --sweep lora_rank_sweep
    
    # Run ALL sweeps in priority order
    uv run python experiments/run_sweep.py --all
    
    # Dry run (show what would run)
    uv run python experiments/run_sweep.py --all --dry-run
    
    # Run single variation
    uv run python experiments/run_sweep.py --sweep lora_rank_sweep --variation rank_16
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MASTER_RESULTS_FILE = RESULTS_DIR / "sweep_results.json"


def log(msg: str, level: str = "INFO"):
    """Print and flush immediately."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_master_results() -> dict:
    """Load or create master results tracker."""
    if MASTER_RESULTS_FILE.exists():
        with open(MASTER_RESULTS_FILE) as f:
            return json.load(f)
    return {"experiments": [], "sweeps": {}}


def save_master_results(results: dict):
    """Save master results tracker."""
    with open(MASTER_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def build_command(defaults: dict, variation: dict, output_dir: Path) -> list[str]:
    """Build training command from config."""
    config = {**defaults, **variation}
    
    cmd = [
        "uv", "run", "train-experience-model",
        "--base-model", str(config["base_model"]),
        "--train-data", str(config["train_data"]),
        "--val-data", str(config["val_data"]),
        "--output", str(output_dir),
        "--epochs", str(config.get("epochs", 2)),
        "--batch-size", str(config["batch_size"]),
        "--gradient-accumulation", str(config["gradient_accumulation"]),
        "--learning-rate", str(config["learning_rate"]),
        "--lora-rank", str(config["lora_rank"]),
        "--lora-alpha", str(config["lora_alpha"]),
    ]
    
    if config.get("max_seq_length"):
        cmd.extend(["--max-seq-length", str(config["max_seq_length"])])
    
    if config.get("max_steps") and int(config["max_steps"]) > 0:
        cmd.extend(["--max-steps", str(config["max_steps"])])
    
    if config.get("seed"):
        cmd.extend(["--seed", str(config["seed"])])
    
    if config.get("use_4bit", False):
        cmd.append("--use-4bit")
    
    # Gradient checkpointing (off by default for A40 48GB)
    if config.get("gradient_checkpointing", False):
        cmd.append("--gradient-checkpointing")
    
    # Early stopping
    if config.get("early_stopping_patience") and int(config["early_stopping_patience"]) > 0:
        cmd.extend(["--early-stopping-patience", str(config["early_stopping_patience"])])
    
    # Always disable wandb for sweeps (use our own tracking)
    cmd.append("--no-wandb")
    
    return cmd


def run_experiment(
    name: str,
    defaults: dict,
    variation: dict,
    output_base: Path,
    dry_run: bool = False,
) -> dict:
    """Run a single experiment with full persistence."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"{name}_{timestamp}"
    
    # Prepare result dict
    result = {
        "name": name,
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "status": "pending",
        "config": {**defaults, **variation},
    }
    
    cmd = build_command(defaults, variation, output_dir)
    result["command"] = cmd
    
    log(f"Experiment: {name}")
    log(f"Output: {output_dir}")
    log(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        log("[DRY RUN] Would execute above command", "WARN")
        result["status"] = "dry_run"
        return result
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config immediately
    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Run with logging to file AND stdout
    log_file = output_dir / "training.log"
    start_time = datetime.now()
    result["start_time"] = start_time.isoformat()
    
    log(f"Starting training (logging to {log_file})")
    print("=" * 70, flush=True)
    
    try:
        with open(log_file, "w") as f:
            # Write header
            f.write(f"Experiment: {name}\n")
            f.write(f"Started: {start_time.isoformat()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 70 + "\n\n")
            f.flush()
            
            # Run process
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Stream output to both file and stdout
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            
            proc.wait()
            exit_code = proc.returncode
            
            # Write footer
            end_time = datetime.now()
            f.write(f"\n\n{'=' * 70}\n")
            f.write(f"Finished: {end_time.isoformat()}\n")
            f.write(f"Duration: {(end_time - start_time).total_seconds():.1f}s\n")
            f.write(f"Exit code: {exit_code}\n")
        
        result["status"] = "success" if exit_code == 0 else "failed"
        result["exit_code"] = exit_code
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        log(f"Error: {e}", "ERROR")
    
    end_time = datetime.now()
    result["end_time"] = end_time.isoformat()
    result["duration_seconds"] = (end_time - start_time).total_seconds()
    
    print("=" * 70, flush=True)
    log(f"Finished: {result['status']} ({result['duration_seconds']:.1f}s)")
    
    # Save results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Extract metrics from trainer_state if available
    trainer_state_file = output_dir / "trainer_state.json"
    if trainer_state_file.exists():
        try:
            with open(trainer_state_file) as f:
                state = json.load(f)
            result["final_step"] = state.get("global_step")
            result["best_metric"] = state.get("best_metric")
            # Get final losses from log history
            log_history = state.get("log_history", [])
            if log_history:
                last_log = log_history[-1]
                result["final_loss"] = last_log.get("loss") or last_log.get("train_loss")
                result["final_eval_loss"] = last_log.get("eval_loss")
        except:
            pass
    
    # Update master results
    master = load_master_results()
    master["experiments"].append(result)
    master["last_updated"] = datetime.now().isoformat()
    save_master_results(master)
    
    return result


def run_sweep(
    sweep_name: str,
    sweep_config: dict,
    defaults: dict,
    output_base: Path,
    variation_filter: Optional[str] = None,
    dry_run: bool = False,
) -> list[dict]:
    """Run all variations in a sweep."""
    log(f"=" * 70)
    log(f"SWEEP: {sweep_name}")
    log(f"Description: {sweep_config.get('description', 'N/A')}")
    log(f"=" * 70)
    
    results = []
    variations = sweep_config.get("variations", [])
    total = len(variations)
    
    for i, var in enumerate(variations):
        var_name = var.get("name", f"var_{i}")
        
        if variation_filter and var_name != variation_filter:
            continue
        
        log(f"\n[{i+1}/{total}] Running variation: {var_name}")
        
        full_name = f"{sweep_name}__{var_name}"
        result = run_experiment(
            name=full_name,
            defaults=defaults,
            variation=var,
            output_base=output_base,
            dry_run=dry_run,
        )
        results.append(result)
        
        # Save sweep progress
        master = load_master_results()
        if sweep_name not in master["sweeps"]:
            master["sweeps"][sweep_name] = {"started": datetime.now().isoformat(), "results": []}
        master["sweeps"][sweep_name]["results"].append(result)
        master["sweeps"][sweep_name]["last_updated"] = datetime.now().isoformat()
        save_master_results(master)
    
    return results


def print_summary(all_results: list[dict]):
    """Print final summary."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT SUMMARY", flush=True)
    print("=" * 70, flush=True)
    
    for result in all_results:
        status = result.get("status", "unknown")
        name = result.get("name", "unnamed")
        duration = result.get("duration_seconds", 0)
        loss = result.get("final_loss") or result.get("final_eval_loss")
        
        icon = "✓" if status == "success" else "✗" if status in ["failed", "error"] else "○"
        loss_str = f" | loss={loss:.4f}" if loss else ""
        print(f"  {icon} {name}: {status} ({duration/60:.1f}min){loss_str}", flush=True)
    
    # Results location
    print(f"\nResults saved to:", flush=True)
    print(f"  Master: {MASTER_RESULTS_FILE}", flush=True)
    print(f"  Per-experiment: models/experiments/<name>/", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps with full persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick-test                    # Sanity check (always run first!)
  %(prog)s --sweep lora_rank_sweep         # Run one sweep
  %(prog)s --sweep lr_sweep --variation lr_2e4  # Run single variation
  %(prog)s --all                           # Run all sweeps in order
  %(prog)s --all --dry-run                 # Show what would run
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
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
        help="Run ALL sweeps in priority order",
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
        log(f"Config file not found: {args.config}", "ERROR")
        return 1
    
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    output_base = args.output_base or Path(defaults.get("output_base", "models/experiments"))
    output_base = PROJECT_ROOT / output_base
    
    all_results = []
    
    # Quick test
    if args.quick_test:
        log("Running quick test...")
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
            log(f"Sweep not found: {args.sweep}", "ERROR")
            log(f"Available sweeps: {list(sweeps.keys())}")
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
        
        total_sweeps = len(sorted_sweeps)
        for i, (sweep_name, sweep_config) in enumerate(sorted_sweeps):
            log(f"\n{'#' * 70}")
            log(f"# SWEEP {i+1}/{total_sweeps}: {sweep_name}")
            log(f"{'#' * 70}")
            
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
    print_summary(all_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
