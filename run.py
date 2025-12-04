#!/usr/bin/env python3
"""Main runner script that launches dashboard + training together.

Usage:
    # YOLO - just run with optimal defaults (recommended!)
    uv run python run.py yolo
    
    # Start dashboard only
    uv run python run.py dashboard
    
    # Run training with custom args
    uv run python run.py train --use-4bit --gradient-checkpointing --epochs 3
    
    # Run sweep with dashboard
    uv run python run.py sweep --quick-test
    uv run python run.py sweep --all
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def start_dashboard(port: int = 3000) -> subprocess.Popen:
    """Start the dashboard server in background."""
    print(f"\n🚀 Starting dashboard at http://0.0.0.0:{port}")
    print(f"   (On RunPod, use your forwarded port URL)\n")
    
    proc = subprocess.Popen(
        ["uv", "run", "python", "dashboard/server.py", "--port", str(port)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # Give it time to start
    return proc


def run_with_logging(cmd: list[str], log_name: str) -> int:
    """Run command with output logged to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{log_name}_{timestamp}.log"
    
    print(f"📝 Logging to: {log_file}")
    print(f"🏃 Running: {' '.join(cmd)}\n")
    print("=" * 60)
    
    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()
        
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        
        proc.wait()
        
        f.write(f"\n\nFinished: {datetime.now().isoformat()}\n")
        f.write(f"Exit code: {proc.returncode}\n")
    
    print("=" * 60)
    print(f"✅ Finished with exit code: {proc.returncode}")
    print(f"📝 Log saved to: {log_file}")
    
    return proc.returncode


def cmd_dashboard(args):
    """Run dashboard only."""
    proc = subprocess.Popen(
        ["uv", "run", "python", "dashboard/server.py", "--port", str(args.port)],
        cwd=PROJECT_ROOT,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def cmd_train(args):
    """Run training with dashboard."""
    dashboard_proc = start_dashboard(args.port)
    
    cmd = [
        "uv", "run", "train-experience-model",
        "--base-model", args.base_model,
        "--train-data", args.train_data,
        "--val-data", args.val_data,
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
        "--learning-rate", str(args.learning_rate),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
    ]
    
    if args.max_steps > 0:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.use_4bit:
        cmd.append("--use-4bit")
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.early_stopping > 0:
        cmd.extend(["--early-stopping-patience", str(args.early_stopping)])
    if args.sample_every > 0:
        cmd.extend(["--sample-every", str(args.sample_every)])
        cmd.extend(["--num-samples", str(args.num_samples)])
    if args.no_wandb:
        cmd.append("--no-wandb")
    
    try:
        exit_code = run_with_logging(cmd, "train_experience_model")
    finally:
        dashboard_proc.terminate()
    
    return exit_code


def cmd_yolo(args):
    """Run training with optimal defaults - just works!"""
    dashboard_proc = start_dashboard(args.port)
    
    cmd = [
        "uv", "run", "train-experience-model",
        "--base-model", "Qwen/Qwen2.5-3B-Instruct",
        "--output", args.output,
        "--epochs", "3",
        "--batch-size", "8",
        "--gradient-accumulation", "4",
        "--learning-rate", "2e-4",
        "--lora-rank", "16",
        "--lora-alpha", "32",
        "--use-4bit",
        "--gradient-checkpointing",
        "--early-stopping-patience", "2",
        "--sample-every", "100",
        "--num-samples", "3",
        "--no-wandb",
    ]
    
    try:
        exit_code = run_with_logging(cmd, "yolo_train")
    finally:
        dashboard_proc.terminate()
    
    return exit_code


def cmd_sweep(args):
    """Run sweep with dashboard."""
    dashboard_proc = start_dashboard(args.port)
    
    cmd = ["uv", "run", "python", "experiments/run_sweep.py"]
    
    if args.quick_test:
        cmd.append("--quick-test")
    elif args.sweep:
        cmd.extend(["--sweep", args.sweep])
        if args.variation:
            cmd.extend(["--variation", args.variation])
    elif args.all:
        cmd.append("--all")
    
    if args.dry_run:
        cmd.append("--dry-run")
    
    try:
        exit_code = run_with_logging(cmd, f"sweep_{args.sweep or 'all'}")
    finally:
        dashboard_proc.terminate()
    
    return exit_code


def cmd_run(args):
    """Run arbitrary command with dashboard."""
    dashboard_proc = start_dashboard(args.port)
    
    cmd = args.command.split()
    log_name = cmd[2] if len(cmd) > 2 else "custom"
    
    try:
        exit_code = run_with_logging(cmd, log_name)
    finally:
        dashboard_proc.terminate()
    
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="DreamGym Runner")
    parser.add_argument("--port", type=int, default=3000, help="Dashboard port")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dashboard only
    dash_parser = subparsers.add_parser("dashboard", help="Run dashboard only")
    
    # YOLO - just run with optimal defaults
    yolo_parser = subparsers.add_parser("yolo", help="🚀 Run with optimal defaults (recommended)")
    yolo_parser.add_argument("--output", default="models/experience_model_final")
    
    # Training with full control
    train_parser = subparsers.add_parser("train", help="Run training with dashboard")
    train_parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    train_parser.add_argument("--train-data", default="data/experience_train_hf.jsonl")
    train_parser.add_argument("--val-data", default="data/experience_val_hf.jsonl")
    train_parser.add_argument("--output", default="models/experience_model")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--lora-rank", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--max-steps", type=int, default=-1)
    train_parser.add_argument("--use-4bit", action="store_true")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--early-stopping", type=int, default=0, help="Early stopping patience")
    train_parser.add_argument("--sample-every", type=int, default=100, help="Generate samples every N steps")
    train_parser.add_argument("--num-samples", type=int, default=3)
    train_parser.add_argument("--no-wandb", action="store_true")
    
    # Sweep
    sweep_parser = subparsers.add_parser("sweep", help="Run experiment sweep with dashboard")
    sweep_parser.add_argument("--sweep", type=str, help="Sweep name")
    sweep_parser.add_argument("--variation", type=str, help="Specific variation")
    sweep_parser.add_argument("--quick-test", action="store_true")
    sweep_parser.add_argument("--all", action="store_true")
    sweep_parser.add_argument("--dry-run", action="store_true")
    
    # Arbitrary command
    cmd_parser = subparsers.add_parser("cmd", help="Run any command with dashboard")
    cmd_parser.add_argument("command", type=str, help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        return cmd_dashboard(args)
    elif args.command == "yolo":
        return cmd_yolo(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "cmd":
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
