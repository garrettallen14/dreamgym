#!/bin/bash
# Setup script for RunPod with CUDA 12.4 + Python 3.12

set -e

echo "=== DreamGym RunPod Setup ==="

# Install torch 2.5.0 with CUDA 12.4
echo "Installing torch 2.5.0 (CUDA 12.4)..."
uv pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn from prebuilt wheel (2x faster attention)
echo "Installing flash-attn 2.8.3 (prebuilt wheel)..."
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# Sync remaining deps
echo "Syncing other dependencies..."
uv sync

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run your training with:"
echo "  uv run train-experience-model --use-4bit --gradient-checkpointing --sample-every 100 --output models/experience_model_final"
