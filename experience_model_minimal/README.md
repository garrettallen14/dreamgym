# Experience Model Training - Minimal Implementation

A minimal, self-contained implementation for studying how to train **Experience Models** - LLMs that learn to predict environment state transitions (state, action → next_state, reward).

Based on the [DreamGym framework](https://arxiv.org/abs/2511.03773).

## Concept

An Experience Model learns the dynamics of an environment:
```
Input:  (current_state, action) 
Output: (next_state, reward, done)
```

Once trained, it can generate synthetic trajectories for training agents without expensive real environment interactions.

## Prerequisites

### Install uv (Fast Python Package Manager)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify installation
uv --version
```

### Setup Project

```bash
cd experience_model_minimal

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

uv pip install -e .
```

---

## Complete Workflow

The recommended workflow ensures data quality before training:

```
Prepare Data → Analyze Split → Clean Leakage → Verify Clean → Train Model
```

### Step 1: Prepare Training Data

Convert raw trajectories to chat format for LLM fine-tuning:

```bash
uv run python src/prepare_data.py \
    --input data/trajectories.jsonl \
    --output-dir data/ \
    --val-split 0.1
```

### Step 2: Analyze Data Quality

Detect leakage and overfitting risks in the train/val split:

```bash
uv run python src/analyze_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --output data/leakage_report.json
```

### Step 3: Clean Data (If Needed)

If risk score is HIGH or CRITICAL, clean the split:

```bash
# Option A: Re-split by instruction (recommended - prevents task memorization)
uv run python src/clean_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --resplit-by instruction

# Option B: Re-split by entity (prevents product memorization)
uv run python src/clean_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --resplit-by entity

# Option C: Just remove duplicates
uv run python src/clean_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl
```

### Step 4: Verify Cleaned Data

Re-run analysis to confirm improvement:

```bash
uv run python src/analyze_split.py \
    --train data/train_clean.jsonl \
    --val data/val_clean.jsonl
```

**Target**: Risk score < 15 (LOW or MINIMAL)

### Step 5: Train the Experience Model

```bash
uv run python src/train.py \
    --train-data data/train_clean.jsonl \
    --val-data data/val_clean.jsonl \
    --output models/experience_model \
    --epochs 3 \
    --use-4bit
```

### Step 6: Validate the Trained Model

```bash
uv run python src/validate.py \
    --model models/experience_model \
    --data data/val_clean.jsonl \
    --num-samples 50
```

## Data Format

### Input: Trajectories (JSONL)
```json
{
  "instruction": "Find a red leather wallet under $50",
  "transitions": [
    {
      "task_instruction": "Find a red leather wallet under $50",
      "state": "[Search Page] Enter query...",
      "action": "search[red leather wallet]",
      "next_state": "[Results] Product A, Product B...",
      "reward": 0.0,
      "done": false
    }
  ],
  "total_reward": 0.85
}
```

### Training Format: Chat Messages (JSONL)
```json
{
  "messages": [
    {"role": "user", "content": "Task: Find a red leather wallet...\n\nCurrent State:\n[Search Page]...\n\nAction taken: search[red leather wallet]\n\nPredict the next state and reward."},
    {"role": "assistant", "content": "Next State:\n[Results] Product A...\n\nReward: 0.0\nDone: False"}
  ]
}
```

## Training Configuration

Key hyperparameters (see `config/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | Qwen/Qwen2.5-3B-Instruct | Base LLM to fine-tune |
| `lora_rank` | 16 | LoRA rank (8, 16, 32 common) |
| `lora_alpha` | 32 | LoRA alpha (typically 2×rank) |
| `learning_rate` | 2e-4 | Learning rate |
| `batch_size` | 8 | Per-device batch size |
| `gradient_accumulation` | 4 | Effective batch = 32 |
| `epochs` | 3 | Training epochs |
| `use_4bit` | true | QLoRA quantization |

## Project Structure

```
experience_model_minimal/
├── README.md
├── pyproject.toml
├── config/
│   └── default.yaml          # Hyperparameter defaults
├── data/
│   ├── trajectories.jsonl    # Raw trajectory data
│   ├── train.jsonl           # Training data (chat format)
│   └── val.jsonl             # Validation data
├── models/                   # Saved models
└── src/
    ├── types.py              # Data structures
    ├── prepare_data.py       # Data preparation
    ├── train.py              # Training script
    └── validate.py           # Validation & metrics
```

## Key Concepts

### 1. LoRA Fine-tuning
We use Low-Rank Adaptation (LoRA) to efficiently fine-tune large models:
- Only ~1% of parameters are trained
- Much lower memory requirements
- Fast training with minimal quality loss

### 2. QLoRA (4-bit Quantization)
For even lower memory usage:
- Base model quantized to 4-bit
- LoRA adapters trained in full precision
- Enables 3B models on 24GB GPUs

### 3. Chat Template Format
Training data uses the model's native chat format:
- User message: state + action + prompt
- Assistant message: predicted next state + reward

## Metrics

| Metric | What it measures |
|--------|------------------|
| State BLEU | Text similarity of predicted vs actual next_state |
| Reward MSE | Accuracy of reward predictions |
| Done F1 | Accuracy of episode termination detection |

## Hardware Requirements

- **Minimum**: 24GB VRAM (with 4-bit quantization)
- **Recommended**: 40GB+ VRAM (A40, A100)
- Training time: ~2-4 hours for 3B model on A40

---

## Data Quality Analysis

This repo includes rigorous tools for detecting train/val data leakage and overfitting risk.

### Analyze Split Quality

```bash
# Basic analysis
python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl

# With semantic similarity (requires sentence-transformers)
python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl --semantic

# Save detailed report
python src/analyze_split.py --train data/train.jsonl --val data/val.jsonl --output report.json
```

### Metrics Computed

| Category | Metrics | Risk Level |
|----------|---------|------------|
| **Exact Duplicates** | Full/prompt/completion duplicates | Critical |
| **Instruction Leakage** | Same tasks in train & val | High |
| **Entity Leakage** | Same products (ASINs) in both | Medium-High |
| **N-gram Overlap** | Jaccard, coverage, novelty for n=1,2,3,5 | Medium |
| **Distribution** | KL divergence, JS divergence, vocab overlap | Diagnostic |
| **Compression** | Normalized Compression Distance | Diagnostic |
| **Semantic** | Embedding similarity distribution | Medium |

### Clean Leaky Splits

```bash
# Remove duplicates only
python src/clean_split.py --train data/train.jsonl --val data/val.jsonl

# Re-split by instruction (prevents task memorization)
python src/clean_split.py --train data/train.jsonl --val data/val.jsonl --resplit-by instruction

# Re-split by entity (prevents product memorization)
python src/clean_split.py --train data/train.jsonl --val data/val.jsonl --resplit-by entity
```

### Risk Score Interpretation

| Score | Level | Action |
|-------|-------|--------|
| 0-5 | MINIMAL | Proceed with training |
| 5-15 | LOW | Monitor for overfitting |
| 15-30 | MODERATE | Consider re-splitting |
| 30-50 | HIGH | Re-split recommended |
| 50+ | CRITICAL | Re-split required |

See [docs/data_leakage_theory.md](docs/data_leakage_theory.md) for full theoretical background.

---

## References

1. [DreamGym: Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773)
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
