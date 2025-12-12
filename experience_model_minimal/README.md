# Experience Model Training

Train an LLM to predict environment transitions: **(state, action) → (next_state, reward, done)**

Based on [DreamGym](https://arxiv.org/abs/2511.03773).

---

## What This Is

An **Experience Model** learns environment dynamics. Once trained, it generates synthetic trajectories for RL agents without real environment calls.

```
Input:  Current state + Action taken
Output: Next state + Reward + Done flag
```

---

## Data Format

Each sample is a chat message pair:

**User (input):**
```
Task: Find a red leather wallet under $50

Current State:
[Search Page] Enter query...

Action taken: search[red leather wallet]

Predict the next state and reward.
```

**Assistant (target):**
```
Next State:
[Results] Product A, Product B...

Reward: 0.0
Done: False
```

---

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
cd experience_model_minimal
uv venv && source .venv/bin/activate
uv pip install -e .
```

---

## Workflow

### 1. Analyze the split
```bash
uv run python src/analyze_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl
```

Look at the **Risk Score**. If HIGH or CRITICAL, clean it.

### 2. Clean if needed
```bash
uv run python src/clean_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --resplit-by instruction
```

### 3. Verify the cleaned split
```bash
uv run python src/analyze_split.py \
    --train data/train_clean.jsonl \
    --val data/val_clean.jsonl
```

Target: Risk score < 15.

### 4. Train
```bash
uv run python src/train.py \
    --train-data data/train_clean.jsonl \
    --val-data data/val_clean.jsonl \
    --output models/exp_model \
    --epochs 3 \
    --use-4bit \
    --sample-every 50
```

### 5. Evaluate
```bash
uv run python src/validate.py \
    --model models/exp_model \
    --data data/val_clean.jsonl \
    --num-samples 50
```

---

## Config

Edit `config/default.yaml`:

| Param | Default | Notes |
|-------|---------|-------|
| `batch_size` | 2 | Increase if you have more VRAM |
| `gradient_accumulation` | 8 | Effective batch = 16 |
| `epochs` | 3 | 1-2 often enough |
| `lora_rank` | 16 | Higher = more capacity |
| `gradient_checkpointing` | true | Required for memory |

---

## Risk Score Guide

| Score | Level | Action |
|-------|-------|--------|
| 0-15 | LOW | Train |
| 15-30 | MODERATE | Consider cleaning |
| 30+ | HIGH | Must clean |

---

## References

- [DreamGym Paper](https://arxiv.org/abs/2511.03773)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
