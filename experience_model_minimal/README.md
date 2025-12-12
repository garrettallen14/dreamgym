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

## Quick Start

```bash
# Install dependencies
pip install -e .

# Prepare training data (converts trajectories to chat format)
python src/prepare_data.py --input data/trajectories.jsonl --output-dir data/

# Train the experience model
python src/train.py \
    --train-data data/train.jsonl \
    --val-data data/val.jsonl \
    --output models/experience_model \
    --epochs 3

# Validate the trained model
python src/validate.py \
    --model models/experience_model \
    --data data/val.jsonl
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

## References

1. [DreamGym: Scaling Agent Learning via Experience Synthesis](https://arxiv.org/abs/2511.03773)
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
