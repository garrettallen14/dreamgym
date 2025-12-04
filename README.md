# WebShop Experience Model

A proof-of-concept implementation of the **DreamGym** framework for the WebShop e-commerce agent benchmark. The Experience Model learns to predict state transitions from real trajectories, enabling synthetic data generation for agent training.

## Overview

**Goal**: Train a small LLM to simulate the WebShop environment, then use it to generate synthetic trajectories for training an agent.

**Success Criteria**: Agent trained on synthetic trajectories matches or beats agent trained on same quantity of real trajectories.

## Quick Start

```bash
# Setup environment with uv
uv sync

# Explore the environment (mock mode if WebShop not installed)
uv run explore-env --interactive

# Run the full pipeline (see Pipeline section below)
```

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended: 40GB+ VRAM for training)

### Setup

```bash
# Clone this repository
git clone <repo-url>
cd dreamgym

# Create virtual environment and install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Optional: WebShop Environment

For real environment evaluation, install WebShop:

```bash
# Clone WebShop
git clone https://github.com/princeton-nlp/webshop.git
cd webshop

# Setup (use small dataset for faster iteration)
./setup.sh -d small
```

Download human demonstration data from:
- https://drive.google.com/file/d/1GWC8UlUzfT9PRTRxgYOwuKSJp4hyV1dp/view

## Pipeline

### Phase 1: Data Extraction

```bash
# Extract trajectories from human demonstrations
uv run extract-trajectories \
    --input-dir data/human_demos \
    --output data/trajectories.jsonl
```

### Phase 2: Experience Model Training

```bash
# Prepare training data
uv run prepare-experience-data \
    --input data/trajectories.jsonl \
    --output-dir data \
    --hf-format

# Train experience model
uv run train-experience-model \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --train-data data/experience_train_hf.jsonl \
    --output models/experience_model \
    --epochs 3 \
    --use-4bit  # Optional: for lower VRAM usage

# Validate experience model
uv run validate-experience-model \
    --model models/experience_model \
    --data data/experience_val.jsonl \
    --show-examples 5
```

### Phase 3: Synthetic Data Generation

```bash
# Generate synthetic trajectories
uv run generate-synthetic \
    --model models/experience_model \
    --output data/synthetic_trajectories.jsonl \
    --num-trajectories 5000
```

### Phase 4: Agent Training

```bash
# Prepare agent training data
uv run prepare-agent-data \
    --input data/trajectories.jsonl \
    --output data/agent_train_real.jsonl \
    --hf-format

uv run prepare-agent-data \
    --input data/synthetic_trajectories.jsonl \
    --output data/agent_train_synthetic.jsonl \
    --hf-format

# Train agents
uv run train-agent --data real --output models/agent_real
uv run train-agent --data synthetic --output models/agent_synthetic
```

### Phase 5: Evaluation

```bash
# Evaluate both agents
uv run evaluate-agents \
    --agents real synthetic \
    --num-episodes 100 \
    --output results/evaluation.json

# Use --mock flag if WebShop is not installed
uv run evaluate-agents --agents real synthetic --mock
```

## Project Structure

```
dreamgym/
├── src/webshop_exp/
│   ├── __init__.py
│   ├── types.py                           # Core data types
│   └── scripts/
│       ├── extract_trajectories.py        # Phase 1
│       ├── explore_env.py
│       ├── prepare_experience_data.py     # Phase 2
│       ├── train_experience_model.py
│       ├── validate_experience_model.py
│       ├── generate_synthetic_rollouts.py # Phase 3
│       ├── prepare_agent_data.py          # Phase 4
│       ├── train_agent.py
│       └── evaluate_agents.py             # Phase 5
├── data/
│   ├── trajectories.jsonl
│   ├── synthetic_trajectories.jsonl
│   ├── experience_train_hf.jsonl
│   └── agent_train_*.jsonl
├── models/
│   ├── experience_model/
│   ├── agent_real/
│   └── agent_synthetic/
├── results/
│   └── evaluation.json
├── pyproject.toml
└── README.md
```

## Available Commands

| Command | Description |
|---------|-------------|
| `uv run extract-trajectories` | Extract trajectories from human demos |
| `uv run explore-env` | Explore WebShop environment interactively |
| `uv run prepare-experience-data` | Prepare data for experience model training |
| `uv run train-experience-model` | Train the experience model with LoRA |
| `uv run validate-experience-model` | Validate experience model predictions |
| `uv run generate-synthetic` | Generate synthetic trajectories |
| `uv run prepare-agent-data` | Prepare data for agent training |
| `uv run train-agent` | Train agent on real or synthetic data |
| `uv run evaluate-agents` | Evaluate agents on environment |

Run any command with `--help` for detailed options.

## Configuration

### Training Arguments

Key training parameters (configurable via CLI):

- `--base-model`: Base model for fine-tuning (default: `Qwen/Qwen2.5-3B-Instruct`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Per-device batch size (default: 4)
- `--lora-rank`: LoRA rank (default: 16)
- `--use-4bit`: Enable QLoRA for lower VRAM usage

### Supported Base Models

1. `Qwen/Qwen2.5-3B-Instruct` - Recommended
2. `meta-llama/Llama-3.2-3B-Instruct` - Alternative
3. `microsoft/Phi-3-mini-4k-instruct` - Smallest, fastest

## Compute Requirements

- **GPU**: 1x A40 (48GB) or 1x A100 (40GB) recommended
- **With QLoRA (4-bit)**: Can run on 24GB GPUs

**Estimated training time** (on A40):
- Experience model: ~8 hours
- Synthetic generation: ~4 hours
- Agent training (x2): ~8 hours
- Evaluation: ~2 hours

## Success Metrics

| Metric | Target |
|--------|--------|
| Experience model state prediction BLEU | >0.5 |
| Synthetic agent vs Real agent reward gap | <10% |
| Synthetic agent success rate | >30% |

## Key References

1. **DreamGym**: arXiv:2511.03773 - "Scaling Agent Learning via Experience Synthesis"
2. **WebShop**: arXiv:2207.01206 - "WebShop: Towards Scalable Real-World Web Interaction"
3. **WebShop GitHub**: https://github.com/princeton-nlp/WebShop

## License

MIT
