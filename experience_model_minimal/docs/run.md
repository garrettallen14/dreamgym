cd /dreamgym/experience_model_minimal

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv
source .venv/bin/activate
uv pip install -e .

uv run python src/analyze_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl

uv run python src/clean_split.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --resplit-by instruction

uv run python src/analyze_split.py \
    --train data/train_clean.jsonl \
    --val data/val_clean.jsonl

uv run python src/train.py \
    --train-data data/train_clean.jsonl \
    --val-data data/val_clean.jsonl \
    --output models/exp_model \
    --epochs 3 \
    --use-4bit \
    --sample-every 50 \
    --num-samples 3

uv run python src/validate.py \
    --model models/exp_model \
    --data data/val_clean.jsonl \
    --num-samples 50


# QUICK TEST:
uv run python src/train.py \
    --train-data data/train_clean.jsonl \
    --val-data data/val_clean.jsonl \
    --output models/test_run \
    --max-steps 100 \
    --use-4bit \
    --sample-every 25