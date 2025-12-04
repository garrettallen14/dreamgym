# Training Insights

Research-based optimizations for QLoRA fine-tuning of experience models.

## Hardware: A40 48GB

### Throughput Optimizations (applied)

| Optimization | Impact | Status |
|--------------|--------|--------|
| Gradient checkpointing OFF | +15-20% speed | ✅ Default off |
| Flash Attention 2 | ~2x speed | ✅ Auto-enabled if available |
| `TOKENIZERS_PARALLELISM=false` | Avoids hangs, <1% impact | ✅ Set in script |
| `dataloader_num_workers=4` | Better GPU utilization | ✅ Default |

### Flash Attention 2 Installation

```bash
# On RunPod/CUDA machines:
pip install flash-attn --no-build-isolation
```

---

## LoRA Hyperparameters

### Rank Selection

| Rank | Trainable Params | Best For |
|------|------------------|----------|
| 8 | ~15M | Simple tasks, data-limited, fast iteration |
| 16 | ~30M | **Default sweet spot** for most seq2seq |
| 32 | ~60M | Complex tasks, if 16 underfits |
| 64+ | ~120M+ | Rarely needed, diminishing returns |

**Insight**: For "predict next state" tasks, rank 8-16 is likely sufficient. Higher ranks risk overfitting on 13k samples.

### Alpha/Rank Ratio

| Ratio | Effect | Use When |
|-------|--------|----------|
| α = r | Neutral scaling | Knowledge preservation, continued pretraining |
| α = 2r | **Recommended** | Seq2seq adaptation, RL world models |
| α = √r | Stabilizes high-r | Ranks ≥32 to prevent gradient explosion |

**Current setting**: α = 2×rank (correct for our task)

### Learning Rate

| LR | Behavior |
|----|----------|
| 1e-4 | Conservative, stable, slow convergence |
| **2e-4** | Standard LoRA default, usually optimal |
| 3e-4 | Slightly aggressive, often best for narrow domains |
| 5e-4 | Aggressive, monitor for instability |
| 1e-3 | Usually diverges with LoRA, avoid |

---

## Training Dynamics

### Expected Loss Curve

```
Step 0:    ~2.0  (base model, no adaptation)
Step 30:   ~0.8  (fast initial drop - NORMAL)
Step 100:  ~0.6  (plateau begins)
Epoch 1:   ~0.5  (monitor eval loss here)
Epoch 2:   ~0.4  (slight overfitting risk)
```

Fast initial convergence (64% → 83% accuracy in 30 steps) is **normal** for pretrained models on narrow domains.

### Red Flags

| Pattern | Cause | Fix |
|---------|-------|-----|
| Loss spikes >2x in warmup | LR too high | Drop LR 2-5x |
| Sawtooth oscillations | Quantization jitter | Use AdamW 8-bit, check batch size |
| Flat loss post-warmup | Underfitting | Increase rank, check data quality |
| Train↓ Eval↑ early | Overfitting | Lower LR, reduce epochs, add early stopping |
| NaN loss | Gradient explosion | Enable grad clipping, lower α/r ratio |

---

## Overfitting Risk Assessment

| Factor | Our Setup | Risk Level |
|--------|-----------|------------|
| Dataset size | 13k samples | Low (pretrained priors dominate) |
| Trainable params | ~30M (rank 16) | Low (parameter-efficient) |
| Epochs | 2 | Medium (monitor post-epoch 1) |
| Quantization | 4-bit QLoRA | Low (freezes base, regularizes) |

**Recommendation**: 2 epochs is safe. Use `--early-stopping-patience 2` for longer runs.

---

## Experience Model Considerations

### Diversity > Reward-Weighting

For robust world models, prioritize trajectory **diversity** over high-reward weighting:
- Suboptimal demos capture rare transitions
- Over-weighting optimal paths → narrow coverage
- Mixed quality mimics realistic offline RL

### Collapse Prevention (for synthetic rollouts)

When generating synthetic data in Phase 2:
- Monitor entropy (should stay >0.5, not collapse to 0)
- Mix real + synthetic data cumulatively (never replace)
- Add KL regularization if using RL fine-tuning
- Track perplexity on held-out transitions

---

## Quick Reference Commands

```bash
# Sanity check (5-10 min)
uv run python experiments/run_sweep.py --quick-test

# Full sweep (~20-24 GPU hours)
uv run python experiments/run_sweep.py --all

# With early stopping
uv run train-experience-model \
  --use-4bit \
  --early-stopping-patience 2 \
  --epochs 3

# Install Flash Attention for 2x speed
pip install flash-attn --no-build-isolation
```
