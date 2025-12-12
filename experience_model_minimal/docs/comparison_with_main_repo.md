# Comparison: Minimal Repo vs Main DreamGym Repository

This document provides a detailed analysis of differences between the minimal experience model training implementation and the full DreamGym repository.

## Summary of Missing Features

| Feature | Main Repo | Minimal Repo | Impact |
|---------|-----------|--------------|--------|
| **SampleGenerationCallback** | ✓ | ✗ → ✓ Fixed | HIGH - Critical for monitoring |
| **torch.compile** | ✓ | ✗ → ✓ Fixed | MEDIUM - 20% speedup |
| **Wandb Integration** | ✓ | ✗ → ✓ Fixed | MEDIUM - Experiment tracking |
| **max_steps** option | ✓ | ✗ → ✓ Fixed | LOW - Convenience |
| **torch.inference_mode()** | ✓ | ✗ → ✓ Fixed | LOW - Slight speedup |
| **autocast in validation** | ✓ | ✗ → ✓ Fixed | LOW - Memory efficiency |
| **remove_unused_columns** | ✓ | ✗ → ✓ Fixed | LOW - Memory |

---

## Detailed Analysis

### 1. SampleGenerationCallback (CRITICAL)

The main repo includes a sophisticated callback that generates sample predictions during training:

```python
class SampleGenerationCallback(TrainerCallback):
    """Generate samples at regular intervals to monitor training progress."""
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.sample_every == 0:
            self._generate_samples(model, step=state.global_step)
```

**Why it matters:**
- See how the model's predictions evolve during training
- Catch overfitting or mode collapse early
- Compare generated vs ground truth at each checkpoint
- Saves samples to JSONL for post-analysis

**Usage:**
```bash
--sample-every 100 --num-samples 3
```

### 2. torch.compile Support

The main repo supports PyTorch 2.0's `torch.compile`:

```python
if args.compile:
    if args.use_4bit:
        logger.warning("torch.compile not compatible with quantized models")
    else:
        model = torch.compile(model)
```

**Impact:**
- ~20% training speedup on compatible hardware
- First few steps slower due to compilation
- Not compatible with 4-bit quantization

### 3. Wandb Integration

```python
parser.add_argument("--use-wandb", action="store_true")
parser.add_argument("--wandb-project", type=str, default="experience-model")

if args.use_wandb:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    training_args.report_to = "wandb"
```

**Benefits:**
- Track experiments across runs
- Compare hyperparameter sweeps
- Visualize training curves
- Share results with team

### 4. Inference Optimizations

Main repo uses optimized inference:

```python
with torch.inference_mode():  # Faster than torch.no_grad()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model.generate(...)
```

**Differences:**
- `torch.inference_mode()` is faster than `torch.no_grad()` (disables version counter)
- `autocast` enables mixed precision during generation
- `use_cache=True` speeds up autoregressive generation

### 5. Training Arguments

Main repo has additional SFTConfig options:

```python
SFTConfig(
    run_name=f"experience-model-{args.base_model.split('/')[-1]}",
    remove_unused_columns=True,
    max_steps=args.max_steps,  # -1 for full epochs
)
```

---

## Performance Comparison

| Setting | Main Repo | Minimal (Before) | Minimal (After) |
|---------|-----------|------------------|-----------------|
| Training speed | Baseline | ~Same | +20% with compile |
| Memory usage | Optimized | ~Same | Same |
| Monitoring | Excellent | Basic | Excellent |
| Experiment tracking | Wandb | None | Wandb |

---

## Architecture Differences

### Main Repo
```
src/webshop_exp/
├── scripts/
│   ├── train_experience_model.py    # 481 lines
│   ├── validate_experience_model.py # 283 lines
│   ├── prepare_experience_data.py   # 206 lines
│   └── generate_synthetic_rollouts.py  # (Agent training)
├── types.py                          # Shared types
└── __init__.py
```

### Minimal Repo
```
src/
├── train.py           # 418 → 550+ lines (with callbacks)
├── validate.py        # 283 lines
├── prepare_data.py    # 183 lines
├── analyze_split.py   # NEW - Data leakage analysis
├── clean_split.py     # NEW - Data cleaning
└── types.py           # Standalone types
```

---

## What's Intentionally Excluded

The minimal repo excludes these for simplicity:

1. **generate_synthetic_rollouts.py** - Uses trained model to simulate environment
2. **Agent training pipeline** - Trains agent on synthetic data
3. **WebShop environment integration** - Real environment for evaluation
4. **Dashboard server** - Web UI for monitoring
5. **Hyperparameter sweep orchestration** - experiments/run_sweep.py

These are excluded because they're not core to understanding experience model training.

---

## Recommendations

### For Learning
Use the minimal repo - it's cleaner and easier to understand.

### For Production
Use the main DreamGym repo - it has full pipeline integration.

### For Research
Start with minimal, add features as needed.
