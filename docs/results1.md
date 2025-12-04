# 🎉 Training Complete!

## Final Results

| Metric | Value |
|--------|-------|
| **Eval Loss** | 0.393 |
| **Token Accuracy** | **90.6%** |
| **Train Loss** | 0.521 |
| **Total Time** | 4h 27min |
| **Speed** | 9.36s/step (was 12s before speedups!) |

## Sample Quality - Looks Great! ✅

| Sample | Ground Truth vs Generated |
|--------|---------------------------|
| #1 (window film) | ✅ **Exact match** |
| #2 (desktop) | ✅ Match (minor price diff: $485 vs $600) |
| #3 (work shoes) | ✅ **Exact match** |

## Model Saved To
```
models/experiments/yolo_final/adapter/
```

---

## Next Steps

1. **Test the model** on held-out trajectories
2. **Push to Hub** (optional):
   ```bash
   uv run python -c "from peft import PeftModel; model.push_to_hub('your-username/webshop-experience-model')"
   ```

3. **Use for inference**:
   ```python
   from peft import AutoPeftModelForCausalLM
   model = AutoPeftModelForCausalLM.from_pretrained("models/experiments/yolo_final/adapter")
   ```

Want me to write an evaluation script to test it on the validation set, or a script to run inference with the trained model?