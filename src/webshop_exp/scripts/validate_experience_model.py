"""Validate the trained Experience Model.

Computes metrics on held-out data and shows qualitative examples.

Usage:
    uv run validate-experience-model --model models/experience_model --data data/experience_val.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: Path, base_model: Optional[str] = None):
    """Load the trained experience model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Try to load adapter config to find base model
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists() and base_model is None:
        with open(adapter_config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path")
    
    if base_model is None:
        base_model = "Qwen/Qwen2.5-3B-Instruct"
        logger.warning(f"Could not find base model, defaulting to {base_model}")
    
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    adapter_path = model_path / "adapter"
    if adapter_path.exists():
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    return model, tokenizer


def parse_completion(completion: str) -> tuple[str, float, bool]:
    """Parse the completion to extract next_state, reward, done."""
    next_state = ""
    reward = 0.0
    done = False
    
    # Extract next state
    state_match = re.search(r"Next State:\s*(.*?)(?=\nReward:|$)", completion, re.DOTALL)
    if state_match:
        next_state = state_match.group(1).strip()
    
    # Extract reward
    reward_match = re.search(r"Reward:\s*([\d.]+)", completion)
    if reward_match:
        try:
            reward = float(reward_match.group(1))
        except ValueError:
            pass
    
    # Extract done
    done_match = re.search(r"Done:\s*(True|False)", completion, re.IGNORECASE)
    if done_match:
        done = done_match.group(1).lower() == "true"
    
    return next_state, reward, done


def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a prediction from the model."""
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute evaluation metrics."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.metrics import f1_score, mean_squared_error
    
    # State prediction: BLEU score
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    for pred in predictions:
        reference = pred["actual_state"].split()
        hypothesis = pred["predicted_state"].split()
        
        if reference and hypothesis:
            score = sentence_bleu([reference], hypothesis, smoothing_function=smoothing)
            bleu_scores.append(score)
    
    # Reward prediction: MSE
    actual_rewards = [p["actual_reward"] for p in predictions]
    predicted_rewards = [p["predicted_reward"] for p in predictions]
    reward_mse = mean_squared_error(actual_rewards, predicted_rewards)
    
    # Done prediction: F1
    actual_done = [int(p["actual_done"]) for p in predictions]
    predicted_done = [int(p["predicted_done"]) for p in predictions]
    done_f1 = f1_score(actual_done, predicted_done, zero_division=0)
    
    return {
        "state_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        "reward_mse": reward_mse,
        "done_f1": done_f1,
        "num_samples": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate experience model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/experience_model"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/experience_val.jsonl"),
        help="Validation data JSONL file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of examples to display",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for predictions",
    )
    
    args = parser.parse_args()
    
    # Check inputs exist
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        return 1
    
    if not args.data.exists():
        logger.error(f"Validation data not found: {args.data}")
        return 1
    
    # Download NLTK data
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model, tokenizer = load_model(args.model, args.base_model)
    
    # Load validation data
    logger.info(f"Loading validation data from {args.data}")
    samples = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    # Limit samples
    if len(samples) > args.num_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, args.num_samples)
    
    logger.info(f"Evaluating on {len(samples)} samples")
    
    # Generate predictions
    predictions = []
    for sample in tqdm(samples, desc="Generating predictions"):
        prompt = sample["prompt"]
        actual_completion = sample["completion"]
        
        # Parse actual values
        actual_state, actual_reward, actual_done = parse_completion(actual_completion)
        
        # Generate prediction
        predicted_completion = generate_prediction(model, tokenizer, prompt)
        predicted_state, predicted_reward, predicted_done = parse_completion(predicted_completion)
        
        predictions.append({
            "prompt": prompt,
            "actual_completion": actual_completion,
            "predicted_completion": predicted_completion,
            "actual_state": actual_state,
            "predicted_state": predicted_state,
            "actual_reward": actual_reward,
            "predicted_reward": predicted_reward,
            "actual_done": actual_done,
            "predicted_done": predicted_done,
        })
    
    # Compute metrics
    metrics = compute_metrics(predictions)
    
    print("\n" + "=" * 80)
    print("EXPERIENCE MODEL VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nMetrics (n={metrics['num_samples']}):")
    print(f"  State Prediction BLEU: {metrics['state_bleu']:.4f}")
    print(f"  Reward Prediction MSE: {metrics['reward_mse']:.4f}")
    print(f"  Done Prediction F1: {metrics['done_f1']:.4f}")
    
    # Show examples
    if args.show_examples > 0:
        print("\n" + "-" * 80)
        print("EXAMPLE PREDICTIONS")
        print("-" * 80)
        
        for i, pred in enumerate(predictions[:args.show_examples]):
            print(f"\n### Example {i + 1} ###")
            print(f"PROMPT:\n{pred['prompt'][:300]}...")
            print(f"\nACTUAL:\n{pred['actual_completion'][:300]}...")
            print(f"\nPREDICTED:\n{pred['predicted_completion'][:300]}...")
            print(f"\nReward - Actual: {pred['actual_reward']}, Predicted: {pred['predicted_reward']}")
            print(f"Done - Actual: {pred['actual_done']}, Predicted: {pred['predicted_done']}")
    
    # Save predictions if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "metrics": metrics,
                "predictions": predictions,
            }, f, indent=2)
        logger.info(f"Saved predictions to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
