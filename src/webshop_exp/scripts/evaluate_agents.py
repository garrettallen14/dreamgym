"""Evaluate trained agents on the WebShop environment.

Usage:
    uv run evaluate-agents --agents real synthetic --num-episodes 100
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Agent:
    """Wrapper for trained agent model."""
    
    def __init__(self, model, tokenizer, temperature: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
    
    def predict(self, observation: str, instruction: str = None) -> str:
        """Predict action given observation."""
        # Format prompt
        if instruction:
            prompt = f"""Task: {instruction}

Current observation:
{observation}

What action should you take?"""
        else:
            prompt = f"""Current observation:
{observation}

What action should you take?"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Extract action from generated text
        action = self._extract_action(generated)
        return action
    
    def _extract_action(self, text: str) -> str:
        """Extract action from generated text."""
        text = text.strip()
        
        # Try to find action pattern
        action_match = re.search(r"(search\[.+?\]|click\[.+?\])", text, re.IGNORECASE)
        if action_match:
            return action_match.group(1)
        
        # If no pattern found, return first line
        first_line = text.split("\n")[0].strip()
        return first_line if first_line else "search[product]"


def load_agent(model_path: Path, base_model: Optional[str] = None) -> Agent:
    """Load a trained agent."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Try to load adapter config to find base model
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists() and base_model is None:
        with open(adapter_config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path")
    
    # Check for adapter in subdirectory
    adapter_path = model_path / "adapter"
    if not adapter_path.exists():
        adapter_path = model_path
    
    if (adapter_path / "adapter_config.json").exists() and base_model is None:
        with open(adapter_path / "adapter_config.json") as f:
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
    
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return Agent(model, tokenizer)


def extract_instruction(observation: str) -> str:
    """Extract task instruction from observation."""
    match = re.search(r"Instruction[:\s]*\n?(.+?)(?:\n\n|\n\[|$)", observation, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def evaluate_agent(
    agent: Agent,
    env,
    num_episodes: int = 100,
    max_steps: int = 15,
) -> list[dict]:
    """Evaluate agent on environment."""
    results = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        instruction = extract_instruction(obs)
        done = False
        total_reward = 0.0
        steps = 0
        actions_taken = []
        
        while not done and steps < max_steps:
            action = agent.predict(obs, instruction)
            actions_taken.append(action)
            
            try:
                obs, reward, done, info = env.step(action)
                total_reward = reward  # WebShop gives final reward
            except Exception as e:
                logger.warning(f"Step error: {e}")
                break
            
            steps += 1
        
        results.append({
            "episode": episode,
            "instruction": instruction,
            "reward": total_reward,
            "steps": steps,
            "success": total_reward > 0.5,
            "actions": actions_taken,
        })
    
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute evaluation metrics."""
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]
    successes = [r["success"] for r in results]
    
    return {
        "num_episodes": len(results),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps)),
        "median_reward": float(np.median(rewards)),
    }


class MockWebShopEnv:
    """Mock environment for testing when WebShop is not installed."""
    
    def __init__(self):
        self.instructions = [
            "Find me a black leather wallet under $50 with RFID blocking",
            "I need a laptop stand for my desk that is adjustable",
            "Search for wireless bluetooth headphones under $100",
        ]
        self.step_count = 0
        self.current_instruction = None
    
    def reset(self):
        import random
        self.step_count = 0
        self.current_instruction = random.choice(self.instructions)
        return f"""Instruction:
{self.current_instruction}

[Search]
Enter a search query to find products.

Available actions: search[query]"""
    
    def step(self, action: str):
        self.step_count += 1
        done = False
        reward = 0.0
        
        if "buy" in action.lower():
            done = True
            reward = np.random.uniform(0.3, 0.9)
        elif self.step_count >= 10:
            done = True
            reward = np.random.uniform(0, 0.3)
        
        obs = f"""Instruction:
{self.current_instruction}

[Product Page]
Product details here...

Available actions: click[Buy Now], click[Back to Search]"""
        
        return obs, reward, done, {}
    
    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["real", "synthetic"],
        help="Agent types to evaluate (real, synthetic, or paths)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per agent",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--num-products",
        type=int,
        default=1000,
        help="Number of products in environment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock environment for testing",
    )
    
    args = parser.parse_args()
    
    # Load environment
    if args.mock:
        logger.info("Using mock environment")
        env = MockWebShopEnv()
    else:
        try:
            import gym as old_gym
            from web_agent_site.envs import WebAgentTextEnv
            env = old_gym.make(
                "WebAgentTextEnv-v0",
                observation_mode="text",
                num_products=args.num_products,
            )
            logger.info(f"Loaded WebShop environment with {args.num_products} products")
        except ImportError:
            logger.warning("WebShop not installed, using mock environment")
            env = MockWebShopEnv()
    
    # Evaluate each agent
    all_results = {}
    
    for agent_name in args.agents:
        # Determine model path
        if "/" in agent_name or Path(agent_name).exists():
            model_path = Path(agent_name)
        else:
            model_path = Path(f"models/agent_{agent_name}")
        
        if not model_path.exists():
            logger.warning(f"Agent not found: {model_path}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating agent: {agent_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"{'='*60}")
        
        # Load agent
        agent = load_agent(model_path, args.base_model)
        
        # Evaluate
        results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
        )
        
        # Compute metrics
        metrics = compute_metrics(results)
        
        print(f"\n{agent_name} agent:")
        print(f"  Mean reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Mean steps: {metrics['mean_steps']:.1f}")
        
        all_results[agent_name] = {
            "metrics": metrics,
            "episodes": results[:10],  # Save first 10 episodes for inspection
        }
    
    # Compare agents
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        
        for name, data in all_results.items():
            m = data["metrics"]
            print(f"{name:15} | Reward: {m['mean_reward']:.3f} | Success: {m['success_rate']:.1%}")
        
        # Compute gap if real and synthetic both exist
        if "real" in all_results and "synthetic" in all_results:
            real_reward = all_results["real"]["metrics"]["mean_reward"]
            synth_reward = all_results["synthetic"]["metrics"]["mean_reward"]
            gap = (real_reward - synth_reward) / real_reward * 100 if real_reward > 0 else 0
            print(f"\nSynthetic vs Real reward gap: {gap:.1f}%")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results to {args.output}")
    
    env.close()
    return 0


if __name__ == "__main__":
    exit(main())
