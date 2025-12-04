"""Generate synthetic trajectories using the trained Experience Model.

Usage:
    uv run generate-synthetic --model models/experience_model --output data/synthetic_trajectories.jsonl
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from webshop_exp.types import Trajectory, Transition, save_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SyntheticEnvironment:
    """Simulated WebShop environment using the trained Experience Model."""
    
    def __init__(
        self,
        model,
        tokenizer,
        instructions: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.instructions = instructions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.current_instruction = None
        self.current_state = None
        self.step_count = 0
    
    def reset(self, instruction: Optional[str] = None) -> str:
        """Initialize episode with instruction, return initial state."""
        if instruction is None:
            instruction = random.choice(self.instructions)
        
        self.current_instruction = instruction
        self.step_count = 0
        
        # Generate initial state (search page)
        self.current_state = self._generate_initial_state(instruction)
        return self.current_state
    
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """Use experience model to predict next state."""
        self.step_count += 1
        
        prompt = self._format_prompt(
            self.current_instruction,
            self.current_state,
            action,
        )
        
        # Generate prediction
        output = self._generate(prompt)
        next_state, reward, done = self._parse_output(output)
        
        self.current_state = next_state
        return next_state, reward, done, {"raw_output": output}
    
    def _format_prompt(self, instruction: str, state: str, action: str) -> str:
        return f"""Task: {instruction}

Current State:
{state}

Action taken: {action}

Predict the next state and reward."""
    
    def _generate_initial_state(self, instruction: str) -> str:
        """Generate initial search page state."""
        # Use a special prompt to generate the initial state
        prompt = f"""Task: {instruction}

Generate the initial WebShop search page observation for this task. The observation should include:
- The task instruction
- A search bar
- Instructions to search for products

Format as a WebShop observation."""
        
        output = self._generate(prompt)
        
        # If generation fails, use a template
        if not output or len(output) < 50:
            output = f"""Instruction:
{instruction}

[Search]
Enter a search query to find products matching your requirements.

Available actions: search[query]"""
        
        return output
    
    def _generate(self, prompt: str) -> str:
        """Generate completion from the model."""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced from 512
                    do_sample=False,     # Greedy is 2x faster
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated
    
    def _parse_output(self, output: str) -> Tuple[str, float, bool]:
        """Parse model output to extract next_state, reward, done."""
        next_state = ""
        reward = 0.0
        done = False
        
        # Extract next state
        state_match = re.search(r"Next State:\s*(.*?)(?=\nReward:|$)", output, re.DOTALL)
        if state_match:
            next_state = state_match.group(1).strip()
        else:
            next_state = output.strip()
        
        # Extract reward
        reward_match = re.search(r"Reward:\s*([\d.]+)", output)
        if reward_match:
            try:
                reward = float(reward_match.group(1))
            except ValueError:
                pass
        
        # Extract done
        done_match = re.search(r"Done:\s*(True|False)", output, re.IGNORECASE)
        if done_match:
            done = done_match.group(1).lower() == "true"
        
        return next_state, reward, done
    
    def get_available_actions(self, state: str) -> List[str]:
        """Extract available actions from state observation."""
        actions = []
        
        # Look for product IDs
        product_ids = re.findall(r"\[([A-Z0-9]{10,})\]", state)
        for pid in product_ids:
            actions.append(f"click[{pid}]")
        
        # Standard actions based on state content
        if "search" in state.lower():
            actions.append("search[query]")
        
        if "buy now" in state.lower() or "product page" in state.lower():
            actions.append("click[Buy Now]")
        
        if "back" in state.lower() or "search results" in state.lower():
            actions.append("click[Back to Search]")
        
        if "next" in state.lower():
            actions.append("click[Next >]")
        
        return actions if actions else ["search[product]"]


class HeuristicPolicy:
    """Simple heuristic policy for generating rollouts."""
    
    def __init__(self, exploration_prob: float = 0.3):
        self.exploration_prob = exploration_prob
    
    def select_action(
        self,
        instruction: str,
        state: str,
        available_actions: List[str],
        step_count: int,
    ) -> str:
        """Select an action based on heuristics."""
        
        # Extract keywords from instruction
        keywords = self._extract_keywords(instruction)
        
        # Decide action type based on state and step
        if step_count == 0 or "search" in state.lower() and "results" not in state.lower():
            # First step: search with keywords
            query = " ".join(keywords[:3]) if keywords else "product"
            return f"search[{query}]"
        
        # Random exploration
        if random.random() < self.exploration_prob and available_actions:
            return random.choice(available_actions)
        
        # Look for buy action late in episode
        if step_count > 3:
            buy_actions = [a for a in available_actions if "buy" in a.lower()]
            if buy_actions:
                return buy_actions[0]
        
        # Click on products
        product_actions = [a for a in available_actions if re.match(r"click\[[A-Z0-9]+\]", a)]
        if product_actions:
            return product_actions[0]
        
        # Default: random available action
        return random.choice(available_actions) if available_actions else "search[product]"
    
    def _extract_keywords(self, instruction: str) -> List[str]:
        """Extract search keywords from instruction."""
        # Remove common words
        stop_words = {
            "find", "me", "a", "an", "the", "with", "that", "is", "are", "for",
            "and", "or", "under", "over", "less", "more", "than", "please", "i",
            "want", "need", "looking", "search", "get", "buy", "purchase",
        }
        
        words = instruction.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords


def load_instructions(path: Optional[Path]) -> List[str]:
    """Load task instructions from file."""
    if path and path.exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    
    # Default sample instructions
    return [
        "Find me a black leather wallet under $50 with RFID blocking",
        "I need a laptop stand for my desk that is adjustable",
        "Search for wireless bluetooth headphones with noise cancellation under $100",
        "Find a red dress for a formal event, size medium",
        "I'm looking for a stainless steel water bottle, 32 oz",
        "Search for running shoes, men's size 10, good for marathon",
        "Find me a mechanical keyboard with RGB lighting",
        "I need a yoga mat that is extra thick and non-slip",
        "Search for a cast iron skillet, 12 inch",
        "Find wireless earbuds with long battery life under $80",
    ]


def generate_single_trajectory(
    env: SyntheticEnvironment,
    policy: HeuristicPolicy,
    max_steps: int,
) -> Trajectory:
    """Generate a single trajectory."""
    instruction = random.choice(env.instructions)
    state = env.reset(instruction)
    
    transitions = []
    done = False
    step = 0
    
    while not done and step < max_steps:
        available_actions = env.get_available_actions(state)
        action = policy.select_action(instruction, state, available_actions, step)
        next_state, reward, done, info = env.step(action)
        
        transitions.append(Transition(
            task_instruction=instruction,
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
        ))
        
        state = next_state
        step += 1
    
    total_reward = transitions[-1].reward if transitions else 0.0
    return Trajectory(
        instruction=instruction,
        transitions=transitions,
        total_reward=total_reward,
        metadata={"synthetic": True, "num_steps": len(transitions)},
    )


def generate_trajectories(
    env: SyntheticEnvironment,
    policy: HeuristicPolicy,
    num_trajectories: int,
    max_steps: int = 8,
) -> List[Trajectory]:
    """Generate synthetic trajectories."""
    trajectories = []
    
    for _ in tqdm(range(num_trajectories), desc="Generating trajectories"):
        traj = generate_single_trajectory(env, policy, max_steps)
        trajectories.append(traj)
    
    return trajectories


def filter_trajectories(
    trajectories: List[Trajectory],
    min_steps: int = 2,
    max_steps: int = 20,
    min_state_length: int = 50,
) -> List[Trajectory]:
    """Filter trajectories for quality."""
    filtered = []
    
    for traj in trajectories:
        # Check step count
        if not (min_steps <= len(traj.transitions) <= max_steps):
            continue
        
        # Check state quality (not too short)
        valid = True
        for t in traj.transitions:
            if len(t.state) < min_state_length or len(t.next_state) < min_state_length:
                valid = False
                break
        
        if valid:
            filtered.append(traj)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic trajectories")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/experience_model"),
        help="Path to trained experience model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=None,
        help="File with task instructions (one per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_trajectories.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1000,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,  # Reduced from 15 - most tasks complete in 5-8 steps
        help="Maximum steps per trajectory",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--exploration-prob",
        type=float,
        default=0.3,
        help="Probability of random action exploration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check model exists
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        logger.info("Run train-experience-model first.")
        return 1
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    from webshop_exp.scripts.validate_experience_model import load_model
    model, tokenizer = load_model(args.model, args.base_model)
    
    # Load instructions
    instructions = load_instructions(args.instructions)
    logger.info(f"Loaded {len(instructions)} task instructions")
    
    # Create environment and policy
    env = SyntheticEnvironment(
        model=model,
        tokenizer=tokenizer,
        instructions=instructions,
        temperature=args.temperature,
    )
    policy = HeuristicPolicy(exploration_prob=args.exploration_prob)
    
    # Generate trajectories
    logger.info(f"Generating {args.num_trajectories} synthetic trajectories")
    trajectories = generate_trajectories(
        env=env,
        policy=policy,
        num_trajectories=args.num_trajectories,
        max_steps=args.max_steps,
    )
    
    # Filter for quality
    original_count = len(trajectories)
    trajectories = filter_trajectories(trajectories)
    logger.info(f"Filtered {original_count} -> {len(trajectories)} trajectories")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories(trajectories, str(args.output))
    logger.info(f"Saved {len(trajectories)} trajectories to {args.output}")
    
    # Statistics
    total_transitions = sum(len(t.transitions) for t in trajectories)
    avg_steps = total_transitions / len(trajectories) if trajectories else 0
    rewards = [t.total_reward for t in trajectories]
    
    logger.info("Statistics:")
    logger.info(f"  Total trajectories: {len(trajectories)}")
    logger.info(f"  Total transitions: {total_transitions}")
    logger.info(f"  Avg steps per trajectory: {avg_steps:.2f}")
    logger.info(f"  Avg reward: {sum(rewards) / len(rewards) if rewards else 0:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
