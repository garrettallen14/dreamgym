"""Extract and parse trajectory data from WebShop human demonstrations.

Usage:
    uv run extract-trajectories --input-dir data/human_demos --output data/trajectories.jsonl

The WebShop human demo format has one JSON line per page state:
- page: "index", "search_results", "item_page", "done"
- goal: contains instruction_text and target product info
- content: page-specific content (keywords, asins, options)
- reward: only on "done" page

Actions are inferred from page transitions:
- index → search_results: search[keywords]
- search_results → item_page: click[asin]
- item_page → search_results: click[Back to Search]
- item_page → item_page (different options): click[option]
- item_page → done: click[Buy Now]
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import unquote

from tqdm import tqdm

from webshop_exp.types import Trajectory, Transition, save_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_state_observation(page_data: dict) -> str:
    """Convert raw page data to a text observation like the WebShop env produces."""
    page_type = page_data.get("page", "")
    goal = page_data.get("goal", {})
    content = page_data.get("content", {})
    instruction = goal.get("instruction_text", "")
    
    if page_type == "index":
        return f"""Instruction:
{instruction}

[Search]
Enter a search query to find products matching your requirements."""

    elif page_type == "search_results":
        keywords = content.get("keywords", [])
        if isinstance(keywords, str):
            keywords = eval(keywords)  # Handle string representation of list
        asins = content.get("search_result_asins", [])
        page_num = content.get("page", 1)
        
        products = "\n".join([f"[{asin}] Product {asin}" for asin in asins[:10]])
        
        return f"""Instruction:
{instruction}

[Search Results]
Keywords: {', '.join(keywords) if keywords else 'N/A'}
Page {page_num}

{products}

Available actions: click[ASIN], click[Next >], click[Back to Search], search[new query]"""

    elif page_type == "item_page":
        asin = content.get("asin", "")
        options = content.get("options", {})
        options_str = ", ".join([f"{k}: {v}" for k, v in options.items()]) if options else "None selected"
        
        return f"""Instruction:
{instruction}

[Product Page: {asin}]
Selected options: {options_str}

Available actions: click[Buy Now], click[Back to Search], click[option_value]"""

    elif page_type == "done":
        asin = content.get("asin", "")
        options = content.get("options", {})
        price = content.get("price", 0)
        reward = page_data.get("reward", 0)
        
        return f"""[Purchase Complete]
Product: {asin}
Options: {options}
Price: ${price}
Reward: {reward:.2f}"""

    return str(page_data)


def infer_action(prev_page: dict, curr_page: dict) -> str:
    """Infer the action taken between two page states."""
    prev_type = prev_page.get("page", "")
    curr_type = curr_page.get("page", "")
    prev_content = prev_page.get("content", {})
    curr_content = curr_page.get("content", {})
    
    # index → search_results: search action
    if prev_type == "index" and curr_type == "search_results":
        keywords = curr_content.get("keywords", [])
        if isinstance(keywords, str):
            keywords = eval(keywords)
        return f"search[{' '.join(keywords)}]"
    
    # search_results → item_page: click on product
    if prev_type == "search_results" and curr_type == "item_page":
        asin = curr_content.get("asin", "")
        return f"click[{asin}]"
    
    # item_page → search_results: back to search
    if prev_type == "item_page" and curr_type == "search_results":
        return "click[Back to Search]"
    
    # item_page → item_page: option selection or different product
    if prev_type == "item_page" and curr_type == "item_page":
        prev_asin = prev_content.get("asin", "")
        curr_asin = curr_content.get("asin", "")
        prev_options = prev_content.get("options", {})
        curr_options = curr_content.get("options", {})
        
        if prev_asin != curr_asin:
            return f"click[{curr_asin}]"
        
        # Find changed option
        for key, value in curr_options.items():
            if key not in prev_options or prev_options[key] != value:
                return f"click[{value}]"
        
        return "click[option]"
    
    # item_page → done: buy
    if curr_type == "done":
        return "click[Buy Now]"
    
    # search_results → search_results: pagination or new search
    if prev_type == "search_results" and curr_type == "search_results":
        prev_page_num = prev_content.get("page", 1)
        curr_page_num = curr_content.get("page", 1)
        prev_keywords = prev_content.get("keywords", [])
        curr_keywords = curr_content.get("keywords", [])
        
        if curr_page_num > prev_page_num:
            return "click[Next >]"
        elif curr_page_num < prev_page_num:
            return "click[< Prev]"
        elif prev_keywords != curr_keywords:
            if isinstance(curr_keywords, str):
                curr_keywords = eval(curr_keywords)
            return f"search[{' '.join(curr_keywords)}]"
    
    return "unknown_action"


def parse_webshop_trajectory(filepath: Path) -> Optional[Trajectory]:
    """Parse a WebShop human demo trajectory file.
    
    Each file contains one episode with JSON lines for each page visited.
    """
    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # Parse all page states
        pages = []
        for line in lines:
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        
        if len(pages) < 2:
            return None
        
        # Get instruction from goal
        instruction = pages[0].get("goal", {}).get("instruction_text", "")
        if not instruction:
            return None
        
        # Build transitions
        transitions = []
        total_reward = 0.0
        
        for i in range(len(pages) - 1):
            prev_page = pages[i]
            curr_page = pages[i + 1]
            
            state = format_state_observation(prev_page)
            action = infer_action(prev_page, curr_page)
            next_state = format_state_observation(curr_page)
            
            # Reward is only given at the end
            is_done = curr_page.get("page") == "done"
            reward = curr_page.get("reward", 0.0) if is_done else 0.0
            
            if is_done:
                total_reward = reward
            
            transitions.append(Transition(
                task_instruction=instruction,
                state=state,
                action=action,
                next_state=next_state,
                reward=float(reward),
                done=is_done,
            ))
        
        if not transitions:
            return None
        
        return Trajectory(
            instruction=instruction,
            transitions=transitions,
            total_reward=total_reward,
            metadata={
                "source_file": str(filepath),
                "target_asin": pages[0].get("goal", {}).get("asin", ""),
                "num_pages": len(pages),
            },
        )
        
    except Exception as e:
        logger.warning(f"Error parsing {filepath}: {e}")
        return None


def find_trajectory_files(input_dir: Path) -> Generator[Path, None, None]:
    """Find all trajectory files in the input directory."""
    extensions = [".jsonl", ".json"]
    
    for ext in extensions:
        yield from input_dir.rglob(f"*{ext}")


def extract_trajectories(input_dir: Path) -> list[Trajectory]:
    """Extract all trajectories from input directory."""
    trajectories = []
    
    files = list(find_trajectory_files(input_dir))
    logger.info(f"Found {len(files)} trajectory files")
    
    for filepath in tqdm(files, desc="Extracting trajectories"):
        traj = parse_webshop_trajectory(filepath)
        if traj:
            trajectories.append(traj)
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Extract trajectories from WebShop human demos")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/human_demos"),
        help="Directory containing human demonstration data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trajectories.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum number of steps per trajectory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of steps per trajectory",
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        logger.info("Please download human demonstrations and place them in the input directory.")
        logger.info("URL: https://drive.google.com/file/d/1GWC8UlUzfT9PRTRxgYOwuKSJp4hyV1dp/view")
        return 1
    
    # Extract trajectories
    logger.info(f"Extracting trajectories from {args.input_dir}")
    trajectories = extract_trajectories(args.input_dir)
    
    # Filter by step count
    original_count = len(trajectories)
    trajectories = [
        t for t in trajectories
        if args.min_steps <= len(t.transitions) <= args.max_steps
    ]
    logger.info(f"Filtered {original_count} -> {len(trajectories)} trajectories by step count")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories(trajectories, str(args.output))
    logger.info(f"Saved {len(trajectories)} trajectories to {args.output}")
    
    # Print statistics
    total_transitions = sum(len(t.transitions) for t in trajectories)
    avg_steps = total_transitions / len(trajectories) if trajectories else 0
    rewards = [t.total_reward for t in trajectories]
    
    logger.info(f"Statistics:")
    logger.info(f"  Total trajectories: {len(trajectories)}")
    logger.info(f"  Total transitions: {total_transitions}")
    logger.info(f"  Avg steps per trajectory: {avg_steps:.2f}")
    logger.info(f"  Avg reward: {sum(rewards) / len(rewards) if rewards else 0:.3f}")
    logger.info(f"  Success rate (reward > 0.5): {sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0:.1%}")
    
    return 0


if __name__ == "__main__":
    exit(main())
