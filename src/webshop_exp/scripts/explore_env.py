"""Explore the WebShop gym environment interface.

Usage:
    uv run explore-env --num-products 1000 --interactive
"""

import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def explore_environment(num_products: int = 1000, interactive: bool = False):
    """Explore the WebShop environment and document its interface."""
    
    try:
        import gymnasium as gym
    except ImportError:
        logger.error("gymnasium not installed. Run: uv pip install gymnasium")
        return 1
    
    # Try to import WebShop environment
    try:
        # WebShop uses the older gym API
        import gym as old_gym
        from web_agent_site.envs import WebAgentTextEnv
        
        env = old_gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=num_products,
        )
        logger.info("Successfully loaded WebShop environment")
        
    except ImportError as e:
        logger.warning(f"WebShop environment not available: {e}")
        logger.info("To use WebShop, clone and setup: https://github.com/princeton-nlp/WebShop")
        logger.info("\nRunning with mock environment for demonstration...")
        env = MockWebShopEnv()
    
    # Reset and explore
    print("\n" + "=" * 80)
    print("WEBSHOP ENVIRONMENT EXPLORATION")
    print("=" * 80)
    
    obs = env.reset()
    print("\n### Initial Observation ###")
    print(obs[:1000] if len(obs) > 1000 else obs)
    print(f"\n[Observation length: {len(obs)} chars]")
    
    # Get available actions if supported
    if hasattr(env, "get_available_actions"):
        actions = env.get_available_actions()
        print("\n### Available Actions ###")
        for action in actions[:20]:
            print(f"  - {action}")
        if len(actions) > 20:
            print(f"  ... and {len(actions) - 20} more")
    
    # Take some sample actions
    print("\n### Sample Interactions ###")
    
    sample_actions = [
        "search[laptop]",
        "click[Next >]",
        "click[Back to Search]",
    ]
    
    for action in sample_actions:
        print(f"\n> Action: {action}")
        try:
            obs, reward, done, info = env.step(action)
            print(f"Observation: {obs[:500]}...")
            print(f"Reward: {reward}, Done: {done}")
            
            if done:
                print("Episode finished!")
                obs = env.reset()
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Interactive mode
    if interactive:
        print("\n### Interactive Mode ###")
        print("Enter actions (or 'quit' to exit, 'reset' to restart):")
        
        while True:
            try:
                action = input("\n> ").strip()
                
                if action.lower() == "quit":
                    break
                elif action.lower() == "reset":
                    obs = env.reset()
                    print(obs[:1000] if len(obs) > 1000 else obs)
                    continue
                elif action.lower() == "actions":
                    if hasattr(env, "get_available_actions"):
                        for a in env.get_available_actions()[:30]:
                            print(f"  - {a}")
                    continue
                
                obs, reward, done, info = env.step(action)
                print(obs[:1000] if len(obs) > 1000 else obs)
                print(f"\n[Reward: {reward}, Done: {done}]")
                
                if done:
                    print("\nEpisode finished! Use 'reset' to start a new episode.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    env.close()
    
    # Document findings
    print("\n" + "=" * 80)
    print("ENVIRONMENT DOCUMENTATION")
    print("=" * 80)
    print("""
## Observation Format
- Text-based observation of the current page
- Includes page type (search, item, item_page)
- Lists visible products/elements with IDs
- Shows task instruction at the top

## Action Space
- search[query]: Search for products with the given query
- click[element]: Click on a visible element (product ID, button, etc.)
- click[Buy Now]: Complete the purchase

## Reward Structure
- Sparse reward: 0 during episode
- Final reward (0-1) at purchase based on match quality
- Match quality considers: price, attributes, title match

## Key Observations
- State includes task instruction for context
- Available actions change based on current page
- Episode ends when Buy Now is clicked or max steps reached
""")
    
    return 0


class MockWebShopEnv:
    """Mock environment for demonstration when WebShop is not installed."""
    
    def __init__(self):
        self.step_count = 0
        self.state = "search"
        self.instruction = "Find me a black leather wallet under $50 with RFID blocking"
        
    def reset(self):
        self.step_count = 0
        self.state = "search"
        return f"""Instruction:
{self.instruction}

[Search]
Search for products matching your task.
Enter a search query to find relevant products.

Available actions: search[query]"""
    
    def step(self, action: str):
        self.step_count += 1
        done = False
        reward = 0.0
        info = {}
        
        if action.startswith("search["):
            self.state = "results"
            obs = f"""Instruction:
{self.instruction}

[Search Results]
Page 1 (Showing 1-10 of 50 results)

[B07XYZ001] Black Leather Wallet with RFID Blocking - $29.99
★★★★☆ (4.2) | 150 reviews | Prime eligible

[B07XYZ002] Genuine Leather Bifold Wallet - $45.00
★★★★★ (4.8) | 89 reviews | Free shipping

[B07XYZ003] RFID Blocking Card Holder - $19.99
★★★☆☆ (3.5) | 42 reviews

Available actions: click[B07XYZ001], click[B07XYZ002], click[B07XYZ003], click[Next >], search[new query]"""
            
        elif action.startswith("click[B07"):
            self.state = "product"
            product_id = action.split("[")[1].rstrip("]")
            obs = f"""Instruction:
{self.instruction}

[Product Page: {product_id}]
Black Leather Wallet with RFID Blocking

Price: $29.99
Rating: ★★★★☆ (4.2/5, 150 reviews)

Features:
- Genuine leather construction
- RFID blocking technology
- 8 card slots + 2 ID windows
- Bifold design
- Color: Black

Available actions: click[Buy Now], click[Back to Search], click[Add to Cart]"""
            
        elif "buy" in action.lower():
            done = True
            reward = 0.85  # Simulated reward
            obs = f"""Thank you for your purchase!

Order confirmed for: Black Leather Wallet with RFID Blocking
Total: $29.99

[Episode Complete]
Task: {self.instruction}
Reward: {reward}"""
            
        else:
            obs = self.reset()
        
        return obs, reward, done, info
    
    def close(self):
        pass
    
    def get_available_actions(self):
        if self.state == "search":
            return ["search[leather wallet]", "search[black wallet rfid]"]
        elif self.state == "results":
            return ["click[B07XYZ001]", "click[B07XYZ002]", "click[Next >]", "search[new query]"]
        else:
            return ["click[Buy Now]", "click[Back to Search]", "click[Add to Cart]"]


def main():
    parser = argparse.ArgumentParser(description="Explore WebShop environment")
    parser.add_argument(
        "--num-products",
        type=int,
        default=1000,
        help="Number of products in the environment",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive exploration mode",
    )
    
    args = parser.parse_args()
    return explore_environment(args.num_products, args.interactive)


if __name__ == "__main__":
    exit(main())
