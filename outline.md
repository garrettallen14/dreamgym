# WebShop Experience Model: Coding Agent Prompt

## Project Overview

Build a proof-of-concept "Experience Model" for the WebShop e-commerce agent benchmark, following the DreamGym framework approach. The Experience Model learns to predict state transitions (state, action → next_state, reward) from real trajectories, enabling synthetic rollout generation for agent training without expensive real environment interactions.

**Goal**: Train a small LLM to simulate the WebShop environment, then use it to generate synthetic trajectories for training an agent.

**Success Criteria**: Agent trained on synthetic trajectories matches or beats agent trained on same quantity of real trajectories.

---

## Phase 1: Environment Setup & Data Extraction

### Task 1.1: Clone and Setup WebShop

```bash
# Clone the repository
git clone https://github.com/princeton-nlp/webshop.git
cd webshop

# Create conda environment
conda create -n webshop python=3.8.13 -y
conda activate webshop

# Run setup (use small dataset for faster iteration)
./setup.sh -d small
```

**Expected output**: WebShop environment running locally with 1,000 products loaded.

### Task 1.2: Download Human Demonstration Data

Download the human demonstration trajectories from:
- URL: https://drive.google.com/file/d/1GWC8UlUzfT9PRTRxgYOwuKSJp4hyV1dp/view

Place in `webshop/data/` directory.

### Task 1.3: Extract and Parse Trajectory Data

Create a script `extract_trajectories.py` that:

1. Loads the human demonstration data (likely in `user_session_logs/` or downloaded archive)
2. Parses trajectory files (each file = one episode, each line = one step)
3. Converts to standardized format:

```python
@dataclass
class Transition:
    task_instruction: str      # "Find me a red leather wallet under $50"
    state: str                 # Current page observation (text)
    action: str                # "search[leather wallet]" or "click[B07XYZ123]"
    next_state: str            # Resulting page observation
    reward: float              # 0.0 for intermediate, 0-1 for terminal
    done: bool                 # Whether episode ended
    
@dataclass  
class Trajectory:
    instruction: str
    transitions: List[Transition]
    total_reward: float
```

4. Save as JSONL file: `data/trajectories.jsonl`

**Key Details**:
- WebShop actions are: `search[query]`, `click[element]`, `click[buy now]`
- Observations include page type (search, item, item_page) and visible elements
- Reward is sparse: 0 during episode, final score (0-1) at purchase based on match quality

### Task 1.4: Explore the Gym Environment

Create a script `explore_env.py` to understand the environment interface:

```python
import gym
from web_agent_site.envs import WebAgentTextEnv

env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)

# Reset and explore
obs = env.reset()
print("Initial observation:", obs[:500])
print("Available actions:", env.get_available_actions())

# Take some actions
obs, reward, done, info = env.step("search[laptop]")
print("After search:", obs[:500])
print("Reward:", reward, "Done:", done)
```

Document the observation format, action space, and reward structure.

---

## Phase 2: Experience Model Training

### Task 2.1: Prepare Training Data for Experience Model

Create `prepare_experience_data.py`:

1. Load trajectories from Phase 1
2. Format each transition as an input-output pair for the LLM:

**Input format** (prompt):
```
Task: {instruction}

Current State:
{state_observation}

Action taken: {action}

Predict the next state and reward.
```

**Output format** (completion):
```
Next State:
{next_state_observation}

Reward: {reward}
Done: {done}
```

3. Split into train/val (90/10)
4. Save as HuggingFace Dataset or JSONL for fine-tuning

### Task 2.2: Fine-tune Experience Model

Create `train_experience_model.py`:

**Recommended base models** (in order of preference):
1. `Qwen/Qwen2.5-3B-Instruct` - Best quality/size tradeoff
2. `meta-llama/Llama-3.2-3B-Instruct` - Good alternative
3. `microsoft/Phi-3-mini-4k-instruct` - Smallest, fastest

**Training approach**:
- Use LoRA/QLoRA for efficient fine-tuning
- Libraries: `transformers`, `peft`, `trl`
- Training config:
  - Learning rate: 2e-4
  - Batch size: 4-8 (gradient accumulation as needed)
  - Epochs: 3-5
  - LoRA rank: 16-32
  - Target modules: q_proj, k_proj, v_proj, o_proj

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Example structure
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # ... config
)

trainer.train()
model.save_pretrained("models/experience_model")
```

### Task 2.3: Validate Experience Model

Create `validate_experience_model.py`:

1. Load trained experience model
2. For held-out trajectories, predict next_state given (state, action)
3. Compute metrics:
   - **State prediction accuracy**: BLEU/ROUGE between predicted and actual next_state
   - **Reward prediction accuracy**: MSE for reward prediction
   - **Terminal state detection**: F1 for done prediction
4. Qualitative inspection: sample 10 predictions, visually inspect quality

---

## Phase 3: Synthetic Trajectory Generation

### Task 3.1: Build Synthetic Rollout Generator

Create `generate_synthetic_rollouts.py`:

```python
class SyntheticEnvironment:
    def __init__(self, experience_model, tokenizer, instructions: List[str]):
        self.model = experience_model
        self.tokenizer = tokenizer
        self.instructions = instructions
        
    def reset(self, instruction: str = None) -> str:
        """Initialize episode with instruction, return initial state"""
        if instruction is None:
            instruction = random.choice(self.instructions)
        self.current_instruction = instruction
        # Generate initial search page state
        self.current_state = self._generate_initial_state(instruction)
        return self.current_state
    
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """Use experience model to predict next state"""
        prompt = self._format_prompt(
            self.current_instruction, 
            self.current_state, 
            action
        )
        
        # Generate prediction
        output = self._generate(prompt)
        next_state, reward, done = self._parse_output(output)
        
        self.current_state = next_state
        return next_state, reward, done, {}
    
    def _format_prompt(self, instruction, state, action) -> str:
        return f"""Task: {instruction}

Current State:
{state}

Action taken: {action}

Predict the next state and reward."""

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Task 3.2: Generate Synthetic Training Data

1. Load experience model
2. Load task instructions from WebShop (the 12,087 crowd-sourced instructions)
3. For each instruction, run rollouts with a simple policy:
   - Random action selection from available actions
   - Or use a heuristic: search with keywords from instruction, click products, buy
4. Generate 5,000-10,000 synthetic trajectories
5. Filter for quality:
   - Remove trajectories with nonsensical states
   - Keep trajectories with reasonable reward signals
6. Save as `data/synthetic_trajectories.jsonl`

### Task 3.3: Implement Experience Replay Buffer (Optional Enhancement)

For grounding synthetic predictions in reality:

```python
class ExperienceReplayBuffer:
    def __init__(self, real_trajectories: List[Trajectory], capacity: int = 10000):
        self.buffer = []
        for traj in real_trajectories:
            for transition in traj.transitions:
                self.buffer.append(transition)
        self.capacity = capacity
    
    def sample_similar(self, state: str, action: str, k: int = 3) -> List[Transition]:
        """Retrieve similar real transitions to ground predictions"""
        # Simple: embedding similarity or keyword matching
        # Return k most similar real transitions
        pass
```

---

## Phase 4: Agent Training

### Task 4.1: Prepare Agent Training Data

Create `prepare_agent_data.py`:

Convert trajectories to agent training format (SFT style):

**Input** (agent sees):
```
Task: {instruction}

Current observation:
{state}

Available actions: {available_actions}

What action should you take?
```

**Output** (agent learns):
```
{action}
```

Create two datasets:
1. `data/agent_train_real.jsonl` - From real trajectories only
2. `data/agent_train_synthetic.jsonl` - From synthetic trajectories only

### Task 4.2: Train Agent (SFT Baseline)

Create `train_agent.py`:

Train two agents for comparison:
1. **Real-only agent**: Trained on real trajectories
2. **Synthetic-only agent**: Trained on synthetic trajectories

Use same architecture and training config for fair comparison:
- Base model: Same as experience model or smaller (1B-3B)
- LoRA fine-tuning
- Same number of training steps

```python
# Train both variants
for data_source in ["real", "synthetic"]:
    train_dataset = load_dataset(f"data/agent_train_{data_source}.jsonl")
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        # ... same config
    )
    trainer.train()
    model.save_pretrained(f"models/agent_{data_source}")
```

### Task 4.3: (Optional) GRPO Training

For more advanced training, implement GRPO:

```python
# GRPO requires:
# 1. Sample multiple actions per state
# 2. Score each action's resulting trajectory
# 3. Update policy using group-relative advantage

class GRPOTrainer:
    def __init__(self, agent_model, experience_model, reward_fn):
        self.agent = agent_model
        self.env = SyntheticEnvironment(experience_model)
        self.reward_fn = reward_fn
    
    def train_step(self, instructions: List[str], group_size: int = 4):
        for instruction in instructions:
            state = self.env.reset(instruction)
            
            # Sample G actions from agent
            actions = self.sample_actions(state, k=group_size)
            
            # Rollout each action, get rewards
            rewards = []
            for action in actions:
                trajectory = self.rollout(state, action)
                rewards.append(self.reward_fn(trajectory))
            
            # Compute advantages (z-score normalization)
            advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            
            # Update policy
            self.update_policy(state, actions, advantages)
```

---

## Phase 5: Evaluation

### Task 5.1: Evaluate on Real WebShop Environment

Create `evaluate_agents.py`:

```python
def evaluate_agent(agent_model, env, num_episodes: int = 100):
    results = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        instruction = extract_instruction(obs)
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 15:
            action = agent_model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward = reward  # WebShop gives final reward
            steps += 1
        
        results.append({
            "instruction": instruction,
            "reward": total_reward,
            "steps": steps,
            "success": total_reward > 0.5
        })
    
    return results

# Evaluate both agents
for agent_type in ["real", "synthetic"]:
    agent = load_agent(f"models/agent_{agent_type}")
    results = evaluate_agent(agent, env)
    
    print(f"{agent_type} agent:")
    print(f"  Mean reward: {np.mean([r['reward'] for r in results]):.3f}")
    print(f"  Success rate: {np.mean([r['success'] for r in results]):.1%}")
```

### Task 5.2: Ablation Studies

Run experiments to understand what matters:

1. **Synthetic data quantity**: 1K, 5K, 10K, 20K trajectories
2. **Experience model quality**: Vary training epochs
3. **Hybrid training**: Mix real + synthetic data (10%, 30%, 50% real)
4. **Sim-to-real transfer**: Pretrain on synthetic, fine-tune on small real

---

## Project Structure

```
webshop-experience-model/
├── data/
│   ├── trajectories.jsonl              # Extracted real trajectories
│   ├── synthetic_trajectories.jsonl    # Generated synthetic trajectories
│   ├── agent_train_real.jsonl          # Agent training data (real)
│   └── agent_train_synthetic.jsonl     # Agent training data (synthetic)
├── models/
│   ├── experience_model/               # Trained experience model
│   ├── agent_real/                     # Agent trained on real data
│   └── agent_synthetic/                # Agent trained on synthetic data
├── scripts/
│   ├── extract_trajectories.py
│   ├── prepare_experience_data.py
│   ├── train_experience_model.py
│   ├── validate_experience_model.py
│   ├── generate_synthetic_rollouts.py
│   ├── prepare_agent_data.py
│   ├── train_agent.py
│   └── evaluate_agents.py
├── notebooks/
│   ├── 01_explore_webshop.ipynb
│   ├── 02_analyze_trajectories.ipynb
│   └── 03_results_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Dependencies

```txt
# requirements.txt
torch>=2.0
transformers>=4.36
peft>=0.7
trl>=0.7
datasets
accelerate
bitsandbytes
wandb
numpy
pandas
tqdm
gymnasium
```

---

## Compute Requirements

- **GPU**: 1x A40 (48GB) or 1x A100 (40GB) recommended
- **Estimated costs** (at $0.40/hr for A40):
  - Experience model training: ~8 hours (~$3)
  - Synthetic rollout generation: ~4 hours (~$2)  
  - Agent training (x2): ~8 hours (~$3)
  - Evaluation: ~2 hours (~$1)
  - **Total: ~$10-15**

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Experience model state prediction BLEU | >0.5 |
| Synthetic agent vs Real agent reward gap | <10% |
| Synthetic agent success rate | >30% |
| Training time reduction vs real RL | >5x |

---

## Key References

1. **DreamGym Paper**: arXiv:2511.03773 - "Scaling Agent Learning via Experience Synthesis"
2. **WebShop Paper**: arXiv:2207.01206 - "WebShop: Towards Scalable Real-World Web Interaction"
3. **WebShop GitHub**: https://github.com/princeton-nlp/WebShop
4. **GRPO**: DeepSeek-R1 Technical Report

---

## Notes for Coding Agent

1. **Start simple**: Get data extraction working first before any ML
2. **Validate each step**: Don't proceed until current step outputs look correct
3. **Log everything**: Use wandb or tensorboard for experiment tracking
4. **Save checkpoints**: Save models frequently, experiments are expensive
5. **Iterate fast**: Use small data subsets for debugging, scale up for real runs
6. **Document findings**: Keep notes on what works and what doesn't

The key insight from DreamGym: **The experience model IS the environment**. Once you have a good experience model, you can train agents entirely in simulation, then transfer to real environments with minimal fine-tuning.