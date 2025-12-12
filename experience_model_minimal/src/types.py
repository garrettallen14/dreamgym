"""Core data types for Experience Model training.

These dataclasses define the structure of trajectory data used to train
an Experience Model that predicts state transitions.
"""

from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class Transition:
    """A single state transition in an episode.
    
    The Experience Model learns to predict (next_state, reward, done) 
    given (task_instruction, state, action).
    
    Attributes:
        task_instruction: The task/goal description (e.g., "Find a red wallet under $50")
        state: Current observation/state (text description of current page/screen)
        action: Action taken (e.g., "search[wallet]", "click[Buy Now]")
        next_state: Resulting observation after taking action
        reward: Reward received (0.0 for intermediate steps, 0-1 for terminal)
        done: Whether the episode ended after this transition
    """
    
    task_instruction: str
    state: str
    action: str
    next_state: str
    reward: float
    done: bool
    
    def to_dict(self) -> dict:
        return {
            "task_instruction": self.task_instruction,
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Transition":
        return cls(
            task_instruction=data["task_instruction"],
            state=data["state"],
            action=data["action"],
            next_state=data["next_state"],
            reward=data["reward"],
            done=data["done"],
        )


@dataclass
class Trajectory:
    """A complete episode trajectory.
    
    A trajectory is a sequence of transitions from episode start to end.
    
    Attributes:
        instruction: The task/goal for this episode
        transitions: List of state transitions
        total_reward: Final reward for the episode
        metadata: Optional metadata (source file, etc.)
    """
    
    instruction: str
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "transitions": [t.to_dict() for t in self.transitions],
            "total_reward": self.total_reward,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        return cls(
            instruction=data["instruction"],
            transitions=[Transition.from_dict(t) for t in data["transitions"]],
            total_reward=data["total_reward"],
            metadata=data.get("metadata", {}),
        )
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_jsonl(cls, line: str) -> "Trajectory":
        return cls.from_dict(json.loads(line))


@dataclass
class TrainingSample:
    """A single training sample for the Experience Model.
    
    This is the format used for fine-tuning: a prompt (input) and 
    completion (expected output).
    
    Attributes:
        prompt: Input text (task + current state + action + instruction to predict)
        completion: Expected output (next state + reward + done)
    """
    
    prompt: str
    completion: str
    
    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}
    
    def to_chat_format(self) -> dict:
        """Convert to HuggingFace chat format for SFTTrainer."""
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.completion},
            ]
        }


# =============================================================================
# Utility Functions
# =============================================================================

def save_trajectories(trajectories: List[Trajectory], path: str) -> None:
    """Save trajectories to JSONL file."""
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(traj.to_jsonl() + "\n")


def load_trajectories(path: str) -> List[Trajectory]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                trajectories.append(Trajectory.from_jsonl(line))
    return trajectories
