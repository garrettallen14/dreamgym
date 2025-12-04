"""Core data types for WebShop Experience Model."""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class Transition:
    """A single state transition in an episode."""
    
    task_instruction: str  # "Find me a red leather wallet under $50"
    state: str  # Current page observation (text)
    action: str  # "search[leather wallet]" or "click[B07XYZ123]"
    next_state: str  # Resulting page observation
    reward: float  # 0.0 for intermediate, 0-1 for terminal
    done: bool  # Whether episode ended
    
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
    """A complete episode trajectory."""
    
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
class ExperienceModelSample:
    """Training sample for the experience model."""
    
    prompt: str  # Input: task + state + action
    completion: str  # Output: next_state + reward + done
    
    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}


@dataclass
class AgentTrainingSample:
    """Training sample for the agent."""
    
    prompt: str  # Input: task + state + available_actions
    completion: str  # Output: action
    
    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}


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
