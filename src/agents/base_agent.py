from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    # Abstract base class for all agents

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        # Selects an action given the current observation
        pass

    @abstractmethod
    def reset(self) -> None:
        # Resets internal state at the start of a new episode
        pass

    def learn(self, *args, **kwargs) -> None:
        # Update the policy from experience
        pass

    def save(self, path: str) -> None:
        # Save agent weights/state to disk. Override in RL agents
        pass

    def load(self, path: str) -> None:
        # Load agent weights/state from disk. Override in RL agents
        pass
