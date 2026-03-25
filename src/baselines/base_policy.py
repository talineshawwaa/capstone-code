from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    # Abstract base class for all rule-based baseline policies
    @property
    @abstractmethod
    def name(self) -> str:
        # Human-readable name for this policy.
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        # Select an action given the current observation
        pass

    @abstractmethod
    def reset(self) -> None:
        # Reset any internal state at the start of a new episode
        pass
