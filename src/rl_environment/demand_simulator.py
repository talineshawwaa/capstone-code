import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class DemandSimulator:
    # Provides demad values to the inventory environment at each time step
    def __init__(
            self,
            demand_sequence: np.ndarray,
            mode: str = "replay",
            seed: int = 42,
    ):
        if mode not in ["replay", "sample"]:
            raise ValueError(f"mode must be 'replay' or 'sample', got {mode}")
        
        self.demand_sequence = np.array(demand_sequence, dtype=np.float32)
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        self._mean = float(np.mean(demand_sequence))
        self._std = float(np.std(demand_sequence))

        self._current_index = 0

        logger.info(
            f"DemandSimulator: mode={mode}, mean={self._mean:.2f}, std={self._std:.2f}, n_days={len(demand_sequence)}"
        )

    def reset(self, start_index: int = 0) -> None:
        # Resets the simulator to the beginning of the demand sequence 
        self._current_index = start_index
    
    def step(self) -> Optional[float]:
        # Returns the demand of the current timestep and advances the counter
        if self.mode == "replay":
            if self._current_index >= len(self.demand_sequence):
                return None  # End of sequence
            demand = float(self.demand_sequence[self._current_index])
            self._current_index += 1
            return demand
        else:
            demand = self.rng.normal(self._mean, self._std)
            return float(max(0.0, demand))  # Ensure non-negative demand
        
    
    def peek(self, n_steps: int = 1) -> np.ndarray:
        # Returns the next n_steps demand values without advancing the counter
        if self.mode == "replay":
            end = min(self._current_index + n_steps, len(self.demand_sequence))
            values = self.demand_sequence[self._current_index:end]

            if len(values) < n_steps:
                values = np.pad(values, (0, n_steps - len(values)))
            return values
        else:
            samples = self.rng.normal(self._mean, self._std, size=n_steps)
            return np.clip(samples, 0.0, None).astype(np.float32)
    
    def get_history(self, n_steps: int) -> np.ndarray:
        # Returns the previous demand sequence without considering the current step we are in
        start = max(0, self._current_index - n_steps)
        history = self.demand_sequence[start:self._current_index]

        if len(history) < n_steps:
            history = np.pad(history, (n_steps - len(history), 0))
        
        return history.astype(np.float32)
    
    @property
    def remaining_steps(self) -> int:
        # Number of steps remaining
        return max(0, len(self.demand_sequence) - self._current_index)
    
    @property
    def current_index(self) -> int:
        # Current position
        return self._current_index
