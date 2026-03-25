import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class ReplayBuffer:
    # Circular replay buffer for storing and sampling DQN experiences

    def __init__(self, capacity: int, state_dim: int):
        self.capacity  = capacity
        self.state_dim = state_dim
        self._ptr      = 0      # write pointer — where next transition goes
        self._size     = 0      # number of transitions currently stored

        # Pre-allocate arrays for each component.
        self._states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions     = np.zeros((capacity,),           dtype=np.int64)
        self._rewards     = np.zeros((capacity,),           dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones       = np.zeros((capacity,),           dtype=np.float32)

        logger.info(
            f"ReplayBuffer: capacity={capacity:,}, state_dim={state_dim}, "
            f"memory≈{capacity * state_dim * 2 * 4 / 1e6:.1f}MB"
        )

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        # Stores one transition in the buffer
        self._states[self._ptr]      = state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr]       = float(done)

        # Advance write pointer, wrapping around when buffer is full
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        # Samples a random mini-batch of transitions
        if self._size < batch_size:
            raise RuntimeError(
                f"Buffer has {self._size} transitions but batch_size={batch_size}. "
                f"Call is_ready() before sample()."
            )

        # Sample random indices without replacement
        indices = np.random.randint(0, self._size, size=batch_size)

        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        )

    def is_ready(self, batch_size: int) -> bool:
        # Returns True if the buffer has enough transitions to sample a batch
        return self._size >= batch_size

    @property
    def size(self) -> int:
        # Number of transitions currently stored in the buffer
        return self._size

    @property
    def is_full(self) -> bool:
        # True if the buffer has reached its maximum capacity
        return self._size == self.capacity

