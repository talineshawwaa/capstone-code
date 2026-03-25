import logging
import numpy as np
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

class RollingForecastWindow:
    def __init__(self, sequence_length: int, n_features: int):
        self.sequence_length = sequence_length
        self.n_features = n_features

        # Using deque to append the oldest observation when a new one is added
        self._window: deque = deque(maxlen=sequence_length)

        logger.info(
            f"RollingForecastWindow: seq_len={sequence_length}"
            f"n_features={n_features}"
        )
    
    def update(self, feature_row: np.ndarray) -> None:
        # Adds a new timestep's feature vector the window
        feature_row = np.array(feature_row, dtype=np.float32).flatten()

        if len(feature_row) != self.n_features:
            raise ValueError(
                f"feature_row has {len(feature_row)} features but rolling windows expects {self.n_features}"
            )
        self._window.append(feature_row)

    def initialize(self, feature_matrix: np.ndarray) -> None:
        # Populates the window with an initial block of feature rows
        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        if feature_matrix.ndim != 2:
            raise ValueError(f"feature_matrx must be 2D. Got shape: {feature_matrix.shape}")
        
        self._window.clear()

        rows_to_use = feature_matrix[-self.sequence_length:]
        for row in rows_to_use:
            self._window.append(row)

        logger.debug(f"Window initialized with {len(self._window)}/{self.sequence_length} rows")
    
    def initialise(self, feature_matrix: np.ndarray) -> None:
        self.initialize(feature_matrix)

    def get_window(self) -> Optional[np.ndarray]:
        # Returns the current window as a 2D array ready for LSTM input
        if len(self._window) < self.sequence_length:
            return None

        return np.array(self._window, dtype=np.float32)
    
    def is_ready(self) -> bool:
        # Returns True if the window has enough history for a valid LSTM input
        return len(self._window) >= self.sequence_length
    
    @property
    def current_length(self) -> int:
        return len(self._window)