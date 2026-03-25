import logging
import numpy as np
from typing import Optional

from src.agents.dqn_agent import DQNAgent

logger = logging.getLogger(__name__)


class StandardRLAgent(DQNAgent):
    # DQN agent trained without LSTM forecast in the state vector
    def __init__(
        self,
        state_dim: int,
        n_actions:  int,
        cfg,
        device:    Optional[str] = None,
    ):
        # Initialise the parent DQNAgent with identical settings
        super().__init__(
            state_dim=state_dim,
            n_actions=n_actions,
            cfg=cfg,
            device=device,
        )
        logger.info(
            f"StandardRLAgent (no LSTM forecast): "
            f"state_dim={state_dim}, n_actions={n_actions}"
        )

    @property
    def name(self) -> str:
        return "Standard RL Agent (No Forecast)"

    def save(self, path: str) -> None:
        # Saves to the standard_rl_agent checkpoint directory
        super().save(path)
        logger.info(f"StandardRLAgent saved to {path}")

    def load(self, path: str) -> None:
        # Loads from a standard_rl_agent checkpoint
        super().load(path)
        logger.info(f"StandardRLAgent loaded from {path}")
