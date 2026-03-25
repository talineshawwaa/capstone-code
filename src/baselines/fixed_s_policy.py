import numpy as np
import logging
from src.baselines.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class FixedSPolicy(BasePolicy):
    # Fixed reorder-point (s, Q) inventory policy

    def __init__(
        self,
        reorder_point:  float,
        order_quantity: float,
        action_space,
        state_inv_idx:  int   = 0,
        max_inventory:  float = 500.0,
    ):
        self.reorder_point  = reorder_point
        self.order_quantity = order_quantity
        self.action_space   = action_space
        self.state_inv_idx  = state_inv_idx
        self.max_inventory  = max_inventory

        # Pre-compute which action index to use when ordering.
        deltas = action_space.get_all_deltas()
        self._order_action_idx  = len(deltas) - 1  # largest positive delta = order most
        self._no_order_action_idx = deltas.index(0.0) if 0.0 in deltas else len(deltas) // 2

        logger.info(
            f"FixedSPolicy: reorder_point={reorder_point}, "
            f"order_quantity={order_quantity}"
        )

    @property
    def name(self) -> str:
        return "Fixed-S Policy"

    def act(self, obs: np.ndarray) -> int:
        # Returns order action if inventory <= reorder_point, else no-order
        # Denormalise inventory from [0,1] back to real units
        inventory = obs[self.state_inv_idx] * self.max_inventory

        if inventory <= self.reorder_point:
            return self._order_action_idx
        else:
            return self._no_order_action_idx

    def reset(self) -> None:
        # Nothing to reset — this policy has no internal state
        pass

