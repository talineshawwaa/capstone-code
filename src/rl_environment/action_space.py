import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class ActionSpace:
    # Manages the discrete delta adjustments action space
    def __init__(
            self,
            delta_min: int = -200,
            delta_max: int = 200,
            delta_step: int =25,
            lead_time: int = 7,
            max_inventory: int = 500
    ):
        # Build the delta grid
        self.deltas = np.arange(delta_min, delta_max + 1, delta_step, dtype=np.float32)
        self.n_actions = len(self.deltas)
        self.lead_time = lead_time
        self.max_inventory = max_inventory

        logger.info(f"ActionSpace: {self.n_actions} actions, deltas from {delta_min} to {delta_max} in steps of {delta_step}, lead time {lead_time} days, max inventory {max_inventory}")

    def get_delta(self, action_index: int) -> float:
        # Returns the delta value for a given action index
        if not (0 <= action_index < self.n_actions):
            raise ValueError(f"Invalid action index {action_index}. Must be between 0 and {self.n_actions - 1}.")
        return float(self.deltas[action_index])

    def compute_order_quantity(
            self, 
            action_index: int, 
            forecast: np.ndarray,
            rolling_std: float,
            inventory_position: float,
        ) -> float:

        # Translates an action index into an actual replenishment quantity
        delta = self.get_delta(action_index)
        
        # Step 1: Base-stock Computation
        steps_to_sum = min(self.lead_time, len(forecast))
        base_stock = float(np.sum(forecast[:steps_to_sum]))

        safety_stock = rolling_std * np.sqrt(self.lead_time)

        target = max(0.0, base_stock + delta + safety_stock)

        # Step 2: Order Quantity Determination
        order_quantity = max(0.0, target - inventory_position)
        order_quantity = min(order_quantity, max(0.0, self.max_inventory - inventory_position))

        return order_quantity

    def sample(self) -> int:
        # Returns a random action index that is used during 
        return int(np.random.randint(0, self.n_actions))

    def get_all_deltas(self) -> List[float]:
        # returns full list of delta values
        return self.deltas.tolist()
        
        