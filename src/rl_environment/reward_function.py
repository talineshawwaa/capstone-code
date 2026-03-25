import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Stores the individual cost components of a single reward computation, as well as the final reward value. 
@dataclass
class RewardComponents:
    holding_cost: float
    lost_sales_cost: float
    ordering_cost: float
    total_cost: float
    reward: float
    lost_sales_units: float
    holding_units: float

# Computes the reward based on the inventory transition
class RewardFunction:
    def __init__(self, holding_cost_per_unit: float = 0.5, lost_sales_cost_per_unit: float = 5.0, ordering_cost_per_unit: float = 1.0):
        self.h = holding_cost_per_unit
        self.l = lost_sales_cost_per_unit
        self.k = ordering_cost_per_unit

    def compute(self, inventory_before: float, demand: float, order_quantity: float, inventory_after: float) -> RewardComponents:
        # Calculate lost sales
        lost_sales_units = max(0.0, demand - inventory_before)
        lost_sales_cost = self.l * lost_sales_units

        # Calculate holding cost
        holding_units = max(0.0, inventory_after)
        holding_cost = self.h *holding_units

        # Calculate ordering cost
        ordering_cost = self.k * order_quantity 

        # Total cost is the sum of all costs
        total_cost = holding_cost + lost_sales_cost + ordering_cost

        # Reward is negative total cost (we want to minimize costs)
        reward = -total_cost

        return RewardComponents(
            holding_cost=holding_cost,
            lost_sales_cost=lost_sales_cost,
            ordering_cost=ordering_cost,
            total_cost=total_cost,
            reward=reward,
            lost_sales_units=lost_sales_units,
            holding_units=holding_units
        )