import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.rl_environment.reward_function import RewardFunction


@pytest.fixture
def rf():
    return RewardFunction(
        holding_cost_per_unit=0.5,
        lost_sales_cost_per_unit=5.0,
        ordering_cost_per_unit=1.0,
    )


def test_reward_is_negative_of_total_cost(rf):
    result = rf.compute(inventory_before=100, demand=80,
                        order_quantity=50, inventory_after=70)
    assert result.reward == -result.total_cost


def test_no_lost_sales_when_inventory_covers_demand(rf):
    result = rf.compute(inventory_before=100, demand=80,
                        order_quantity=0, inventory_after=20)
    assert result.lost_sales_units == 0.0
    assert result.lost_sales_cost == 0.0


def test_lost_sales_when_demand_exceeds_inventory(rf):
    result = rf.compute(inventory_before=50, demand=80,
                        order_quantity=0, inventory_after=0)
    assert result.lost_sales_units == 30.0
    assert result.lost_sales_cost == 30.0 * 5.0


def test_holding_cost_proportional_to_inventory(rf):
    result = rf.compute(inventory_before=100, demand=0,
                        order_quantity=0, inventory_after=100)
    assert result.holding_cost == 100 * 0.5


def test_ordering_cost_proportional_to_quantity(rf):
    result = rf.compute(inventory_before=100, demand=50,
                        order_quantity=80, inventory_after=50)
    assert result.ordering_cost == 80 * 1.0


def test_total_cost_is_sum_of_components(rf):
    result = rf.compute(inventory_before=100, demand=80,
                        order_quantity=50, inventory_after=70)
    expected = result.holding_cost + result.lost_sales_cost + result.ordering_cost
    assert abs(result.total_cost - expected) < 1e-5


def test_zero_order_zero_ordering_cost(rf):
    result = rf.compute(inventory_before=100, demand=50,
                        order_quantity=0, inventory_after=50)
    assert result.ordering_cost == 0.0


def test_reward_components_dataclass_fields(rf):
    result = rf.compute(100, 80, 50, 70)
    assert hasattr(result, "holding_cost")
    assert hasattr(result, "lost_sales_cost")
    assert hasattr(result, "ordering_cost")
    assert hasattr(result, "total_cost")
    assert hasattr(result, "reward")
    assert hasattr(result, "lost_sales_units")
    assert hasattr(result, "holding_units")
