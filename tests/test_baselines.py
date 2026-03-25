import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from config.config_loader import ConfigLoader
from src.rl_environment.action_space import ActionSpace
from src.baselines.fixed_s_policy import FixedSPolicy
from src.baselines.forecast_basestock_policy import ForecastBasestockPolicy

@pytest.fixture
def cfg():
    return ConfigLoader()


@pytest.fixture
def action_space(cfg):
    act_cfg = cfg.rl.action_space
    return ActionSpace(
        delta_min=act_cfg.delta_min,
        delta_max=act_cfg.delta_max,
        delta_step=act_cfg.delta_step,
        lead_time=cfg.rl.environment.lead_time,
        max_inventory=cfg.rl.environment.max_inventory,
    )


@pytest.fixture
def low_inventory_state():
    # State with inventory = 40 (below reorder point of 50)
    state = np.zeros(50, dtype=np.float32)
    state[0] = 40.0 / 500.0
    state[1] = 40.0 / 500.0
    return state


@pytest.fixture
def high_inventory_state():
    # State with inventory = 300 (above reorder point of 50)
    state = np.zeros(50, dtype=np.float32)
    state[0] = 300.0 / 500.0
    state[1] = 300.0 / 500.0
    return state


# FixedSPolicy tests 

def test_fixed_s_orders_when_below_reorder_point(action_space, low_inventory_state, cfg):
    policy = FixedSPolicy(
        reorder_point=cfg.rl.baselines.fixed_s.reorder_point,
        order_quantity=cfg.rl.baselines.fixed_s.order_quantity,
        action_space=action_space,
        max_inventory=cfg.rl.environment.max_inventory,
    )
    action = policy.act(low_inventory_state)
    # Should return the largest delta (order most)
    assert action == action_space.n_actions - 1


def test_fixed_s_no_order_above_reorder_point(action_space, high_inventory_state, cfg):
    policy = FixedSPolicy(
        reorder_point=cfg.rl.baselines.fixed_s.reorder_point,
        order_quantity=cfg.rl.baselines.fixed_s.order_quantity,
        action_space=action_space,
        max_inventory=cfg.rl.environment.max_inventory,
    )
    action = policy.act(high_inventory_state)
    # Should return zero delta action
    deltas = action_space.get_all_deltas()
    assert deltas[action] == 0.0


def test_fixed_s_reset_does_nothing(action_space, cfg):
    policy = FixedSPolicy(50, 100, action_space)
    policy.reset()   # should not raise


def test_fixed_s_name(action_space):
    policy = FixedSPolicy(50, 100, action_space)
    assert isinstance(policy.name, str)
    assert len(policy.name) > 0


# ForecastBasestockPolicy tests

def test_basestock_always_returns_zero_delta(action_space, low_inventory_state, cfg):
    def mock_forecast(h=None):
        return np.ones(7) * 60.0

    policy = ForecastBasestockPolicy(
        safety_stock=cfg.rl.baselines.forecast_basestock.safety_stock,
        forecast_provider=mock_forecast,
        lead_time=cfg.rl.environment.lead_time,
        action_space=action_space,
    )
    action = policy.act(low_inventory_state)
    deltas = action_space.get_all_deltas()
    assert deltas[action] == 0.0


def test_basestock_same_action_regardless_of_inventory(action_space, cfg):
    # Base-stock always returns delta=0 regardless of inventory level
    def mock_forecast(h=None):
        return np.ones(7) * 60.0

    policy = ForecastBasestockPolicy(20, mock_forecast, 7, action_space)

    state_low  = np.zeros(50, dtype=np.float32)
    state_low[0] = 0.05
    state_high = np.zeros(50, dtype=np.float32)
    state_high[0] = 0.9

    assert policy.act(state_low) == policy.act(state_high)


def test_basestock_name(action_space):
    policy = ForecastBasestockPolicy(20, lambda h: np.ones(7), 7, action_space)
    assert isinstance(policy.name, str)
    assert len(policy.name) > 0


def test_basestock_reset_does_nothing(action_space):
    policy = ForecastBasestockPolicy(20, lambda h: np.ones(7), 7, action_space)
    policy.reset()   # should not raise
