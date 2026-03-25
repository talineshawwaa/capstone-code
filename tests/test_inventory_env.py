import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from config.config_loader import ConfigLoader
from src.rl_environment.inventory_env import InventoryEnv


@pytest.fixture
def cfg():
    return ConfigLoader()


@pytest.fixture
def demand():
    np.random.seed(42)
    return np.random.normal(80, 20, size=365).clip(0).astype(np.float32)


@pytest.fixture
def env(cfg, demand):
    return InventoryEnv(
        demand_sequence=demand,
        forecast_provider=None,
        cfg=cfg,
        use_forecast=False,
    )


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert "inventory" in info


def test_obs_shape(env, cfg):
    obs, _ = env.reset()
    assert obs.shape == (env.observation_dim,)


def test_obs_values_in_range(env):
    obs, _ = env.reset()
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_step_returns_correct_tuple(env):
    env.reset()
    result = env.step(8)   # delta=0 action
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_is_negative(env):
    # Reward should always be <= 0 (it's negative cost)
    env.reset()
    for _ in range(10):
        _, reward, terminated, truncated, _ = env.step(8)
        assert reward <= 0.0
        if terminated or truncated:
            break


def test_inventory_never_negative(env):
    # On-hand inventory should never go below 0
    env.reset()
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(8)
        assert env.inventory >= 0.0
        if terminated or truncated:
            break


def test_step_count_increments(env):
    env.reset()
    assert env.step_count == 0
    env.step(8)
    assert env.step_count == 1


def test_reset_restores_initial_inventory(env, cfg):
    env.reset()
    for _ in range(20):
        env.step(8)
    env.reset()
    assert env.inventory == float(cfg.rl.environment.initial_inventory)


def test_episode_terminates(env):
    # Episode must eventually terminate
    env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        _, _, terminated, truncated, _ = env.step(8)
        done = terminated or truncated
        steps += 1
    assert done, "Episode did not terminate within 500 steps"
