import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.integration.rolling_forecast import RollingForecastWindow
from src.integration.forecast_state_bridge import ForecastStateBridge


class MockPredictor:
    # Mock LSTM predictor that returns fixed forecast values
    forecast_horizon = 7
    input_size       = 10

    def forecast(self, history):
        return np.full(7, 0.4, dtype=np.float32)


# RollingForecastWindow tests

def test_window_not_ready_when_empty():
    w = RollingForecastWindow(sequence_length=30, n_features=5)
    assert not w.is_ready()


def test_window_ready_after_filling():
    w = RollingForecastWindow(sequence_length=5, n_features=3)
    for i in range(5):
        w.update(np.ones(3) * i)
    assert w.is_ready()


def test_window_shape_after_filling():
    w = RollingForecastWindow(sequence_length=5, n_features=3)
    for i in range(5):
        w.update(np.ones(3) * i)
    arr = w.get_window()
    assert arr.shape == (5, 3)


def test_window_evicts_oldest():
    w = RollingForecastWindow(sequence_length=3, n_features=1)
    for i in range(3):
        w.update(np.array([float(i)]))
    w.update(np.array([99.0]))
    arr = w.get_window()
    assert arr[0, 0] == 1.0   # row 0 is now gone, row 1 is first


def test_window_initialise():
    w = RollingForecastWindow(sequence_length=5, n_features=3)
    matrix = np.random.rand(10, 3).astype(np.float32)
    w.initialise(matrix)
    assert w.is_ready()
    assert w.get_window().shape == (5, 3)


def test_window_wrong_feature_size_raises():
    w = RollingForecastWindow(sequence_length=5, n_features=3)
    with pytest.raises(ValueError):
        w.update(np.ones(5))   # wrong size


# ForecastStateBridge tests

def test_bridge_returns_zeros_before_ready():
    bridge = ForecastStateBridge(
        predictor=MockPredictor(),
        sequence_length=5,
        n_features=10,
        forecast_horizon=7,
    )
    forecast = bridge.get_forecast()
    assert np.all(forecast == 0.0)
    assert forecast.shape == (7,)


def test_bridge_returns_forecast_when_ready():
    bridge = ForecastStateBridge(
        predictor=MockPredictor(),
        sequence_length=5,
        n_features=10,
        forecast_horizon=7,
    )
    matrix = np.random.rand(5, 10).astype(np.float32)
    bridge.initialise(matrix)
    forecast = bridge.get_forecast()
    assert forecast.shape == (7,)
    assert not np.all(forecast == 0.0)


def test_bridge_inverse_transform_applied():
    # Inverse transform should scale forecast to real units
    bridge = ForecastStateBridge(
        predictor=MockPredictor(),
        sequence_length=5,
        n_features=10,
        inverse_transform=lambda x: x * 500.0,
        forecast_horizon=7,
    )
    matrix = np.random.rand(5, 10).astype(np.float32)
    bridge.initialise(matrix)
    forecast = bridge.get_forecast()
    # MockPredictor returns 0.4, inverse transform → 200.0
    assert np.allclose(forecast, 200.0, atol=1.0)


def test_bridge_cache_invalidated_after_update():
    bridge = ForecastStateBridge(
        predictor=MockPredictor(),
        sequence_length=5,
        n_features=10,
        forecast_horizon=7,
    )
    matrix = np.random.rand(5, 10).astype(np.float32)
    bridge.initialise(matrix)
    bridge.get_forecast()
    assert bridge._last_forecast is not None
    bridge.update(np.random.rand(10).astype(np.float32))
    assert bridge._last_forecast is None
