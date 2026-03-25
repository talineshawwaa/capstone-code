import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

try:
    import torch
    from src.forecasting.lstm_model import LSTMForecastModel, build_model_from_config
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def make_model(input_size=10, hidden_size=32, num_layers=2,
               forecast_horizon=7, dropout=0.2):
    return LSTMForecastModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_horizon=forecast_horizon,
        dropout=dropout,
    )


def test_model_output_shape():
    # Output shape should be (batch_size, forecast_horizon)
    model = make_model()
    x     = torch.randn(8, 30, 10)   # batch=8, seq=30, features=10
    out   = model(x)
    assert out.shape == (8, 7)


def test_model_single_sample():
    # Works with batch size of 1
    model = make_model()
    x   = torch.randn(1, 30, 10)
    out = model(x)
    assert out.shape == (1, 7)


def test_dropout_rate_stored():
    # dropout_rate attribute must be stored for checkpoint saving
    model = make_model(dropout=0.3)
    assert model.dropout_rate == 0.3


def test_predict_no_grad():
    # predict() should work without gradients
    model = make_model()
    x     = torch.randn(4, 30, 10)
    out   = model.predict(x)
    assert out.shape == (4, 7)


def test_model_eval_mode_after_predict():
    # Model should return to training mode after predict() if it was training
    model = make_model()
    model.train()
    model.predict(torch.randn(2, 30, 10))
    assert model.training


def test_forward_is_deterministic_in_eval():
    # In eval mode, same input should give same output
    model = make_model()
    model.eval()
    x    = torch.randn(2, 30, 10)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)
