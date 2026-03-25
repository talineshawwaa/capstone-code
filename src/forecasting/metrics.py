import numpy as np
import logging
from typing import Dict, Union

logger = logging.getLogger(__name__)

# Functions will only accept numpy arrays or lists as input for y_true and y_pred.
ArrayLike = Union[np.ndarray, list]

def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Mean Absolute Error (MAE) between true and predicted values."""
    y_true = np.array(y_true, dtype=np.float32).flatten()
    y_pred = np.array(y_pred, dtype=np.float32).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Root Mean Squared Error (RMSE) between true and predicted values."""
    y_true = np.array(y_true, dtype=np.float32).flatten()
    y_pred = np.array(y_pred, dtype=np.float32).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    results = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
    logger.info(
        f"Forecast metrics — "
        f"MAE: {results['mae']:.4f} | "
        f"RMSE: {results['rmse']:.4f} "
    )
    return results

def compute_per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, np.ndarray]:
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays of shape (n_samples, forecast_horizon). "
            f"Got y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    horizon = y_true.shape[1]
    mae_per_step  = np.array([mae(y_true[:, h], y_pred[:, h]) for h in range(horizon)])
    rmse_per_step = np.array([rmse(y_true[:, h], y_pred[:, h]) for h in range(horizon)])

    return {
        "mae_per_step":  mae_per_step,
        "rmse_per_step": rmse_per_step,
    }