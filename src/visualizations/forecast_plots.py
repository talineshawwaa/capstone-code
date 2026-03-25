#   Generates plots for the LSTM forecasting results.
#   Used in the thesis results chapter to show forecast quality.

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    dates:        Optional[np.ndarray] = None,
    title:        str  = "LSTM Demand Forecast: Actual vs Predicted",
    save_path:    Optional[str] = None,
    show:         bool = False,
) -> None:
    # Plots actual vs predicted demand over time
    
    # Flatten to 1D if multi-step
    if y_true.ndim == 2:
        y_true = y_true[:, 0]
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    x = dates if dates is not None else np.arange(len(y_true))

    # Top panel: actual vs predicted
    axes[0].plot(x, y_true, label="Actual Demand",    color="#2196F3", linewidth=1.5, alpha=0.9)
    axes[0].plot(x, y_pred, label="LSTM Forecast",    color="#FF5722", linewidth=1.5, alpha=0.9, linestyle="--")
    axes[0].set_ylabel("Demand (units)")
    axes[0].set_title("Actual vs Predicted Demand")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: forecast error 
    error = y_true - y_pred
    axes[1].bar(x, error, color=["#4CAF50" if e >= 0 else "#F44336" for e in error],
                alpha=0.7, width=0.8)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Forecast Error (units)")
    axes[1].set_title("Forecast Error (Actual − Predicted)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_per_horizon_metrics(
    mae_per_step:  np.ndarray,
    rmse_per_step: np.ndarray,
    save_path:     Optional[str] = None,
    show:          bool = False,
) -> None:
    # Plots MAE and RMSE broken down by forecast day
    horizon = len(mae_per_step)
    days    = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, mae_per_step,  marker="o", label="MAE",  color="#2196F3", linewidth=2)
    ax.plot(days, rmse_per_step, marker="s", label="RMSE", color="#FF5722", linewidth=2)
    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("Error (scaled units)")
    ax.set_title("Forecast Accuracy by Horizon Day")
    ax.set_xticks(days)
    ax.set_xticklabels([f"Day {d}" for d in days])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def _save_or_show(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
