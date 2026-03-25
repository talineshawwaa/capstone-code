#   Generates plots showing inventory dynamics during evaluation episodes.
#   Shows inventory level, order quantities, and demand over time.

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def plot_inventory_trajectory(
    step_records: List[Dict],
    agent_name:   str  = "Agent",
    save_path:    Optional[str] = None,
    show:         bool = False,
) -> None:
    # Plots inventory level, demand and order quantities over one episode
    steps       = np.arange(len(step_records))
    inventory   = np.array([s["inventory"]       for s in step_records])
    demand      = np.array([s["demand"]           for s in step_records])
    order_qty   = np.array([s["order_quantity"]   for s in step_records])
    lost_sales  = np.array([s["lost_sales"]       for s in step_records])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Inventory Dynamics — {agent_name}", fontsize=14, fontweight="bold")

    # Panel 1: Inventory level vs demand 
    axes[0].fill_between(steps, inventory, alpha=0.3, color="#2196F3")
    axes[0].plot(steps, inventory, color="#2196F3", linewidth=1.5, label="Inventory Level")
    axes[0].plot(steps, demand,    color="#FF5722", linewidth=1.2, label="Demand", alpha=0.8)
    axes[0].set_ylabel("Units")
    axes[0].set_title("Inventory Level vs Demand")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Order quantities
    axes[1].bar(steps, order_qty, color="#4CAF50", alpha=0.8, label="Order Quantity")
    axes[1].set_ylabel("Units Ordered")
    axes[1].set_title("Replenishment Orders")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Lost sales (stockouts)
    axes[2].bar(steps, lost_sales, color="#F44336", alpha=0.8, label="Lost Sales")
    axes[2].set_ylabel("Units")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Lost Sales (Unmet Demand)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_all_agents_inventory(
    all_results:  Dict,
    save_path:    Optional[str] = None,
    show:         bool = False,
) -> None:
    # Plots inventory trajectories for all 4 agents on the same figure
    colors = {
        "RL+LSTM Agent":      "#2196F3",
        "Standard RL Agent":  "#4CAF50",
        "Fixed-S Policy":     "#FF9800",
        "Forecast Base-Stock": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for agent_name, result in all_results.items():
        if not result.get("all_steps"):
            continue
        steps     = np.arange(len(result["all_steps"]))
        inventory = np.array([s["inventory"] for s in result["all_steps"]])
        color     = colors.get(agent_name, "gray")
        ax.plot(steps, inventory, label=agent_name, color=color,
                linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Day")
    ax.set_ylabel("Inventory Level (units)")
    ax.set_title("Inventory Level Comparison — All Agents")
    ax.legend(loc="upper right")
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
