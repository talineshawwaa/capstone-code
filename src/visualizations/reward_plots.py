#   Plots training reward curves for the RL agents.
#   Shows that the agent is learning over episodes.

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def plot_training_curve(
    episode_rewards: List[float],
    agent_name:      str  = "Agent",
    window_size:     int  = 50,
    save_path:       Optional[str] = None,
    show:            bool = False,
) -> None:
    # Plots the training reward curve with a rolling average overlay
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards  = np.array(episode_rewards)

    # Compute rolling average
    rolling_avg = np.convolve(
        rewards,
        np.ones(window_size) / window_size,
        mode="valid"
    )
    rolling_x = np.arange(window_size, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw rewards (faint)
    ax.plot(episodes, rewards, color="#90CAF9", linewidth=0.8,
            alpha=0.5, label="Episode Reward")

    # Rolling average (bold)
    ax.plot(rolling_x, rolling_avg, color="#1565C0", linewidth=2.0,
            label=f"Rolling Avg (window={window_size})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"Training Reward Curve — {agent_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_both_agents_training(
    rl_lstm_rewards:   List[float],
    standard_rewards:  List[float],
    window_size:       int  = 50,
    save_path:         Optional[str] = None,
    show:              bool = False,
) -> None:
    # Plots training curves for both RL agents side by side for comparison
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle("Training Reward Comparison", fontsize=14, fontweight="bold")

    for ax, rewards, name, color in zip(
        axes,
        [rl_lstm_rewards, standard_rewards],
        ["RL + LSTM Agent", "Standard RL Agent (No Forecast)"],
        ["#1565C0", "#2E7D32"],
    ):
        episodes    = np.arange(1, len(rewards) + 1)
        rewards_arr = np.array(rewards)
        rolling     = np.convolve(rewards_arr, np.ones(window_size) / window_size, mode="valid")
        rolling_x   = np.arange(window_size, len(rewards_arr) + 1)

        ax.plot(episodes, rewards_arr, color=color, linewidth=0.8, alpha=0.3)
        ax.plot(rolling_x, rolling, color=color, linewidth=2.0, label=f"Avg (w={window_size})")
        ax.set_title(name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
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
