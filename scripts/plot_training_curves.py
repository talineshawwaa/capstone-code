# scripts/plot_training_curves.py

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from src.visualizations.reward_plots import plot_both_agents_training

cfg = ConfigLoader()

# Load histories
with open(os.path.join(cfg.base.paths.results, "rl_lstm_training_history.json")) as f:
    rl_lstm_history = json.load(f)

with open(os.path.join(cfg.base.paths.results, "rl_standard_training_history.json")) as f:
    standard_history = json.load(f)

# Plot both agents
plot_both_agents_training(
    rl_lstm_rewards=rl_lstm_history["episode_rewards"],
    standard_rewards=standard_history["episode_rewards"],
    window_size=50,
    save_path=os.path.join(cfg.base.paths.plots, "training_curves_comparison.png"),
    show=False,
)

print("Training curve plot saved to outputs/plots/training_curves_comparison.png")