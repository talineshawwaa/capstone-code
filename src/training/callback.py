import os
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TrainingCallback:
    # Tracks training progress and handles checkpointing during RL training
    # Saves checkpoint whenever a new best average reward is achieved
    def __init__(
        self,
        save_dir:      str,
        agent_name:    str  = "agent",
        window_size:   int  = 50,
        log_interval:  int  = 100,
        save_interval: int  = 500,
    ):
        self.save_dir      = save_dir
        self.agent_name    = agent_name
        self.window_size   = window_size
        self.log_interval  = log_interval
        self.save_interval = save_interval

        os.makedirs(save_dir, exist_ok=True)

        # Tracking
        self.episode_rewards:  List[float] = []
        self.episode_lengths:  List[int]   = []
        self.best_avg_reward:  float       = float("-inf")
        self.episode_count:    int         = 0
        self.total_steps:      int         = 0

    def on_episode_end(
        self,
        episode_reward: float,
        episode_length: int,
        agent,
        epsilon:        Optional[float] = None,
    ) -> None:
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count   += 1
        self.total_steps     += episode_length

        # Compute rolling average reward over last window_size episodes
        recent_rewards = self.episode_rewards[-self.window_size:]
        avg_reward     = float(np.mean(recent_rewards))

        # Log progress at regular intervals
        if self.episode_count % self.log_interval == 0:
            avg_length = float(np.mean(self.episode_lengths[-self.window_size:]))
            eps_str    = f", epsilon={epsilon:.4f}" if epsilon is not None else ""
            logger.info(
                f"Episode {self.episode_count:5d} | "
                f"Avg Reward (last {self.window_size}): {avg_reward:8.2f} | "
                f"Avg Length: {avg_length:.0f} | "
                f"Total Steps: {self.total_steps:,}"
                f"{eps_str}"
            )

        # Save best checkpoint when average reward improves
        if avg_reward > self.best_avg_reward and len(recent_rewards) >= self.window_size:
            self.best_avg_reward = avg_reward
            best_path = os.path.join(self.save_dir, f"{self.agent_name}_best.pt")
            agent.save(best_path)
            logger.info(
                f"New best avg reward: {avg_reward:.2f} → saved to {best_path}"
            )

        # Save periodic checkpoint regardless of performance
        if self.episode_count % self.save_interval == 0:
            periodic_path = os.path.join(
                self.save_dir,
                f"{self.agent_name}_ep{self.episode_count}.pt"
            )
            agent.save(periodic_path)

    def get_history(self) -> Dict[str, List]:
        """Returns the full training history for plotting."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }

