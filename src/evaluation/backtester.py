import logging
import numpy as np
from typing import Dict, List, Optional

from src.evaluation.kpi_calculator import KPICalculator

logger = logging.getLogger(__name__)

class Backtester:
    # Evaluates any agent on the inventory environment and computes KPIs
    def __init__(
        self,
        env,
        agent,
        n_episodes: int  = 10,
        agent_name: str  = "agent",
    ):
        self.env        = env
        self.agent      = agent
        self.n_episodes = n_episodes
        self.agent_name = agent_name
        self.calculator = KPICalculator()

    def run(self) -> Dict:
        # Runs n_episodes evaluations episodes and returns aggregated KPIs
        
        # Disable exploration for evaluation
        original_epsilon = None
        if hasattr(self.agent, "epsilon"):
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0

        all_episode_kpis = []
        all_steps        = []

        logger.info(
            f"Backtester: evaluating {self.agent_name} "
            f"for {self.n_episodes} episodes..."
        )

        for episode in range(self.n_episodes):
            episode_steps = []
            self.agent.reset()
            obs, _ = self.env.reset()

            while True:
                action = self.agent.act(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Record this step
                episode_steps.append(info)

                if terminated or truncated:
                    break

            # Compute KPIs for this episode
            episode_kpis = self.calculator.compute(episode_steps)
            all_episode_kpis.append(episode_kpis)
            all_steps.extend(episode_steps)

            logger.info(
                f"  Episode {episode+1}/{self.n_episodes} | "
                f"Reward: {episode_kpis['total_cumulative_reward']:.2f} | "
                f"Lost Sales: {episode_kpis['total_lost_sales']:.1f} | "
                f"Service Level: {episode_kpis['service_level']:.3f}"
            )

        # Aggregate across episodes
        aggregated = self.calculator.aggregate(all_episode_kpis)

        logger.info(
            f"Backtester: {self.agent_name} complete | "
            f"Avg Reward: {aggregated['total_cumulative_reward']:.2f} | "
            f"Avg Service Level: {aggregated['service_level']:.3f}"
        )

        # Restore original epsilon
        if original_epsilon is not None:
            self.agent.epsilon = original_epsilon

        return {
            "agent_name":    self.agent_name,
            "episode_kpis":  all_episode_kpis,
            "aggregated":    aggregated,
            "all_steps":     all_steps,
        }
