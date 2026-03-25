import os
import sys
import logging
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from src.training.rl_pipeline import RLTrainingPipeline


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train_rl_agent.log")),
        ],
    )


def main():
    cfg = ConfigLoader()
    setup_logging(cfg.base.paths.logs)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("RL + LSTM AGENT TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(cfg)

    pipeline = RLTrainingPipeline(
        cfg=cfg,
        use_forecast=True,           # ← uses LSTM forecast in state
        agent_name="rl_lstm_agent",
    )

    results = pipeline.run()
    # Save training history
    history_path = os.path.join(cfg.base.paths.results, "rl_lstm_training_history.json")
    with open(history_path, "w") as f:
        json.dump(results["history"], f)
    logger.info(f"Training history saved to {history_path}")

    logger.info("\n--- Training Complete ---")
    logger.info(f"Episodes completed:  {results['episodes']}")
    logger.info(f"Total steps:         {results['total_steps']:,}")
    logger.info(f"Agent saved to:      {results['agent_path']}")

    rewards = results["history"]["episode_rewards"]
    if rewards:
        logger.info(f"Final avg reward (last 50 episodes): {np.mean(rewards[-50:]):.2f}")

    history_path = os.path.join(cfg.base.paths.results, "rl_lstm_training_history.json")
    with open(history_path, "w") as f:
        json.dump(results["history"], f)
    logger.info(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()
