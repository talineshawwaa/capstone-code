import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.feature_engineering import FeatureEngineer
from src.evaluation.scenario_runner import ScenarioRunner


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "run_scenarios.log")),
        ],
    )


def main():
    cfg = ConfigLoader()
    setup_logging(cfg.base.paths.logs)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SCENARIO EVALUATION PIPELINE")
    logger.info("=" * 60)

    np.random.seed(cfg.base.project.random_seed)

    # Step 1: Load and preprocess data 
    logger.info("Loading data...")
    raw_path = os.path.join(cfg.base.paths.data_raw, "retail_store_inventory.csv")
    loader   = DataLoader(raw_path)
    df_raw   = loader.load()

    scaler_dir = os.path.join(cfg.base.paths.model_lstm, "scalers")
    pre = Preprocessor(scaler_save_dir=scaler_dir)
    pre.load_scalers()
    df_clean = pre.transform(df_raw)

    fe = FeatureEngineer(
        demand_history_length=cfg.rl.environment.demand_history_length
    )
    df_featured = fe.transform(df_clean)
    feature_cols = fe.numeric_feature_columns + fe.categorical_feature_columns

    # Step 2: Set training pair (same pair used for RL agent training)
    pairs      = loader.get_store_product_pairs(df_raw)
    train_store   = pairs[0][0]
    train_product = pairs[0][1]
    logger.info(f"Training pair: {train_store}/{train_product}")

    # Step 3: Run all scenarios 
    runner = ScenarioRunner(
        cfg=cfg,
        results_dir=cfg.base.paths.results,
    )

    results = runner.run_all(
        df_featured=df_featured,
        preprocessor=pre,
        feature_cols=feature_cols,
        train_store=train_store,
        train_product=train_product,
    )

    logger.info("\n--- Scenario evaluation complete ---")
    logger.info(f"Results saved to: {cfg.base.paths.results}")


if __name__ == "__main__":
    main()
