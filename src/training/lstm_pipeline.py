import os
import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


class LSTMTrainingPipeline:
    """
    End-to-end LSTM training pipeline.

    Wraps the full sequence: load → preprocess → feature engineer →
    build sequences → train → evaluate → save.

    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self) -> Dict:
        # Runs full LSTM pipeline
        import torch
        from src.data.loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.data.feature_engineering import FeatureEngineer
        from src.data.sequence_builder import SequenceBuilder
        from src.forecasting.lstm_model import LSTMForecastModel
        from src.forecasting.trainer import LSTMTrainer
        from src.forecasting.metrics import compute_all_metrics

        cfg = self.cfg
        np.random.seed(cfg.base.project.random_seed)
        torch.manual_seed(cfg.base.project.random_seed)

        # Load
        raw_path = os.path.join(cfg.base.paths.data_raw, "retail_store_inventory.csv")
        loader   = DataLoader(raw_path)
        df_raw   = loader.load()

        # Preprocess
        scaler_dir = os.path.join(cfg.base.paths.model_lstm, "scalers")
        pre = Preprocessor(scaler_save_dir=scaler_dir)
        df_clean = pre.fit_transform(df_raw)

        # Feature engineer
        fe = FeatureEngineer(
            demand_history_length=cfg.rl.environment.demand_history_length
        )
        df_featured = fe.transform(df_clean)

        # Build sequences
        feature_cols = fe.numeric_feature_columns + fe.categorical_feature_columns
        builder = SequenceBuilder(
            sequence_length=cfg.lstm.data.sequence_length,
            forecast_horizon=cfg.lstm.data.forecast_horizon,
            train_split=cfg.lstm.data.train_split,
            val_split=cfg.lstm.data.val_split,
        )
        splits = builder.build(df_featured, feature_cols)
        X_train, y_train = splits["train"]
        X_val,   y_val   = splits["val"]
        X_test,  y_test  = splits["test"]

        # Build model
        actual_input_size = X_train.shape[2]
        model = LSTMForecastModel(
            input_size=actual_input_size,
            hidden_size=cfg.lstm.model.hidden_size,
            num_layers=cfg.lstm.model.num_layers,
            forecast_horizon=cfg.lstm.model.output_size,
            dropout=cfg.lstm.model.dropout,
        )

        # Train
        trainer = LSTMTrainer(
            model=model,
            save_dir=cfg.base.paths.model_lstm,
            learning_rate=cfg.lstm.training.learning_rate,
            batch_size=cfg.lstm.training.batch_size,
            epochs=cfg.lstm.training.epochs,
            patience=cfg.lstm.training.early_stopping_patience,
        )
        history = trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate
        test_metrics = trainer.evaluate(X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")

        return {
            "history":      history,
            "test_metrics": test_metrics,
        }
