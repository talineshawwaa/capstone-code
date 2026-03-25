import os
import sys
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.feature_engineering import FeatureEngineer
from src.data.sequence_builder import SequenceBuilder
from src.forecasting.lstm_model import LSTMForecastModel
from src.forecasting.trainer import LSTMTrainer
from src.forecasting.metrics import compute_per_horizon_metrics

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train_lstm.log")),
        ],
    )

def main():
    # Step 1: Load configuration and set up logging
    cfg = ConfigLoader()
    setup_logging(cfg.base.paths.logs)
    logger = logging.getLogger(__name__)

    logger.info("LSTM TRAINING PIPELINE")
    logger.info(cfg)

    # Set random seeds for reproducibility
    np.random.seed(cfg.base.project.random_seed)

    # Step 2: Load raw data
    logger.info("Loading raw data...")
    raw_data_path = os.path.join(cfg.base.paths.data_raw, "retail_store_inventory.csv")

    loader = DataLoader(raw_data_path)
    df_raw = loader.load()
    pairs = loader.get_store_product_pairs(df_raw)
    logger.info(f"Loaded raw data with {len(df_raw)} rows and {len(pairs)} unique store-product pairs.")

    # Step 3: Preprocess data
    logger.info("Preprocessing data...")
    scaler_dir = os.path.join(cfg.base.paths.model_lstm, "scalers")
    pre = Preprocessor(scaler_save_dir=scaler_dir)
    df_clean = pre.fit_transform(df_raw)
    logger.info(f"Preprocessed data. Sample after preprocessing:\n{df_clean.shape}")

    # Step 4: Feature engineering
    logger.info("Performing feature engineering...")
    fe = FeatureEngineer(demand_history_length=cfg.rl.environment.demand_history_length)
    df_featured = fe.transform(df_clean)
    logger.info(f"Featured shapes: {df_featured.shape}")
    logger.info(f"Numeric features: {len(fe.numeric_feature_columns)}")
    logger.info(f"Categorical features: {len(fe.categorical_feature_columns)}")

    # Step 5: Build sequences for LSTM training
    logger.info("\n Building LSTM sequence")
    feature_cols = fe.numeric_feature_columns + fe.categorical_feature_columns
    n_features = len(feature_cols) + 1

    builder = SequenceBuilder(
        sequence_length=cfg.lstm.data.sequence_length,
        forecast_horizon=cfg.lstm.data.forecast_horizon,
        train_split=cfg.lstm.data.train_split,
        val_split=cfg.lstm.data.val_split,
    )

    splits = builder.build(df_featured, feature_cols)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test: X={X_test.shape}, y={y_test.shape}")

    # Step 6: Initialize the LSTM model
    logger.info("\nTraining LSTM model...")
    actual_input_size = X_train.shape[2]
    logger.info(f"Actual input size for LSTM: {actual_input_size} (features={n_features})")

    model = LSTMForecastModel(
        input_size=actual_input_size,
        hidden_size=cfg.lstm.model.hidden_size,
        num_layers=cfg.lstm.model.num_layers,
        forecast_horizon=cfg.lstm.model.output_size,
        dropout=cfg.lstm.model.dropout
    )

    # Step 7: Train the model
    logger.info("\n LSTM Model Training")

    trainer = LSTMTrainer(
        model=model,
        save_dir=cfg.base.paths.model_lstm,
        learning_rate=cfg.lstm.training.learning_rate,
        batch_size=cfg.lstm.training.batch_size,
        epochs=cfg.lstm.training.epochs,
        patience=cfg.lstm.training.early_stopping_patience,
    )

    history = trainer.train(X_train, y_train, X_val, y_val)

    # Step 8: Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)

    logger.info("\nLSTM training and evaluation complete.")
    logger.info(f"Test Metrics Results")
    logger.info(f"  MAE: {test_metrics['mae']:.4f} (scaled units)")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f} (scaled units)")

    model.eval()
    device = next(model.parameters()).device
    x_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_preds = model(x_tensor).cpu().numpy()
    
    per_step = compute_per_horizon_metrics(y_test, y_preds)
    logger.info("\nPer-day forecast accuracy:")

    for h in range(cfg.lstm.data.forecast_horizon):
        logger.info(f"  Day {h+1}: MAE={per_step['mae_per_step'][h]:.4f}, RMSE={per_step['rmse_per_step'][h]:.4f}")

    logger.info("\nTraining Complete.")
    logger.info(f"Model saved to: {cfg.base.paths.model_lstm}best_model.pt")
    logger.info(F"Scalers saved to {scaler_dir}")
    logger.info(f"Training ran for {len(history['train_loss'])} epochs with best validation loss {history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()