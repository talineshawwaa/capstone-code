import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

from src.forecasting.lstm_model import LSTMForecastModel
from src.forecasting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Trains an LSTM model for forecasting and provides a method to load the trained model
class LSTMTrainer:
    def __init__(
            self, model: LSTMForecastModel, 
            save_dir: str = "models/lstm",
            learning_rate: float = 0.001,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 10,
            device: str = None  
        ):
        self.model = model
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # Device Setup
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        logger.info(f"LSTMTrainer initialized on device {self.device}.")

        # Loss function, optimizer, and learning rate scheduler setup
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.save_dir, "best_lstm_model.pt")

    # Run the full training loop
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, List[float]]:
        # Build DataLoaders for training and validation sets
        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._make_dataloader(X_val, y_val, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        logger.info(
            f"Starting training for {self.epochs} epochs with batch size {self.batch_size} and learning rate {self.learning_rate}."
            f"train_samples={len(X_train)}, val_samples={len(X_val)}"
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss = self._run_epoch(train_loader, training=True) # Training
            val_loss = self._run_epoch(val_loader, training=False) # Validation

            # Adjust learning rate based on validation loss
            self.scheduler.step(val_loss) 

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            elapsed = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
        
        # Save best model based on validation loss and implement early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            self._save_checkpoint(epoch, val_loss)
            logger.info(f"New best model saved with val_loss={best_val_loss:.6f} at epoch {epoch}.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= self.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs with no improvement."
                        f"Best val_loss={best_val_loss:.6f} at epoch {epoch - epochs_no_improve}."
                )
            
        logger.info(f"Training completed. Best val_loss={best_val_loss:.6f} at epoch {epoch - epochs_no_improve}.")

        # Load the best model weights before returning the training history
        self._load_best_checkpoint()
        return history
    
    # Evaluate the model on the test set and compute overall metrics
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        test_loader = self._make_dataloader(X_test, y_test, shuffle=False)
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_batch.numpy())
        
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        metrics = compute_all_metrics(y_true, y_pred)
        
        # Print the computed metrics for immediate console output
        print(f"Evaluation Metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return metrics
    
    # Helper method to create DataLoader from numpy arrays
    def _make_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        # Convert numpy arrays to PyTorch tensors and create a DataLoader for batching
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        # This method runs one epoch of training or validation depending on the 'training' flag.
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        n_batches = 0

        context = torch.no_grad() if not training else torch.enable_grad()
        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                if training:
                    self.optimizer.zero_grad()

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                if training: 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # Save the model checkpoint with the current epoch, validation loss, and model configuration for reproducibility.
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "model_config": {
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "forecast_horizon": self.model.forecast_horizon,
                "dropout": self.model.dropout_rate,
            }
        }, self.best_model_path)    

    def _load_best_checkpoint(self) -> None:
        # Load the best saved weights back into the model 
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Best model loaded from {self.best_model_path} (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}).")