import os 
import logging
import numpy as np
import torch

from src.forecasting.lstm_model import LSTMForecastModel

logger = logging.getLogger(__name__)


class LSTMPredictor:
    # Loads a pretrained LSTM model from a checkpoint and forecasts future demand
    def __init__(self, model_path: str, device: str=None):
        if not os.path.isfile(model_path):
            raise ValueError(
                f"LSTM model checkpoint not found at: {model_path}"
                f"Run scripts/train_lstm.py first to train and save the model."
            )

        # Device setup
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load the model checkpoint and initialize the LSTM model with the saved configuration and weights
        checkpoint = torch.load(model_path, map_location=self.device)

        # Reconstruct the model architecture from saved configurations
        model_config = checkpoint["model_config"]
        self.model = LSTMForecastModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            forecast_horizon=model_config["forecast_horizon"],
            dropout=model_config["dropout"]
        )

        # Load the saved weights into the model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.forecast_horizon = model_config["forecast_horizon"]
        self.input_size = model_config["input_size"]
        
        logger.info(
            f"LSTMPredictor loaded from {model_path} on device {self.device}."
            f"horizon={self.forecast_horizon}"
            f"device={self.device}"
        )

    def forecast(self, history: np.ndarray) -> np.ndarray:
        # Produce a multi-step forecast given a history of past demand and features.
        
        # Input validation
        if history.ndim != 2:
            raise ValueError(
                f"history must be 2D array with shape (sequence_length, input_size)."
                f"Got shape {history.shape}."
            )
        if history.shape[1] != self.model.input_size:
            raise ValueError(
                f"history has input_size={history.shape[1]}, but model expects {self.model.input_size}."
                f"Check that the input features are correctly aligned and preprocessed."
            )

        x = np.expand_dims(history, axis=0).astype(np.float32)  # Add batch dimension: (1, sequence_length, input_size)

        x_tensor  = torch.FloatTensor(x).to(self.device)  # Convert to tensor and move to device

        with torch.no_grad():
            output = self.model(x_tensor)  # Get forecast from model
        
        forecast = output.cpu().numpy().flatten()  # Move to CPU and convert to numpy
        return np.clip(forecast, 0.0, 1.0)  # Ensure forecasts are non-negative
        
    def forecast_batch(self, histories: np.ndarray) -> np.ndarray:
        # Produces a forecast for a batch of history windows simultaneously
        # Which is more efficient for evaluating on a test set.
        if histories.ndim != 2:
            raise ValueError(
                f"history_batch must be 3D array with shape (batch_size, sequence_length, input_size)."
                f"Got shape {histories.shape}."
            )
        

        x_tensor = torch.FloatTensor(histories.astype(np.float32)).to(self.device)  # Convert to tensor and move to device

        with torch.no_grad():
            output = self.model(x_tensor)  # Get forecast from model
        
        return np.clip(output.cpu().numpy(), 0.0, 1.0)