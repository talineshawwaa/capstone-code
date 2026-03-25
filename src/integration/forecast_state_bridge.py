import logging
import numpy as np
from typing import Optional, Callable

try:
    from src.forecasting.predictor import LSTMPredictor
except ModuleNotFoundError:
    LSTMPredictor = object 

from src.integration.rolling_forecast import RollingForecastWindow

logger = logging.getLogger(__name__)

class ForecastStateBridge:
    # This is what bridges the gap between forecasting and demand
    # Connects the LSTM predictor into the RL inventory environment
    def __init__(
            self, 
            predictor, 
            sequence_length: int, 
            n_features: int,
            inverse_transform: Optional[Callable] = None,
            forecast_horizon: int = 7,
        ):
        self.predictor = predictor
        self.inverse_transform = inverse_transform
        self.forecast_horizon = forecast_horizon
        self._fallback_value = 0.0 # Used when the rolling window is not full yet 

        # Rolling Window
        self.window = RollingForecastWindow(
            sequence_length=sequence_length,
            n_features=n_features
        )

        self._last_forecast: Optional[np.ndarray] = None

        has_transform = 'yes' if inverse_transform else 'no'
        logger.info(
            f"ForecastStateBridge initialised | "
            f"seq_len={sequence_length}, n_features={n_features}, "
            f"horizon={forecast_horizon}, inverse_transform={has_transform}"
        )

    def initialise(self, feature_matrix: np.ndarray) -> None:
        # Fills the rolling window with the historical feature rows in the very beginning
        self.window.initialise(feature_matrix)
        self._last_forecast = None
        logger.debug(f"Bridge initialized with {self.window.current_length} rows")
    
    def initialize(self, feature_matrix: np.ndarray) -> None:
        self.initialise(feature_matrix)

    def update(self, feature_row: np.ndarray) -> None:
        # Adds the latest timestep's feature vector to the slidig window
        self.window.update(feature_row)
        self._last_forecast = None
    
    def get_forecast(self, demand_history: np.ndarray = None) -> np.ndarray:
        # Returns the LSTM forecast for the current window state
        if not self.window.is_ready():
            return np.zeros(self.forecast_horizon, dtype=np.float32)

        # Returns cached forecast if window has not changed since last call
        if self._last_forecast is not None:
            return self._last_forecast

        # Get the current window
        window_array = self.window.get_window()

        # Run LSTM inference
        scaled_forecast = self.predictor.forecast(window_array)

        # Converts the scaled prediction back to actual demand units needed for the action space
        if self.inverse_transform is not None:
            try:
                real_forecast = self.inverse_transform(scaled_forecast)
                real_forecast = np.clip(real_forecast, 0.0, None)
            except Exception as e:
                logger.warning(f"Inverse transform failed: {e}, using scaled values")
                real_forecast = scaled_forecast
        else:
            real_forecast = scaled_forecast
        
        self._last_forecast = real_forecast.astype(np.float32)
        return self._last_forecast

    def get_scaled_forecast(self) -> Optional[np.ndarray]:
        # Returns the raw scaled LSTM forecast (before inverse transform) needed for state space vector
        if not self.window.is_ready():
            return None
        window_array = self.window.get_window()
        return self.predictor.forecast(window_array)
    
    @staticmethod
    # Builds a ForecastStateBrige directly from the ConfigLoader, to ensure consistency with the Config files
    def build_from_config(
        cfg,
        predictor,
        inverse_transform: Optional[Callable],
        n_features: int
    ) -> "ForecastStateBridge":
        return ForecastStateBridge(
            predictor=predictor,
            sequence_length=cfg.lstm.data.sequence_length,
            n_features=n_features,
            inverse_transform=inverse_transform,
            forecast_horizon=cfg.lstm.data.forecast_horizon
        )