import torch 
import torch.nn as nn
import logging 

logger = logging.getLogger(__name__)

class LSTMForecastModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, forecast_horizon: int, dropout: float = 0.2):
        super(LSTMForecastModel, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.forecast_horizon = forecast_horizon
        self.dropout_rate = dropout

        # Creating LSTM layers with the specified parameters.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Implementing dropout for regularization — helps prevent overfitting 
        # by randomly zeroing out some of the hidden units during training.
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(hidden_size, forecast_horizon)  # Output a single value for the forecast

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"LSTMForecastModel created: "
            f"input={input_size}, hidden={hidden_size}, "
            f"layers={num_layers}, horizon={forecast_horizon}, "
            f"params={total_params:,}"
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # Take the output of the last time step
        last_hidden = self.dropout(last_hidden)  # Apply dropout to the last hidden state
        forecast = self.fc(last_hidden)      # output shape: (batch_size, forecast_horizon)
        return forecast           # Return shape: (batch_size, forecast_horizon)

    # This method allows us to get predictions without affecting the training state of the model,
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training  # Remember if the model was in training mode
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            forecast = self.forward(x)
        if was_training:
            self.train()  # Restore original training mode
        return forecast

    # This method allows us to create a model instance directly from a configuration object, 
    # which is useful for keeping the code organized and flexible.
    def build_model_from_config(self, config):
        return LSTMForecastModel(
            input_size=config.lstm.model.input_size,
            hidden_size=config.lstm.model.hidden_size,
            num_layers=config.lstm.model.num_layers,
            forecast_horizon=config.lstm.model.forecast_horizon,
            dropout=config.lstm.model.dropout
        )