import numpy as np
import logging
from src.baselines.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ForecastBasestockPolicy(BasePolicy):
    # Classical base-stock policy driven by the LSTM demand forecast
    def __init__(
        self,
        safety_stock:      float,
        forecast_provider,
        lead_time:         int,
        action_space,
        state_inv_idx:     int   = 0,
        state_ip_idx:      int   = 1,
        max_inventory:     float = 500.0,
    ):
        self.safety_stock      = safety_stock
        self.forecast_provider = forecast_provider
        self.lead_time         = lead_time
        self.action_space      = action_space
        self.state_inv_idx     = state_inv_idx
        self.state_ip_idx      = state_ip_idx
        self.max_inventory     = max_inventory

        # Find the zero-delta action index — this policy always uses delta=0
        # (follow the base-stock exactly, no RL adjustment)
        deltas = action_space.get_all_deltas()
        self._zero_delta_idx = deltas.index(0.0) if 0.0 in deltas else len(deltas) // 2

        logger.info(
            f"ForecastBasestockPolicy: safety_stock={safety_stock}, "
            f"lead_time={lead_time}"
        )

    @property
    def name(self) -> str:
        return "Forecast Base-Stock Policy"

    def act(self, obs: np.ndarray) -> int:
        # Computes the base-stock target using the LSTM forecast and return the action index that will place the appropriate order
        return self._zero_delta_idx

    def reset(self) -> None:
        # Nothing to reset — this policy has no internal state
        pass
