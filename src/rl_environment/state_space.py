import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class StateSpace:
    # Constructs and normalizes the RL state vector at each time step
    def __init__(
            self,
            lead_time: int = 7,
            demand_history_length: int = 30,
            forecast_horizon: int = 7,
            num_external_numeric: int = 4,
            num_external_categorical: int = 0,
            max_inventory: float = 500.0,
            max_demand: float = 500.0,
            use_forecast: bool = True,
    ):
        self.lead_time = lead_time
        self.demand_history_length = demand_history_length
        self.forecast_horizon = forecast_horizon
        self.num_external_numeric = num_external_numeric
        self.num_external_categorical = num_external_categorical
        self.max_inventory = max_inventory
        self.max_demand = max_demand
        self.use_forecast = use_forecast

        # Calculate total state dimension
        self.state_dim = (
            1
            + 1
            + lead_time
            + demand_history_length
            + forecast_horizon
            + num_external_numeric
            + num_external_categorical
        )

        logger.info(
            f"StateSpace: dim{self.state_dim}"
            f"[inventory(2) + pipeline({lead_time})] +"
            f"[history({demand_history_length})] +"
            f"[forecast({forecast_horizon})] +"
            f"[numeric({num_external_numeric})] +"
            f"[categorical({num_external_categorical})]"
        )
    
    # Builds and returns the normalized state vector s_t.
    def build(
            self,
            inventory: float,
            inventory_position: float,
            pipeline: np.ndarray,
            demand_history: np.ndarray,
            lstm_forecast: np.ndarray,
            external_numeric: Optional[np.ndarray] = None,
            external_categorical: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        components = []

        # 1. Reactive: inventory and inventory position
        # Normalize inventory and inventory position by max_inventory to keep values in [0, 1]
        components.append(
            np.array([
                inventory          / self.max_inventory,
                inventory_position / self.max_inventory,
            ], dtype=np.float32)
        )

        # 2. Reactive: pipeline vector c_t
        # Each element is units arriving in a future period, normalized by max_inventory
        pipeline = self._pad_or_trim(pipeline, self.lead_time)
        components.append((pipeline / self.max_inventory).astype(np.float32))

        # 3. Historical: rolling demand history
        # Normalize by max_demand
        demand_history = self._pad_or_trim(demand_history, self.demand_history_length)
        components.append((demand_history / self.max_demand).astype(np.float32))

        # 4. Predictive: LSTM forecast  
        lstm_forecast = self._pad_or_trim(lstm_forecast, self.forecast_horizon)
        if self.use_forecast:
            components.append((lstm_forecast / self.max_demand).astype(np.float32))
        else:
            components.append(np.zeros(self.forecast_horizon, dtype=np.float32))
        
        # 5. Contextual: numeric external features
        if self.num_external_numeric > 0:
            if external_numeric is None:
                ext_num = np.zeros(self.num_external_numeric, dtype=np.float32)
            else:
                ext_num = self._pad_or_trim(np.array(external_numeric, dtype=np.float32), self.num_external_numeric)
            components.append(ext_num)
        
        # 6. Contextual: categorical external features
        if self.num_external_categorical > 0:
            if external_categorical is None:
                ext_cat = np.zeros(self.num_external_categorical, dtype=np.float32)
            else:
                ext_cat = self._pad_or_trim(np.array(external_categorical, dtype=np.float32), self.num_external_categorical)
            components.append(ext_cat)
        
        # Concatenate all components into a single state vector
        state = np.concatenate(components)
        state = np.clip(state, 0.0, 1.0)
        return state

    def _pad_or_trim(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        arr = np.array(arr, dtype=np.float32).flatten()
        if len(arr) < target_length:
            arr = np.pad(arr, (target_length - len(arr), 0))
        elif len(arr) > target_length:
            arr = arr[-target_length:]
        return arr    