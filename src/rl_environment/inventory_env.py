import logging
import numpy as np
from typing import Optional, Dict, Tuple, Any

from src.rl_environment.reward_function import RewardFunction, RewardComponents
from src.rl_environment.state_space import StateSpace
from src.rl_environment.action_space import ActionSpace
from src.rl_environment.demand_simulator import DemandSimulator

logger = logging.getLogger(__name__)

class InventoryEnv:
    def __init__(
            self,
            demand_sequence: np.ndarray,
            forecast_provider: Optional[callable],
            cfg,
            external_features = None,
            use_forecast: bool = True,
            seed: int = 42
    ):
        self.cfg = cfg
        self.use_forecast = use_forecast
        self.external_features = external_features
        self.seed = seed

        env_cfg = cfg.rl.environment
        act_cfg = cfg.rl.action_space

        self.demand_sim = DemandSimulator(
            demand_sequence=demand_sequence,
            mode="replay",
            seed=seed
        )

        self.action_space = ActionSpace(
            delta_min=act_cfg.delta_min,
            delta_max=act_cfg.delta_max,
            delta_step=act_cfg.delta_step,
            lead_time=env_cfg.lead_time,
            max_inventory=env_cfg.max_inventory
        )

        self.state_space = StateSpace(
            lead_time=env_cfg.lead_time,
            demand_history_length=env_cfg.demand_history_length,
            forecast_horizon=env_cfg.forecast_horizon,
            num_external_numeric=env_cfg.num_external_numeric,
            num_external_categorical=env_cfg.num_external_categorical,
            max_inventory=float(env_cfg.max_inventory),
            max_demand=float(demand_sequence.max()) if len(demand_sequence) > 0 else 500.0,
            use_forecast=use_forecast,
        )

        self.reward_fn = RewardFunction(
            holding_cost_per_unit=env_cfg.holding_cost_per_unit,
            lost_sales_cost_per_unit=env_cfg.stockout_penalty_per_unit,
            ordering_cost_per_unit=env_cfg.ordering_cost_per_unit
        )

        self.forecast_provider = forecast_provider

        self.initial_inventory = env_cfg.initial_inventory
        self.max_inventory = env_cfg.max_inventory
        self.lead_time = env_cfg.lead_time
        self.episode_length = env_cfg.episode_length
        self.demand_history_length = env_cfg.demand_history_length

        self.observation_dim = self.state_space.state_dim
        self.n_actions = self.action_space.n_actions

        self._inventory = 0.0
        self._pipeline = np.zeros(self.lead_time, dtype=np.float32)
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_reward_components: Optional[RewardComponents] = None
        self._rolling_std = 0.0

        logger.info(
            f"InventoryEnv Created |"
            f"obs_dim={self.observation_dim} | "
            f"n_actions={self.n_actions} | "
            f"use_forecast={use_forecast}"
        )

    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        
        self._inventory = float(self.initial_inventory)
        self._pipeline = np.zeros(self.lead_time, dtype=np.float32)
        self.demand_sim.reset(start_index=0)

        self._step_count = 0
        self._episode_reward = 0.0
        self._rolling_std = 0.0

        obs = self._build_state()
        info = {"step": 0, "inventory": self._inventory}

        return obs, info

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Step 1: Get demand forecast for action computation
        demand_history = self.demand_sim.get_history(self.demand_history_length)
        lstm_forecast = self._get_forecast(demand_history)

        # Step 2: Compute rolling std of recent demand
        if len(demand_history) > 1: 
            self._rolling_std = float(np.std(demand_history))
        else:
            self._rolling_std = 0.0
    
        # 3. Compute inventory position IP_t = b_t + sum(pipeline) 
        inventory_position = self._inventory + float(np.sum(self._pipeline))

        # 4. Compute order quantity from action 
        order_quantity = self.action_space.compute_order_quantity(
            action_index=action,
            forecast=lstm_forecast,
            rolling_std=self._rolling_std,
            inventory_position=inventory_position,
        )

        # 5. Realise demand
        demand = self.demand_sim.step()
        if demand is None:
            # Sequence exhausted — terminate episode
            obs = self._build_state()
            return obs, 0.0, True, False, {"reason": "sequence_exhausted"}

        # 6. Update inventory (lost-sales mechanism) 
        inventory_before       = self._inventory
        inventory_after_demand = max(0.0, self._inventory - demand)

        # 7. Process pipeline arrivals 
        arriving_order    = float(self._pipeline[0])
        inventory_after   = min(
            inventory_after_demand + arriving_order,
            float(self.max_inventory)   # cap at warehouse capacity
        )

        # 8. Shift pipeline and append new order 
        self._pipeline = np.roll(self._pipeline, -1)
        self._pipeline[-1] = order_quantity   # new order goes to end of pipeline

        # Update on-hand inventory
        self._inventory = inventory_after

        # 9. Compute reward
        reward_components = self.reward_fn.compute(
            inventory_before=inventory_before,
            demand=demand,
            order_quantity=order_quantity,
            inventory_after=inventory_after,
        )
        self._last_reward_components = reward_components
        reward = reward_components.reward
        self._episode_reward += reward

        # 10. Advance step counter
        self._step_count += 1
        terminated = (self.demand_sim.remaining_steps == 0)
        truncated  = (self._step_count >= self.episode_length)

        # 11. Build new state 
        obs = self._build_state()

        # 12. Build info dict 
        info = {
            "step":             self._step_count,
            "demand":           demand,
            "order_quantity":   order_quantity,
            "inventory":        self._inventory,
            "inventory_position": inventory_position,
            "lost_sales":       reward_components.lost_sales_units,
            "holding_cost":     reward_components.holding_cost,
            "lost_sales_cost":  reward_components.lost_sales_cost,
            "ordering_cost":    reward_components.ordering_cost,
            "total_cost":       reward_components.total_cost,
            "reward":           reward,
            "episode_reward":   self._episode_reward,
        }

        return obs, reward, terminated, truncated, info
    
    # PRIVATE HELPERS

    def _get_forecast(self, demand_history: np.ndarray) -> np.ndarray:
        # Gets the LSTM forecast for the current demand history.
        if not self.use_forecast or self.forecast_provider is None:
            return np.zeros(self.cfg.rl.environment.forecast_horizon, dtype=np.float32)

        try:
            return self.forecast_provider(demand_history)
        except Exception as e:
            logger.warning(f"Forecast failed at step {self._step_count}: {e}")
            return np.zeros(self.cfg.rl.environment.forecast_horizon, dtype=np.float32)

    def _build_state(self) -> np.ndarray:
        # Constructs the full state vector s_t from current environment state.
        demand_history = self.demand_sim.get_history(self.demand_history_length)
        lstm_forecast  = self._get_forecast(demand_history)
        inventory_position = self._inventory + float(np.sum(self._pipeline))

        # External features (if provided)
        ext_num = None
        ext_cat = None
        if self.external_features is not None and self._step_count < len(self.external_features):
            row = self.external_features.iloc[self._step_count]
            # Numeric: price, discount, competitor_pricing, holiday_promotion
            ext_num = np.array([
                row.get("price", 0.0),
                row.get("discount", 0.0),
                row.get("competitor_pricing", 0.0),
                row.get("holiday_promotion", 0.0),
            ], dtype=np.float32)

        return self.state_space.build(
            inventory=self._inventory,
            inventory_position=inventory_position,
            pipeline=self._pipeline.copy(),
            demand_history=demand_history,
            lstm_forecast=lstm_forecast,
            external_numeric=ext_num,
            external_categorical=ext_cat,
        )

    @property
    def inventory(self) -> float:
        # Current on-hand inventory level
        return self._inventory

    @property
    def step_count(self) -> int:
        # Number of steps taken in current episode
        return self._step_count