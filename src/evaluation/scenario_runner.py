import os
import json
import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)


class ScenarioRunner:
    # Runs all 4 agents across all 3 evaluation scenarios

    def __init__(self, cfg, results_dir: str = "outputs/results/"):
        self.cfg         = cfg
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run_all(
        self,
        df_featured,
        preprocessor,
        feature_cols: List[str],
        train_store:   str,
        train_product: str,
    ) -> Dict:
        # Runs the full evalution matrix for 4 agents for all 3 scenarios
        
        from src.evaluation.scenario_builder import ScenarioBuilder
        from src.evaluation.backtester import Backtester

        builder = ScenarioBuilder(df_featured, preprocessor, self.cfg)

        # Build scenario pair lists
        scenarios = {
            "A": builder.build_scenario_a(n_pairs=10),
            "B": builder.build_scenario_b(n_pairs=10),
            "C": builder.build_scenario_c(
                train_store=train_store,
                train_product=train_product,
                n_test_pairs=10,
            ),
        }

        # Load agents
        agents = self._load_all_agents(
            df_featured, preprocessor, feature_cols,
            train_store, train_product
        )

        all_results = {}

        for scenario_name, pair_configs in scenarios.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"SCENARIO {scenario_name}")
            logger.info(f"{'='*50}")

            scenario_results = {}

            for pair_config in pair_configs:
                store_id   = pair_config["store_id"]
                product_id = pair_config["product_id"]
                demand     = pair_config["demand_real"]
                pair_key   = f"{store_id}_{product_id}"

                logger.info(f"\nPair: {store_id}/{product_id}")

                pair_results = {}

                for agent_name, agent_info in agents.items():
                    logger.info(f"  Running {agent_name}...")

                    # Build environment for this pair
                    env = self._build_env(
                        demand=demand,
                        use_forecast=agent_info["use_forecast"],
                        forecast_provider=agent_info.get("forecast_provider"),
                    )

                    backtester = Backtester(
                        env=env,
                        agent=agent_info["agent"],
                        n_episodes=5,
                        agent_name=agent_name,
                    )

                    result = backtester.run()
                    pair_results[agent_name] = result["aggregated"]

                scenario_results[pair_key] = pair_results

            all_results[f"scenario_{scenario_name}"] = scenario_results

            # Save scenario results to disk
            save_path = os.path.join(
                self.results_dir,
                f"scenario_{scenario_name.lower()}",
                "results.json"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(scenario_results, f, indent=2)
            logger.info(f"Saved Scenario {scenario_name} results to {save_path}")

        return all_results

    # PRIVATE HELPERS

    def _load_all_agents(
        self,
        df_featured,
        preprocessor,
        feature_cols,
        train_store,
        train_product,
    ) -> Dict:
        # Loads all 4 agents and returns them in a dict
        import torch
        from src.agents.dqn_agent import DQNAgent
        from src.baselines.fixed_s_policy import FixedSPolicy
        from src.baselines.forecast_basestock_policy import ForecastBasestockPolicy
        from src.forecasting.predictor import LSTMPredictor
        from src.integration.forecast_state_bridge import ForecastStateBridge
        from src.rl_environment.action_space import ActionSpace

        cfg = self.cfg

        # Build action space (shared by all agents)
        act_cfg = cfg.rl.action_space
        action_space = ActionSpace(
            delta_min=act_cfg.delta_min,
            delta_max=act_cfg.delta_max,
            delta_step=act_cfg.delta_step,
            lead_time=cfg.rl.environment.lead_time,
            max_inventory=cfg.rl.environment.max_inventory,
        )

        # Build forecast bridge (shared by LSTM-using agents)
        model_path = os.path.join(cfg.base.paths.model_lstm, "best_lstm_model.pt")
        predictor  = LSTMPredictor(model_path)

        inverse_fn = lambda x: preprocessor.inverse_transform_demand(
            x, train_store, train_product
        )

        # Get feature matrix for bridge initialisation
        mask = (
            (df_featured["store_id"] == train_store) &
            (df_featured["product_id"] == train_product)
        )
        df_pair      = df_featured[mask].sort_values("date").reset_index(drop=True)
        feature_matrix = df_pair[["units_sold"] + feature_cols].values.astype(np.float32)
        n_features     = feature_matrix.shape[1]

        bridge = ForecastStateBridge(
            predictor=predictor,
            sequence_length=cfg.lstm.data.sequence_length,
            n_features=n_features,
            inverse_transform=inverse_fn,
            forecast_horizon=cfg.lstm.data.forecast_horizon,
        )
        bridge.initialise(feature_matrix[:cfg.lstm.data.sequence_length])

        # Build a temporary env to get state_dim
        temp_demand = df_pair["units_sold"].values.astype(np.float32) * 500.0
        temp_env    = self._build_env(temp_demand, use_forecast=True,
                                      forecast_provider=bridge.get_forecast)
        state_dim   = temp_env.observation_dim
        n_actions   = temp_env.n_actions

        # 1. RL + LSTM Agent 
        rl_lstm = DQNAgent(state_dim=state_dim, n_actions=n_actions, cfg=cfg)
        rl_lstm_path = os.path.join(
            cfg.base.paths.model_rl, "rl_lstm_agent", "rl_lstm_agent_best.pt"
        )
        if os.path.exists(rl_lstm_path):
            rl_lstm.load(rl_lstm_path)
        else:
            logger.warning(f"RL+LSTM checkpoint not found: {rl_lstm_path}")

        # 2. Standard RL Agent (no forecast) 
        std_rl = DQNAgent(state_dim=state_dim, n_actions=n_actions, cfg=cfg)
        std_rl_path = os.path.join(
            cfg.base.paths.model_rl, "standard_rl_agent", "standard_rl_agent_best.pt"
        )
        if os.path.exists(std_rl_path):
            std_rl.load(std_rl_path)
        else:
            logger.warning(f"Standard RL checkpoint not found: {std_rl_path}")

        # 3. Fixed-S Policy 
        fixed_s = FixedSPolicy(
            reorder_point=cfg.rl.baselines.fixed_s.reorder_point,
            order_quantity=cfg.rl.baselines.fixed_s.order_quantity,
            action_space=action_space,
            max_inventory=cfg.rl.environment.max_inventory,
        )

        # 4. Forecast Base-Stock Policy 
        basestock = ForecastBasestockPolicy(
            safety_stock=cfg.rl.baselines.forecast_basestock.safety_stock,
            forecast_provider=bridge.get_forecast,
            lead_time=cfg.rl.environment.lead_time,
            action_space=action_space,
        )

        return {
            "RL+LSTM Agent": {
                "agent":            rl_lstm,
                "use_forecast":     True,
                "forecast_provider": bridge.get_forecast,
            },
            "Standard RL Agent": {
                "agent":            std_rl,
                "use_forecast":     False,
                "forecast_provider": None,
            },
            "Fixed-S Policy": {
                "agent":            fixed_s,
                "use_forecast":     False,
                "forecast_provider": None,
            },
            "Forecast Base-Stock": {
                "agent":            basestock,
                "use_forecast":     True,
                "forecast_provider": bridge.get_forecast,
            },
        }

    def _build_env(self, demand, use_forecast, forecast_provider):
        # Builds an InventoryEnv for evaluation
        from src.rl_environment.inventory_env import InventoryEnv
        return InventoryEnv(
            demand_sequence=demand,
            forecast_provider=forecast_provider,
            cfg=self.cfg,
            use_forecast=use_forecast,
        )
