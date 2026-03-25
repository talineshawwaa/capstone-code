import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class RLTrainingPipeline:
    # Runs the full RL pipeline

    def __init__(
        self,
        cfg,
        use_forecast: bool = True,
        agent_name:   str  = "rl_lstm_agent",
        store_id:     Optional[str] = None,
        product_id:   Optional[str] = None,
    ):
        self.cfg          = cfg
        self.use_forecast = use_forecast
        self.agent_name   = agent_name
        self.store_id     = store_id
        self.product_id   = product_id

    def run(self) -> Dict:
        # Runs full training pipeline

        import torch
        from src.data.loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.data.feature_engineering import FeatureEngineer
        from src.rl_environment.inventory_env import InventoryEnv
        from src.rl_environment.action_space import ActionSpace
        from src.agents.dqn_agent import DQNAgent
        from src.training.callback import TrainingCallback

        cfg = self.cfg
        np.random.seed(cfg.base.project.random_seed)
        torch.manual_seed(cfg.base.project.random_seed)

        # Step 1: Load and preprocess data
        logger.info("Loading and preprocessing data...")
        raw_path = os.path.join(cfg.base.paths.data_raw, "retail_store_inventory.csv")
        loader   = DataLoader(raw_path)
        df_raw   = loader.load()

        scaler_dir = os.path.join(cfg.base.paths.model_lstm, "scalers")
        pre = Preprocessor(scaler_save_dir=scaler_dir)
        pre.load_scalers()
        df_clean = pre.transform(df_raw)

        fe = FeatureEngineer(
            demand_history_length=cfg.rl.environment.demand_history_length
        )
        df_featured = fe.transform(df_clean)

        # Step 2: Select store-product pair
        pairs = loader.get_store_product_pairs(df_raw)
        store_id   = self.store_id   or pairs[0][0]
        product_id = self.product_id or pairs[0][1]
        logger.info(f"Training on: {store_id} / {product_id}")

        mask = (
            (df_featured["store_id"] == store_id) &
            (df_featured["product_id"] == product_id)
        )
        df_pair = df_featured[mask].sort_values("date").reset_index(drop=True)

        # Extract demand in real units (inverse transform)
        demand_scaled = df_pair["units_sold"].values.astype(np.float32)
        demand_real   = pre.inverse_transform_demand(
            demand_scaled, store_id, product_id
        ).astype(np.float32)

        # Step 3: Build forecast bridge (if using LSTM) 
        forecast_provider = None
        feature_matrix    = None
        n_features        = None

        if self.use_forecast:
            logger.info("Loading LSTM predictor...")
            from src.forecasting.predictor import LSTMPredictor
            from src.integration.forecast_state_bridge import ForecastStateBridge

            model_path = os.path.join(
                cfg.base.paths.model_lstm, "best_lstm_model.pt"
            )
            predictor = LSTMPredictor(model_path)
            n_features = predictor.input_size

            inverse_fn = lambda x: pre.inverse_transform_demand(
                x, store_id, product_id
            )

            bridge = ForecastStateBridge.build_from_config(
                cfg, predictor, inverse_fn, n_features
            )

            # Get feature columns for initialising the bridge window
            feature_cols = fe.numeric_feature_columns + fe.categorical_feature_columns
            feature_matrix = df_pair[["units_sold"] + feature_cols].values.astype(np.float32)
            n_features = feature_matrix.shape[1]

            # Rebuild bridge with correct n_features
            bridge = ForecastStateBridge(
                predictor=predictor,
                sequence_length=cfg.lstm.data.sequence_length,
                n_features=n_features,
                inverse_transform=inverse_fn,
                forecast_horizon=cfg.lstm.data.forecast_horizon,
            )
            bridge.initialise(feature_matrix[:cfg.lstm.data.sequence_length])
            forecast_provider = bridge.get_forecast

        # Step 4: Build environment 
        logger.info("Building inventory environment...")
        env = InventoryEnv(
            demand_sequence=demand_real,
            forecast_provider=forecast_provider,
            cfg=cfg,
            use_forecast=self.use_forecast,
        )

        # Step 5: Build agent 
        logger.info("Building DQN agent...")
        agent = DQNAgent(
            state_dim=env.observation_dim,
            n_actions=env.n_actions,
            cfg=cfg,
        )

        # Step 6: Training loop
        save_dir = os.path.join(cfg.base.paths.model_rl, self.agent_name)
        callback = TrainingCallback(
            save_dir=save_dir,
            agent_name=self.agent_name,
            window_size=50,
            log_interval=100,
            save_interval=500,
        )

        total_timesteps = cfg.rl.dqn.total_timesteps
        logger.info(f"Starting training for {total_timesteps:,} timesteps...")

        total_steps    = 0
        episode        = 0

        while total_steps < total_timesteps:
            obs, _ = env.reset()

            # Re-initialise bridge window at episode start
            if self.use_forecast and feature_matrix is not None:
                bridge.initialise(feature_matrix[:cfg.lstm.data.sequence_length])

            episode_reward = 0.0
            episode_steps  = 0

            while True:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.learn(obs, action, reward, next_obs, done)

                # Update bridge window with latest feature row
                if self.use_forecast and feature_matrix is not None:
                    step_idx = min(env.step_count, len(feature_matrix) - 1)
                    bridge.update(feature_matrix[step_idx])

                obs             = next_obs
                episode_reward += reward
                episode_steps  += 1
                total_steps    += 1

                if done or total_steps >= total_timesteps:
                    break

            episode += 1
            callback.on_episode_end(
                episode_reward=episode_reward,
                episode_length=episode_steps,
                agent=agent,
                epsilon=agent.epsilon,
            )

        # Save final agent
        final_path = os.path.join(save_dir, f"{self.agent_name}_final.pt")
        agent.save(final_path)
        logger.info(f"Training complete. Final agent saved to {final_path}")

        return {
            "history":    callback.get_history(),
            "agent_path": final_path,
            "episodes":   episode,
            "total_steps": total_steps,
        }
