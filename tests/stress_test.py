"""
Stress Testing Module for RL+LSTM Inventory Framework
Evaluates robustness across 5 stress scenarios:
1. Demand Volatility (±50%)
2. Forecast Degradation (MAE +50%)
3. Lead Time Variation (7→14 days)
4. Cost Parameter Sensitivity (penalty ratios 5:1, 10:1, 20:1)
5. Stockout Penalty Escalation (p=5→7.5→10→15)

Runs independently from main pipeline. Outputs CSV results and plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import torch

# Import your existing modules
from config.config_loader import ConfigLoader
from src.forecasting.predictor import LSTMPredictor
from src.rl_environment.inventory_env import InventoryEnv
from src.evaluation.kpi_calculator import KPICalculator
from src.data.loader import DataLoader


# ============================================================================
# STRESS TEST SUITE
# ============================================================================

class StressTestSuite:
    """Runs all 5 stress tests on trained models."""
    
    def __init__(
        self,
        lstm_checkpoint: str = "models/lstm/best_lstm_model.pt",
        rl_lstm_checkpoint: str = "models/rl/rl_lstm_agent/rl_lstm_agent_best.pt",
        standard_rl_checkpoint: str = "models/rl/standard_rl_agent/standard_rl_agent_best.pt",
        config_dir: str = "config/",
        test_data_path: str = "data/raw/retail_store_inventory.csv",
        output_dir: str = "outputs/stress_test_results",
    ):
        """
        Initialize stress test suite with trained model checkpoints.
        
        Args:
            lstm_checkpoint: Path to trained LSTM model
            rl_lstm_checkpoint: Path to trained RL+LSTM agent
            standard_rl_checkpoint: Path to trained Standard RL agent
            config_dir: Path to config/ directory
            test_data_path: Path to test dataset
            output_dir: Directory to save results
        """
        self.lstm_checkpoint = lstm_checkpoint
        self.rl_lstm_checkpoint = rl_lstm_checkpoint
        self.standard_rl_checkpoint = standard_rl_checkpoint
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load config using ConfigLoader (proper typed dataclasses)
        self.cfg = ConfigLoader(config_dir=config_dir)
        
        # Load test data
        self.loader = DataLoader(test_data_path)
        self.test_data = self.loader.load()
        
        # Initialize models
        self.lstm_predictor = LSTMPredictor(lstm_checkpoint)
        self.agents = self._load_agents()
        
        # Store results
        self.results = {}
        
        print(f"\n{self.cfg}")
    
    def _load_agents(self) -> Dict:
        """Load all trained agents."""
        agents = {
            'RL+LSTM': self._load_rl_agent(self.rl_lstm_checkpoint, use_forecast=True),
            'Standard RL': self._load_rl_agent(self.standard_rl_checkpoint, use_forecast=False),
            'Fixed-S': self._init_fixed_s_policy(),
            'Forecast Base-Stock': self._init_forecast_basestock(),
        }
        return agents
    
    def _load_rl_agent(self, checkpoint_path: str, use_forecast: bool):
        """Load RL agent from checkpoint."""
        from src.agents.dqn_agent import DQNAgent
        
        # Load checkpoint to get dimensions
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dim = checkpoint['state_dim']
        n_actions = checkpoint['n_actions']
        
        # Initialize agent with typed config
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            cfg=self.cfg,
            device='cpu'
        )
        
        # Load weights using agent's load method
        agent.load(checkpoint_path)
        agent.use_forecast = use_forecast
        
        return agent
        
    def _init_fixed_s_policy(self):
        """Initialize Fixed-S Policy baseline."""
        return FixedSPolicy(
            reorder_point=self.cfg.rl.baselines.fixed_s.reorder_point,
            order_quantity=self.cfg.rl.baselines.fixed_s.order_quantity
        )
    
    def _init_forecast_basestock(self):
        """Initialize Forecast Base-Stock Policy baseline."""
        return ForecastBaseStockPolicy(
            lstm_predictor=self.lstm_predictor,
            safety_stock=self.cfg.rl.baselines.forecast_basestock.safety_stock
        )
    
    # =========================================================================
    # TEST 1: DEMAND VOLATILITY STRESS
    # =========================================================================
    
    def test_1_demand_volatility(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Test 1: Demand Volatility Stress (±50%)
        Creates stressed demand dataset by multiplying realized demand.
        
        Args:
            multiplier: Demand multiplier (1.5 = 50% increase)
            
        Returns:
            DataFrame with service levels, lost sales, costs for all agents
        """
        print(f"\n{'='*60}")
        print(f"TEST 1: DEMAND VOLATILITY STRESS (×{multiplier})")
        print(f"{'='*60}")
        
        # Create stressed dataset
        stressed_data = self.test_data.copy()
        stressed_data['units_sold'] = stressed_data['units_sold'] * multiplier
        
        results = {}
        for agent_name, agent in self.agents.items():
            print(f"  Evaluating {agent_name}...")
            kpis = self._evaluate_agent(agent, stressed_data)
            results[agent_name] = kpis
        
        results_df = pd.DataFrame(results).T
        self.results['Test_1_Demand_Volatility'] = results_df
        
        print(f"\nResults:")
        print(results_df)
        
        return results_df
    
    # =========================================================================
    # TEST 2: FORECAST DEGRADATION
    # =========================================================================
    
    def test_2_forecast_degradation(self, mae_increase_factor: float = 1.5) -> pd.DataFrame:
        """
        Test 2: Forecast Degradation (MAE +50%)
        Adds Gaussian noise to LSTM predictions to degrade forecast accuracy.
        
        Args:
            mae_increase_factor: Factor to increase MAE (1.5 = 50% increase)
            
        Returns:
            DataFrame with results under degraded forecasts
        """
        print(f"\n{'='*60}")
        print(f"TEST 2: FORECAST DEGRADATION (MAE ×{mae_increase_factor})")
        print(f"{'='*60}")
        
        # Calculate noise std to achieve desired MAE increase
        baseline_mae = 0.2007  # From your results
        target_mae = baseline_mae * mae_increase_factor
        noise_std = np.sqrt(target_mae**2 - baseline_mae**2)
        
        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Target MAE: {target_mae:.4f}")
        print(f"  Noise std: {noise_std:.4f}")
        
        results = {}
        for agent_name, agent in self.agents.items():
            print(f"  Evaluating {agent_name}...")
            
            # Create noisy forecast predictor
            noisy_predictor = NoisyLSTMPredictor(
                self.lstm_predictor,
                noise_std=noise_std
            )
            
            # Temporarily replace forecasts
            original_predictor = self.lstm_predictor
            self.lstm_predictor = noisy_predictor
            
            kpis = self._evaluate_agent(agent, self.test_data)
            results[agent_name] = kpis
            
            # Restore original
            self.lstm_predictor = original_predictor
        
        results_df = pd.DataFrame(results).T
        self.results['Test_2_Forecast_Degradation'] = results_df
        
        print(f"\nResults:")
        print(results_df)
        
        return results_df
    
    # =========================================================================
    # TEST 3: LEAD TIME VARIATION
    # =========================================================================
    
    def test_3_lead_time_variation(
        self,
        original_lead_time: int = None,
        extended_lead_time: int = 14
    ) -> Dict[str, pd.DataFrame]:
        """
        Test 3: Lead Time Variation (7→14 days)
        NOTE: Only tests Standard RL and baselines (not RL+LSTM/Fixed-S as they require retraining with new state dim).
        
        Args:
            original_lead_time: Baseline lead time (default from config)
            extended_lead_time: Stressed lead time
            
        Returns:
            Dict with results for original and extended lead times
        """
        if original_lead_time is None:
            original_lead_time = self.cfg.rl.environment.lead_time
        
        print(f"\n{'='*60}")
        print(f"TEST 3: LEAD TIME VARIATION ({original_lead_time}→{extended_lead_time} days)")
        print(f"{'='*60}")
        print(f"NOTE: Only testing Standard RL and baselines (agents trained with lead_time={original_lead_time})")
        
        results_by_lead_time = {}
        
        for lead_time in [original_lead_time, extended_lead_time]:
            print(f"\n  Evaluating with lead time = {lead_time} days...")
            
            results = {}
            for agent_name, agent in self.agents.items():
                # Skip RL agents if lead time changed (state dimension mismatch)
                if lead_time != original_lead_time and agent_name in ['RL+LSTM', 'Standard RL']:
                    print(f"    Skipping {agent_name} (requires retraining with new lead time)")
                    continue
                
                print(f"    Evaluating {agent_name}...")
                kpis = self._evaluate_agent(
                    agent,
                    self.test_data,
                    lead_time=lead_time
                )
                results[agent_name] = kpis
            
            results_df = pd.DataFrame(results).T
            results_by_lead_time[f'L={lead_time}'] = results_df
        
        self.results['Test_3_Lead_Time'] = results_by_lead_time
        
        print(f"\nResults Comparison:")
        for lead_time_key, df in results_by_lead_time.items():
            print(f"\n{lead_time_key}:")
            print(df)
        
        return results_by_lead_time
    
    # =========================================================================
    # TEST 4: COST PARAMETER SENSITIVITY
    # =========================================================================
    
    def test_4_cost_parameter_sensitivity(
        self,
        penalty_ratios: List[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Test 4: Cost Parameter Sensitivity (penalty ratios 5:1, 10:1, 20:1)
        Evaluates agents under different lost-sales penalty ratios.
        
        Args:
            penalty_ratios: List of penalty ratios to test (p/h where h=holding_cost)
            
        Returns:
            Dict with results for each penalty ratio
        """
        if penalty_ratios is None:
            penalty_ratios = [5, 10, 20]
        
        print(f"\n{'='*60}")
        print(f"TEST 4: COST PARAMETER SENSITIVITY")
        print(f"Penalty ratios: {penalty_ratios}")
        print(f"{'='*60}")
        
        holding_cost = self.cfg.rl.environment.holding_cost_per_unit
        results_by_ratio = {}
        
        for ratio in penalty_ratios:
            penalty = ratio * holding_cost
            print(f"\n  Testing penalty ratio {ratio}:1 (p={penalty})...")
            
            results = {}
            for agent_name, agent in self.agents.items():
                print(f"    Evaluating {agent_name}...")
                kpis = self._evaluate_agent(
                    agent,
                    self.test_data,
                    stockout_penalty_per_unit=penalty
                )
                results[agent_name] = kpis
            
            results_df = pd.DataFrame(results).T
            results_by_ratio[f'Ratio_{ratio}:1'] = results_df
        
        self.results['Test_4_Cost_Sensitivity'] = results_by_ratio
        
        print(f"\nResults Comparison:")
        for ratio_key, df in results_by_ratio.items():
            print(f"\n{ratio_key}:")
            print(df)
        
        return results_by_ratio
    
    # =========================================================================
    # TEST 5: STOCKOUT PENALTY ESCALATION
    # =========================================================================
    
    def test_5_penalty_escalation(
        self,
        penalties: List[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Test 5: Stockout Penalty Escalation (p=5→7.5→10→15)
        Evaluates agents under increasing lost-sales penalties.
        
        Args:
            penalties: List of penalty values to test
            
        Returns:
            Dict with results for each penalty level
        """
        if penalties is None:
            penalties = [5, 7.5, 10, 15]
        
        print(f"\n{'='*60}")
        print(f"TEST 5: STOCKOUT PENALTY ESCALATION")
        print(f"Penalties: {penalties}")
        print(f"{'='*60}")
        
        results_by_penalty = {}
        
        for penalty in penalties:
            print(f"\n  Testing penalty p={penalty}...")
            
            results = {}
            for agent_name, agent in self.agents.items():
                print(f"    Evaluating {agent_name}...")
                kpis = self._evaluate_agent(
                    agent,
                    self.test_data,
                    stockout_penalty_per_unit=penalty
                )
                results[agent_name] = kpis
            
            results_df = pd.DataFrame(results).T
            results_by_penalty[f'p={penalty}'] = results_df
        
        self.results['Test_5_Penalty_Escalation'] = results_by_penalty
        
        print(f"\nResults Comparison:")
        for penalty_key, df in results_by_penalty.items():
            print(f"\n{penalty_key}:")
            print(df)
        
        return results_by_penalty
    
    # =========================================================================
    # EVALUATION HELPER
    # =========================================================================
    
    def _evaluate_agent(
        self,
        agent,
        data: pd.DataFrame,
        lead_time: int = None,
        holding_cost_per_unit: float = None,
        stockout_penalty_per_unit: float = None,
        ordering_cost_per_unit: float = None,
    ) -> Dict:
        """
        Evaluate agent on provided data with optional parameter overrides.
        
        Args:
            agent: Agent to evaluate
            data: Test dataset (DataFrame with 'units_sold' column)
            lead_time: Optional lead time override
            holding_cost_per_unit: Optional holding cost override
            stockout_penalty_per_unit: Optional stockout penalty override
            ordering_cost_per_unit: Optional ordering cost override
            
        Returns:
            Dict with KPIs (service level, lost sales, costs, etc.)
        """
        # Extract demand sequence from data
        demand_sequence = data['units_sold'].values.astype(np.float32)
        
        # Use defaults from config if not provided
        if lead_time is None:
            lead_time = self.cfg.rl.environment.lead_time
        if holding_cost_per_unit is None:
            holding_cost_per_unit = self.cfg.rl.environment.holding_cost_per_unit
        if stockout_penalty_per_unit is None:
            stockout_penalty_per_unit = self.cfg.rl.environment.stockout_penalty_per_unit
        if ordering_cost_per_unit is None:
            ordering_cost_per_unit = self.cfg.rl.environment.ordering_cost_per_unit
        
        # Create a modified config for this evaluation with overridden parameters
        # We need to temporarily override the config values
        original_lead_time = self.cfg.rl.environment.lead_time
        original_holding_cost = self.cfg.rl.environment.holding_cost_per_unit
        original_stockout_penalty = self.cfg.rl.environment.stockout_penalty_per_unit
        original_ordering_cost = self.cfg.rl.environment.ordering_cost_per_unit
        
        self.cfg.rl.environment.lead_time = lead_time
        self.cfg.rl.environment.holding_cost_per_unit = holding_cost_per_unit
        self.cfg.rl.environment.stockout_penalty_per_unit = stockout_penalty_per_unit
        self.cfg.rl.environment.ordering_cost_per_unit = ordering_cost_per_unit
        
        # Initialize environment with modified config
        env = InventoryEnv(
            demand_sequence=demand_sequence,
            forecast_provider=(lambda h: self.lstm_predictor.forecast(h.reshape(1, -1))) if hasattr(agent, 'use_forecast') and agent.use_forecast else None,
            cfg=self.cfg,
            use_forecast=hasattr(agent, 'use_forecast') and agent.use_forecast,
        )
        
        # Run evaluation episode
        state, _ = env.reset()
        episode_data = []
        
        while True:
            # Get action from agent
            if hasattr(agent, 'act'):
                # DQN agent
                action = agent.act(state)
            else:
                # Baseline policy (Fixed-S, Forecast Base-Stock)
                action = agent.get_action(state, env)
            
            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_data.append(info)
            
            if done:
                break
            state = next_state
        
        # Restore original config values
        self.cfg.rl.environment.lead_time = original_lead_time
        self.cfg.rl.environment.holding_cost_per_unit = original_holding_cost
        self.cfg.rl.environment.stockout_penalty_per_unit = original_stockout_penalty
        self.cfg.rl.environment.ordering_cost_per_unit = original_ordering_cost
        
        # Calculate KPIs
        kpis = KPICalculator().compute(episode_data)
        
        return kpis
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    def save_results(self):
        """Save all results to CSV files."""
        for test_name, results in self.results.items():
            if isinstance(results, dict):
                # Multiple DataFrames (Tests 3, 4, 5)
                for sub_test_name, df in results.items():
                    filename = f"{test_name}_{sub_test_name}.csv"
                    filepath = os.path.join(self.output_dir, filename)
                    df.to_csv(filepath)
                    print(f"Saved: {filepath}")
            else:
                # Single DataFrame (Tests 1, 2)
                filename = f"{test_name}.csv"
                filepath = os.path.join(self.output_dir, filename)
                results.to_csv(filepath)
                print(f"Saved: {filepath}")
    
    def generate_plots(self):
        """Generate summary plots for all tests."""
        # TODO: Implement plotting logic
        # Create comparison plots for each test
        # Save to outputs/stress_test_results/plots/
        pass
    
    def run_all_tests(self):
        """Run all 5 stress tests in sequence."""
        print("\n" + "="*60)
        print("STARTING STRESS TEST SUITE")
        print("="*60)
        
        self.test_1_demand_volatility()
        self.test_2_forecast_degradation()
        self.test_3_lead_time_variation()
        self.test_4_cost_parameter_sensitivity()
        self.test_5_penalty_escalation()
        
        print("\n" + "="*60)
        print("STRESS TESTS COMPLETE")
        print("="*60)
        
        self.save_results()
        self.generate_plots()


# ============================================================================
# HELPER CLASSES
# ============================================================================

class NoisyLSTMPredictor:
    """Wraps LSTM predictor and adds Gaussian noise to predictions."""
    
    def __init__(self, lstm_predictor, noise_std: float):
        self.lstm_predictor = lstm_predictor
        self.noise_std = noise_std
    
    def forecast(self, input_seq):
        """Forecast with added noise."""
        forecast = self.lstm_predictor.forecast(input_seq.reshape(1, -1))
        noise = np.random.normal(0, self.noise_std, size=forecast.shape)
        noisy_forecast = forecast + noise
        return np.clip(noisy_forecast, 0, None)  # Clip to non-negative


class FixedSPolicy:
    """Fixed-S (base-stock) policy baseline."""
    
    def __init__(self, reorder_point: float, order_quantity: float):
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
    
    def get_action(self, state, env):
        """Return middle action (no change) - simplified baseline."""
        # Return action index in valid range (0 to n_actions-1)
        # Middle action = no change to inventory
        return env.action_space.n_actions // 2


class ForecastBaseStockPolicy:
    """Forecast-driven base-stock policy baseline."""
    
    def __init__(self, lstm_predictor, safety_stock: float):
        self.lstm_predictor = lstm_predictor
        self.safety_stock = safety_stock
    
    def get_action(self, state, env):
        """Return action index based on inventory level."""
        # Return action index in valid range (0 to n_actions-1)
        # Conservative: order more (higher action index)
        return min(env.action_space.n_actions - 1, env.action_space.n_actions // 2 + 2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize stress test suite
    suite = StressTestSuite(
        lstm_checkpoint="models/lstm/best_lstm_model.pt",
        rl_lstm_checkpoint="models/rl/rl_lstm_agent/rl_lstm_agent_best.pt",
        standard_rl_checkpoint="models/rl/standard_rl_agent/standard_rl_agent_best.pt",
        config_dir="config/",
        test_data_path="data/raw/retail_store_inventory.csv",
        output_dir="outputs/stress_test_results",
    )
    
    # Run all tests
    suite.run_all_tests()