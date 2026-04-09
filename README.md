# Bridging the Gap: An Integrated LSTM and Deep Reinforcement Learning Framework for Proactive Inventory Replenishment

---

## Overview

This repository contains the full implementation of a two-stage AI framework that combines **Long Short-Term Memory (LSTM)** demand forecasting with a **Deep Q-Network (DQN)** reinforcement learning agent for proactive inventory replenishment decisions.

The core idea is a **Forecast State Bridge** — an integration mechanism that feeds real-time 7-day LSTM demand forecasts directly into the RL agent's state representation at each decision epoch. This allows the agent to make anticipatory replenishment decisions rather than reacting to inventory shortfalls after the fact.

The framework is evaluated against three baselines across 60 store-product pairs:
- Fixed-(s, Q) reorder-point policy
- Forecast-driven base-stock policy (LSTM only, no RL)
- Standard DQN agent (RL without LSTM forecast)

---

## Architecture

```
Raw Demand Data
      |
      v
[ Preprocessor ]  — MinMax scaling per store-product pair
      |
      v
[ Feature Engineer ]  — cyclical encoding (sin/cos), one-hot encoding,
      |                  rolling demand history
      v
[ LSTM Forecast Model ]
  • Input: 30-step rolling window
  • Output: 7-day ahead demand forecast
  • Architecture: 2-layer LSTM, hidden_size=64, dropout=0.2
      |
      v
[ Forecast State Bridge ]
  • Provides rolling window to LSTM at each RL decision step
  • Inverse-transforms scaled forecast back to real units
  • Zero-padding fallback during warm-up period
      |
      v
[ Inventory Environment (MDP) ]
  • Lost-sales model (no backorders)
  • Lead time: 7 days
  • Reward: -(holding cost + lost-sales penalty + ordering cost)
      |
      v
[ DQN Agent ]
  • Discrete action space: δk ∈ {-200, -175, ..., 0, ..., 175, 200} (17 actions)
  • Applied as delta adjustment over forecast-informed base-stock target
  • 2-layer Q-network, hidden_size=256, ReLU activations
  • Experience replay (buffer=50,000), batch_size=64
  • Huber loss, Adam optimizer, gradient clipping (max_norm=10.0)
  • ε-greedy exploration: ε=1.0 → 0.01 (decay=0.995)
  • Target network updated every 500 steps
```

---

## Repository Structure

```
capstone-code-1/
├── config/
│   ├── base_config.yaml          # Paths, hardware, logging settings
│   ├── lstm_config.yaml          # LSTM model and training hyperparameters
│   ├── rl_config.yaml            # DQN agent, environment, and baseline settings
│   └── config_loader.py          # Unified config loader (dot-access)
│
├── data/
│   └── raw/
│       └── retail_store_inventory.csv   # Source dataset (5 stores × 20 products)
│
├── models/
│   ├── lstm/
│   │   ├── best_lstm_model.pt    # Best LSTM checkpoint (lowest val loss)
│   │   └── scalers/              # Per store-product MinMax scalers (.pkl)
│   └── rl/
│       ├── rl_lstm_agent/        # DQN + LSTM agent checkpoints
│       └── standard_rl_agent/    # Baseline DQN agent checkpoints (no LSTM)
│
├── outputs/
│   ├── logs/                     # Training logs
│   ├── plots/                    # Generated visualizations
│   └── results/                  # KPI CSVs and comparison reports
│
├── scripts/
│   ├── train_lstm.py             # Train the LSTM forecast model
│   ├── train_rl_agent.py         # Train the DQN + LSTM agent
│   ├── train_standard_rl.py      # Train the baseline DQN agent (no forecast)
│   ├── run_scenarios.py          # Evaluate all policies across scenarios
│   ├── evaluate.py               # Build and save comparison report
│   ├── run_visualization.py      # Generate inventory and reward plots
│   └── plot_training_curves.py   # Plot training reward/loss curves
│
├── src/
│   ├── data/
│   │   ├── loader.py             # CSV loading and store-product pair extraction
│   │   ├── preprocessor.py       # MinMax scaling, scaler persistence
│   │   ├── feature_engineering.py # Cyclical, one-hot, rolling features
│   │   └── sequence_builder.py   # LSTM sliding-window sequence builder
│   │
│   ├── forecasting/
│   │   ├── lstm_model.py         # LSTMForecastModel (PyTorch nn.Module)
│   │   ├── trainer.py            # LSTMTrainer with early stopping
│   │   ├── predictor.py          # Inference wrapper
│   │   └── metrics.py            # MAE, RMSE per forecast horizon step
│   │
│   ├── integration/
│   │   ├── forecast_state_bridge.py  # Connects LSTM to RL state at each step
│   │   └── rolling_forecast.py       # Rolling window buffer management
│   │
│   ├── rl_environment/
│   │   ├── inventory_env.py      # Gymnasium-compatible inventory MDP
│   │   ├── reward_function.py    # Cost-based reward (holding + lost-sales + ordering)
│   │   ├── state_space.py        # State vector construction and normalization
│   │   ├── action_space.py       # Discrete delta action space
│   │   └── demand_simulator.py   # Replay-mode demand sequencer
│   │
│   ├── agents/
│   │   ├── dqn_agent.py          # DQN agent (Bellman update, target network)
│   │   ├── networks.py           # Q-network definition
│   │   ├── replay_buffer.py      # Experience replay buffer
│   │   └── base_agent.py         # Abstract agent interface
│   │
│   ├── baselines/
│   │   ├── fixed_s_policy.py         # Fixed (s, Q) reorder-point policy
│   │   ├── forecast_basestock_policy.py  # LSTM-driven base-stock policy
│   │   └── standard_rl_agent.py      # DQN without LSTM forecast
│   │
│   ├── training/
│   │   ├── lstm_pipeline.py      # End-to-end LSTM training orchestration
│   │   ├── rl_pipeline.py        # End-to-end RL training orchestration
│   │   └── callback.py           # Training callbacks (checkpointing, logging)
│   │
│   ├── evaluation/
│   │   ├── scenario_runner.py    # Runs all policies across scenario pairs
│   │   ├── scenario_builder.py   # Splits pairs into Scenario A/B by CV
│   │   ├── backtester.py         # Episode-level backtesting engine
│   │   ├── kpi_calculator.py     # Fill rate, stockout rate, total cost, etc.
│   │   └── comparison_report.py  # Builds master comparison CSV
│   │
│   └── visualizations/
│       ├── comparison_plots.py   # Policy comparison bar/line charts
│       ├── forecast_plots.py     # LSTM forecast vs. actual demand
│       ├── inventory_plots.py    # Inventory level time series
│       └── reward_plots.py       # Training reward curves
│
└── tests/
    ├── test_lstm_model.py        # LSTM unit tests
    ├── test_forecast_bridge.py   # Forecast state bridge tests
    ├── test_inventory_env.py     # RL environment step tests
    ├── test_reward_function.py   # Reward computation tests
    ├── test_preprocessor.py      # Data preprocessing tests
    ├── test_baselines.py         # Baseline policy tests
    └── stress_test.py            # Forecast degradation and lead time stress tests
```

---

## Installation

**Python 3.10+** is required.

```bash
git clone https://github.com/talineshawwaa/capstone-code-1.git
cd capstone-code-1
pip install -r requirements.txt
```

Key dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `matplotlib`.

GPU training is supported and recommended. The framework was trained on IE University's HPC cluster (Haskell node: 512 CPU threads, 503 GB RAM, 2× NVIDIA RTX 6000 Ada GPUs).

---

## Configuration

All hyperparameters are controlled via YAML files in `config/`. No code changes are needed for standard experiments.

### `config/base_config.yaml`
Global paths, hardware flags (`use_gpu: true`), and logging level.

### `config/lstm_config.yaml`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sequence_length` | 30 | Rolling window of historical demand days fed into LSTM |
| `forecast_horizon` | 7 | Days ahead to forecast |
| `hidden_size` | 64 | LSTM hidden state dimension |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Dropout rate |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `batch_size` | 64 | Training batch size |
| `epochs` | 30 | Max training epochs |
| `early_stopping_patience` | 10 | Patience before stopping |

### `config/rl_config.yaml`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `initial_inventory` | 300 | Starting inventory level |
| `max_inventory` | 800 | Inventory capacity |
| `lead_time` | 7 | Order lead time (days) |
| `episode_length` | 30 | Steps per training episode |
| `stockout_penalty_per_unit` | 30.0 | Lost-sales cost per unit |
| `holding_cost_per_unit` | 1.0 | Holding cost per unit per day |
| `ordering_cost_per_unit` | 1.0 | Cost per unit ordered |
| `delta_min / delta_max` | -200 / 200 | Action space bounds |
| `delta_step` | 25 | Action space resolution |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start / min / decay` | 1.0 / 0.01 / 0.995 | ε-greedy schedule |
| `replay_buffer_size` | 50,000 | Experience replay capacity |
| `batch_size` | 64 | DQN training batch size |
| `target_update_frequency` | 500 | Steps between target net syncs |
| `learning_rate` | 0.005 | Adam optimizer LR |
| `total_timesteps` | 200,000 | Total environment steps |

---

## Usage

Run scripts from the project root directory.

### 1. Train the LSTM Forecast Model

```bash
python scripts/train_lstm.py
```
       
Loads `data/raw/retail_store_inventory.csv`, preprocesses and engineers features, builds 30→7 sliding-window sequences, trains the LSTM with early stopping, and saves:
- `models/lstm/best_lstm_model.pt`
- `models/lstm/scalers/` — per store-product MinMax scaler `.pkl` files

### 2. Train the DQN + LSTM Agent

```bash
python scripts/train_rl_agent.py
```

Instantiates the inventory environment with the Forecast State Bridge active (`use_forecast=True`), trains the DQN agent for 200,000 timesteps, and saves checkpoints to `models/rl/rl_lstm_agent/`.

### 3. Train the Standard RL Baseline (no LSTM)

```bash
python scripts/train_standard_rl.py
```

Same as above but with `use_forecast=False`. Saves to `models/rl/standard_rl_agent/`.

### 4. Run Scenario Evaluation

```bash
python scripts/run_scenarios.py
```

Evaluates all policies (DQN+LSTM, Standard DQN, Fixed-(s,Q), Forecast Base-Stock) across 60 store-product pairs, split into:
- **Scenario A** — high demand volatility (high CV)
- **Scenario B** — stable demand (low CV)

Results saved as JSON files in `outputs/results/`.

### 5. Generate Comparison Report

```bash
python scripts/evaluate.py
```

Aggregates scenario results into a master CSV at `outputs/results/comparison_report.csv` with KPIs per policy per scenario.

### 6. Visualizations

```bash
python scripts/run_visualization.py      # Inventory level plots, forecast vs. actual
python scripts/plot_training_curves.py   # Training reward and loss curves
```

Plots saved to `outputs/plots/`.

---

## Dataset

The dataset is a retail store inventory simulation (`data/raw/retail_store_inventory.csv`) spanning **5 stores × 20 products = 100 unique store-product pairs**. Features include:

- Daily demand (units sold)
- Store and product identifiers
- Day-of-week and month (encoded as sin/cos cyclical features)
- Weather conditions, seasonality flags, region (one-hot encoded)
- Rolling demand history (30-day window)

**Preprocessing:** Each store-product pair is independently MinMax scaled to prevent leakage across heterogeneous demand ranges. Scalers are persisted as `.pkl` files and reloaded during evaluation.

---

## Key Design Decisions

**Lost-sales model:** Unmet demand is permanently lost — no backorders. This is the harder and more realistic formulation.

**Delta adjustment policy:** Rather than choosing a raw order quantity, the agent selects a signed adjustment δk over a forecast-informed base-stock target. This reduces the action space complexity and grounds the agent in supply chain logic.

**Forecast State Bridge:** The LSTM is queried at every RL decision step using the current 30-day demand history maintained in a rolling buffer. If the buffer is not yet full (warm-up), a zero-forecast vector is substituted (zero-padding fallback).

**Scenario partitioning:** Store-product pairs are categorized into high-volatility (Scenario A) and stable (Scenario B) groups using the Coefficient of Variation (CV) of demand. This enables differential analysis of where the LSTM forecast adds most value.

**Statistical validation:** Results are validated with paired t-tests and Cohen's d effect size across all 60 store-product pairs.

---

## Running Tests

```bash
python -m pytest tests/
```

Test coverage includes: LSTM forward pass and output shapes, forecast bridge warm-up and inverse-transform behavior, inventory environment step transitions (lost-sales accounting, pipeline shifting), reward function cost decomposition, data preprocessing correctness, and baseline policy action selection.

The `stress_test.py` additionally tests robustness to forecast degradation and lead time variation.

---

## Reproducibility

All random seeds are fixed via `base_config.yaml` (`random_seed: 42`) and applied with `np.random.seed()` at the start of each pipeline. Model checkpoints are saved at every 500 training episodes in addition to the best and final checkpoints.

---

## Acknowledgments

This project was completed as a Master's capstone at IE University. Training was performed on IE University's HPC infrastructure (Haskell node). The retail inventory dataset was sourced from Kaggle.
