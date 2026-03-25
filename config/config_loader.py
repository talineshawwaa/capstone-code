import yaml
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ProjectConfig:
    name: str
    version: str
    random_seed: int

@dataclass
class PathsConfig:
    data_raw: str
    data_processed: str
    data_splits: str
    data_synthetic: str
    model_lstm: str
    model_rl: str
    logs: str
    plots: str
    results: str

@dataclass
class HardwareConfig:
    use_gpu: bool
    num_workers: int

@dataclass
class LoggingConfig:
    level: str
    log_to_file: str

@dataclass
class BaseConfig:
    project: ProjectConfig
    paths: PathsConfig
    hardware: HardwareConfig
    logging: LoggingConfig

# LSTM-specific configurations

@dataclass
class LSTMDataConfig:
    sequence_length: int
    forecast_horizon: int
    train_split: str
    val_split: str

@dataclass
class LSTMModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    output_size: int

@dataclass
class LSTMTrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    checkpoint_every_n_epochs: int
    early_stopping_patience: int

@dataclass
class LSTMConfig:
    data: LSTMDataConfig
    model: LSTMModelConfig
    training: LSTMTrainingConfig

# DQN-specific configurations
@dataclass
class EnvironmentConfig:
    initial_inventory: int
    max_inventory: int
    lead_time: int
    episode_length: int
    holding_cost_per_unit: float
    stockout_penalty_per_unit: float
    ordering_cost_per_unit: float
    demand_history_length: int
    forecast_horizon: int
    num_external_numeric: int
    num_external_categorical: int  

@dataclass
class ActionSpaceConfig:
    type: str
    delta_min: int
    delta_max: int
    delta_step: int 

    def get_deltas(self) -> List[int]:
        return list(range(self.delta_min, self.delta_max +1, self.delta_step))
    
    @property
    def n_actions(self) -> int:
        return len(self.get_deltas())

@dataclass
class NetworkConfig:
    hidden_size: int
    num_hidden_layers: int
    activation: str 
    dropout: float

@dataclass
class DQNConfig:
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    replay_buffer_size: int
    batch_size: int
    target_update_frequency: int
    learning_rate: float
    total_timesteps: int
    network: NetworkConfig

@dataclass
class FixedSConfig:
    reorder_point: int
    order_quantity: int

@dataclass
class ForecastBasestockConfig:
    safety_stock: int

@dataclass
class StandardRLConfig:
    uses_forecast: bool

@dataclass
class BaselinesConfig:
    fixed_s: FixedSConfig
    forecast_basestock: ForecastBasestockConfig
    standard_rl: StandardRLConfig

@dataclass
class RLConfig:
    environment: EnvironmentConfig
    action_space: ActionSpaceConfig
    dqn: DQNConfig
    baselines: BaselinesConfig

#Helper function to load YAML config
def _load_yaml(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Config file not found: {filepath}\n"
            f"Make sure you're running from the project root directory."
        )
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def _dict_to_base_config(d: dict) -> BaseConfig:
    """Converts the raw base_config.yaml dict into a BaseConfig dataclass."""
    return BaseConfig(
        project=ProjectConfig(**d["project"]),
        paths=PathsConfig(**d["paths"]),
        hardware=HardwareConfig(**d["hardware"]),
        logging=LoggingConfig(**d["logging"]),
    )

def _dict_to_lstm_config(d: dict) -> LSTMConfig:
    """Converts the raw lstm_config.yaml dict into an LSTMConfig dataclass."""
    return LSTMConfig(
        data=LSTMDataConfig(**d["data"]),
        model=LSTMModelConfig(**d["model"]),
        training=LSTMTrainingConfig(**d["training"]),
    )

def _dict_to_rl_config(d: dict) -> RLConfig:
    """Converts the raw rl_config.yaml dict into an RLConfig dataclass."""
    baselines_d = d["baselines"]
    dqn_d = d["dqn"]
    return RLConfig(
        environment=EnvironmentConfig(**d["environment"]),
        action_space=ActionSpaceConfig(**d["action_space"]),
        dqn=DQNConfig(
            gamma=dqn_d["gamma"],
            epsilon_start=dqn_d["epsilon_start"],
            epsilon_min=dqn_d["epsilon_min"],
            epsilon_decay=dqn_d["epsilon_decay"],
            replay_buffer_size=dqn_d["replay_buffer_size"],
            batch_size=dqn_d["batch_size"],
            target_update_frequency=dqn_d["target_update_frequency"],
            learning_rate=dqn_d["learning_rate"],
            total_timesteps=dqn_d["total_timesteps"],
            network=NetworkConfig(**dqn_d["network"]),
        ),
        baselines=BaselinesConfig(
            fixed_s=FixedSConfig(**baselines_d["fixed_s"]),
            forecast_basestock=ForecastBasestockConfig(
                **baselines_d["forecast_basestock"]
            ),
            standard_rl=StandardRLConfig(**baselines_d["standard_rl"])
        ),
    )

class ConfigLoader:
    """
    Loads all three config YAML files and exposes them as typed attributes.

    Attributes:
        base  (BaseConfig):  Global project settings and paths.
        lstm  (LSTMConfig):  LSTM model and training hyperparameters.
        rl    (RLConfig):    RL environment, agent, and baseline settings.

    Args:
        config_dir: Path to the config/ folder. Defaults to "config/",
                    which works when you run scripts from the project root.
    """

    def __init__(self, config_dir: str = "config/"):
        self.config_dir = config_dir
        self.base: BaseConfig = self._load_base()
        self.lstm: LSTMConfig = self._load_lstm()
        self.rl: RLConfig     = self._load_rl()

    def _load_base(self) -> BaseConfig:
        path = os.path.join(self.config_dir, "base_config.yaml")
        return _dict_to_base_config(_load_yaml(path))

    def _load_lstm(self) -> LSTMConfig:
        path = os.path.join(self.config_dir, "lstm_config.yaml")
        return _dict_to_lstm_config(_load_yaml(path))

    def _load_rl(self) -> RLConfig:
        path = os.path.join(self.config_dir, "rl_config.yaml")
        return _dict_to_rl_config(_load_yaml(path))

    def __repr__(self) -> str:
        """Prints a readable summary — useful at the start of every script."""
        return (
            f"ConfigLoader(\n"
            f"  project={self.base.project.name} v{self.base.project.version}\n"
            f"  seed={self.base.project.random_seed}\n"
            f"  lstm_hidden={self.lstm.model.hidden_size}, "
            f"horizon={self.lstm.data.forecast_horizon}\n"
            f"  rl_lead_time={self.rl.environment.lead_time}, "
            f"episode={self.rl.environment.episode_length} days\n"
            f")"
        )

