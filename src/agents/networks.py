import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    # Multi-layer perceptron that approximates the Q-function
    def __init__(
        self,
        state_dim:         int,
        n_actions:         int,
        hidden_size:       int   = 256,
        num_hidden_layers: int   = 2,
        activation:        str   = "relu",
        dropout:           float = 0.0,
    ):
        super(QNetwork, self).__init__()

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'tanh'.")

        # Build layers dynamically 
        layers = []

        # First layer: state_dim → hidden_size
        layers.append(nn.Linear(state_dim, hidden_size))
        layers.append(act_fn())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        # Additional hidden layers: hidden_size → hidden_size
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        # Output layer: hidden_size → n_actions (one Q-value per action)
        # No activation here — Q-values are unbounded real numbers.
        layers.append(nn.Linear(hidden_size, n_actions))

        # nn.Sequential chains all layers into one callable module
        self.network = nn.Sequential(*layers)

        # Log architecture summary
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"QNetwork: state_dim={state_dim} → "
            f"[{hidden_size}×{num_hidden_layers}] → "
            f"n_actions={n_actions} | "
            f"params={total_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Maps state vectors to Q-values
        return self.network(x)


def build_qnetwork_from_config(cfg, state_dim: int, n_actions: int) -> QNetwork:
    # Builds a QNetwork from the ConfigLoader
    net_cfg = cfg.rl.dqn.network
    return QNetwork(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_size=net_cfg.hidden_size,
        num_hidden_layers=net_cfg.num_hidden_layers,
        activation=net_cfg.activation,
        dropout=net_cfg.dropout
    )

