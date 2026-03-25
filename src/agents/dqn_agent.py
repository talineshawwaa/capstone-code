import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.networks import QNetwork, build_qnetwork_from_config
from src.agents.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

class DQNAgent(BaseAgent):
    # Deep Q-Network agent for discrete inventory replenishment
    def __init__(
        self,
        state_dim: int,
        n_actions:  int,
        cfg,
        device:    Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.n_actions  = n_actions
        self.cfg        = cfg

        dqn_cfg = cfg.rl.dqn

        # Device 
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        # Online network: updated at every training step via gradient descent.
        # This is the network we use to SELECT actions.
        self.online_net = build_qnetwork_from_config(cfg, state_dim, n_actions)
        self.online_net = self.online_net.to(self.device)

        # Target network: a delayed copy of the online network.
        # Used to COMPUTE target Q-values in the Bellman equation.
        self.target_net = build_qnetwork_from_config(cfg, state_dim, n_actions)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target network never trains — no gradients needed

        # Optimiser 
        self.optimiser = torch.optim.Adam(
            self.online_net.parameters(),
            lr=dqn_cfg.learning_rate
        )

        #  Loss function 
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer 
        self.replay_buffer = ReplayBuffer(
            capacity=dqn_cfg.replay_buffer_size,
            state_dim=state_dim,
        )

        # Hyperparameters 
        self.gamma                  = dqn_cfg.gamma
        self.batch_size             = dqn_cfg.batch_size
        self.target_update_freq     = dqn_cfg.target_update_frequency
        self.epsilon                = dqn_cfg.epsilon_start
        self.epsilon_min            = dqn_cfg.epsilon_min
        self.epsilon_decay          = dqn_cfg.epsilon_decay

        # Step counter — used to trigger target network updates
        self._step_count   = 0
        self._update_count = 0

        logger.info(
            f"DQNAgent: state_dim={state_dim}, n_actions={n_actions}, "
            f"device={self.device}, gamma={self.gamma}, "
            f"epsilon={self.epsilon}→{self.epsilon_min}"
        )

    def act(self, obs: np.ndarray) -> int:
        # Selects an action using epsilon-greedy policy
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))

        # Exploitation: greedy action from Q-network
        self.online_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values     = self.online_net(state_tensor)
            action       = q_values.argmax(dim=1).item()
        self.online_net.train()

        return int(action)

    def learn(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> Optional[float]:
        # Stores a transition and updates the network if ready
        # Store transition in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        self._step_count += 1

        # Decay epsilon after each step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Don't train until buffer has enough transitions
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample mini-batch and update the online network
        loss = self._update()

        # Periodically copy online network weights to target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logger.debug(f"Target network updated at step {self._step_count}")

        return loss

    def reset(self) -> None:
        # Reset episode-level state. Epsilon and buffer persist across episodes
        pass

    def save(self, path: str) -> None:
        # Saves the online network weights and training state to disk
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "online_net_state_dict": self.online_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimiser_state_dict":  self.optimiser.state_dict(),
                "epsilon":               self.epsilon,
                "step_count":            self._step_count,
                "state_dim":             self.state_dim,
                "n_actions":             self.n_actions,
            },
            path,
        )
        logger.info(f"DQNAgent saved to {path}")

    def load(self, path: str) -> None:
        """
        Loads network weights and training state from a checkpoint.

        Args:
            path: Path to the saved checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.epsilon     = checkpoint["epsilon"]
        self._step_count = checkpoint["step_count"]
        logger.info(
            f"DQNAgent loaded from {path} | "
            f"step={self._step_count}, epsilon={self.epsilon:.4f}"
        )

    # =========================================================================
    # PRIVATE: NETWORK UPDATE
    # =========================================================================

    def _update(self) -> float:
        """
        Samples a mini-batch from the replay buffer and performs one
        gradient update on the online network.

        Returns:
            Loss value for this update step.
        """
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # ── Current Q-values ──────────────────────────────────────────────────
        # Q(s, a; θ) — get Q-values for all actions, then index the ones taken
        # gather(1, actions.unsqueeze(1)) selects the Q-value for the action
        # that was actually taken in each transition.
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # ── Target Q-values (Bellman equation) ───────────────────────────────
        # r + gamma * max_a' Q(s', a'; θ⁻)
        # torch.no_grad() because we don't backpropagate through the target network.
        with torch.no_grad():
            next_q      = self.target_net(next_states).max(dim=1)[0]
            # If done=1 (episode ended), there is no future reward
            target_q    = rewards + self.gamma * next_q * (1.0 - dones)

        # ── Compute loss and update ───────────────────────────────────────────
        loss = self.criterion(current_q, target_q)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._update_count += 1
        return float(loss.item())

