"""
DQN Agent - Deep Q-Network for Connect 4

This module implements a DQN (Deep Q-Network) agent that learns to play Connect 4
through Q-learning with experience replay and target networks.

Key Concepts:
-------------
1. **Q-Learning**: Learn Q(s,a) = expected future reward for taking action a in state s
2. **Experience Replay**: Store and sample past experiences to break temporal correlation
3. **Target Network**: Use a separate, slowly-updated network for stable learning
4. **ε-Greedy Exploration**: Balance exploration (random) vs exploitation (best known action)

DQN Algorithm:
--------------
1. Select action using ε-greedy policy (explore vs exploit)
2. Execute action, observe reward and next state
3. Store experience (s, a, r, s', done) in replay buffer
4. Sample random batch from replay buffer
5. Compute target: y = r + γ * max_a' Q_target(s', a') if not done, else r
6. Update Q-network to minimize: (Q(s,a) - y)²
7. Periodically update target network: Q_target ← Q

References:
-----------
- Original DQN paper: Mnih et al. (2015) "Human-level control through deep RL"
- Double DQN: van Hasselt et al. (2016) "Deep RL with Double Q-learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Dict, Any
import copy

from agents.base_agent import BaseAgent
from networks.dqn_value_network import DQNValueNetwork
from utils.dqn_replay_buffer import DQNReplayBuffer


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for Connect 4.
    
    This agent learns to play Connect 4 by:
    1. Estimating Q-values Q(s,a) for each state-action pair
    2. Selecting actions using ε-greedy exploration
    3. Learning from experience replay to break temporal correlations
    4. Using a target network for stable learning
    
    Attributes:
        q_network (DQNValueNetwork): Main Q-network for action selection
        target_network (DQNValueNetwork): Target network for computing Q-learning targets
        replay_buffer (DQNReplayBuffer): Experience replay memory
        optimizer (torch.optim.Optimizer): Optimizer for Q-network
        epsilon (float): Current exploration rate for ε-greedy
        gamma (float): Discount factor for future rewards
        batch_size (int): Number of experiences to sample for training
        target_update_freq (int): How often to update target network
        train_steps (int): Total training steps performed
    
    Example Usage:
        # Create agent
        agent = DQNAgent(
            name="DQN-Agent",
            player_id=1,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            target_update_freq=1000
        )
        
        # Training loop
        state = env.get_state()
        action = agent.select_action(state, env.get_legal_moves())
        next_state, reward, done = env.play_move(action)
        agent.observe(state, action, reward, next_state, done)
        
        # Train periodically
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train()
    """
    
    def __init__(
        self,
        name: str = "DQN-Agent",
        player_id: int = 1,
        # Network architecture
        conv_channels: List[int] = [32, 64],
        fc_dims: List[int] = [128],
        dropout_rate: float = 0.0,
        # Learning parameters
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        weight_decay: float = 0.0,  # L2 regularization
        # Exploration parameters
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        # Replay buffer parameters
        buffer_size: int = 100000,
        batch_size: int = 64,
        min_buffer_size: int = 1000,
        # Target network parameters
        target_update_freq: int = 1000,
        # Double DQN
        use_double_dqn: bool = True,
        # Device
        device: Optional[str] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            name: Agent name for identification
            player_id: Player identifier (1 or -1)
            
            Network Architecture:
            conv_channels: List of conv layer output channels [32, 64, 128]
            fc_dims: List of fully connected layer dimensions [256, 128]
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
            
            Learning Parameters:
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards (0.99 = value future highly)
            weight_decay: L2 regularization strength (0.0 = no regularization)
            
            Exploration Parameters:
            epsilon_start: Initial exploration rate (1.0 = fully random)
            epsilon_end: Minimum exploration rate (0.01 = 1% random)
            epsilon_decay: Decay rate per episode (0.995 = slow decay)
            
            Replay Buffer Parameters:
            buffer_size: Maximum experiences to store (100000 = 100k transitions)
            batch_size: Experiences per training step (64 = good default)
            min_buffer_size: Minimum experiences before training (1000 = warmup)
            
            Target Network Parameters:
            target_update_freq: Steps between target network updates (1000 = stable)
            
            Algorithm Variants:
            use_double_dqn: Use Double DQN to reduce overestimation bias
            
            Device:
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        super().__init__(name=name, player_id=player_id)
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Create Q-network and target network
        self.q_network = DQNValueNetwork(
            input_channels=3,
            num_actions=7,
            conv_channels=conv_channels,
            fc_dims=fc_dims,
            dropout_rate=dropout_rate,
            board_height=6,
            board_width=7
        ).to(self.device)
        
        # Target network is a copy of Q-network (frozen initially)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()  # Always in eval mode
        
        # Optimizer with optional L2 regularization
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = DQNReplayBuffer(capacity=buffer_size)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training statistics
        self.train_steps = 0
        self.total_loss = 0.0
        self.episodes_completed = 0
        self.recent_loss = 0.0
        self.recent_q = 0.0
    
    def select_action(self, state: np.ndarray, legal_moves: List[int]) -> int:
        """
        Select action using ε-greedy policy.
        
        With probability ε: choose random legal action (exploration)
        With probability 1-ε: choose action with highest Q-value (exploitation)
        
        Args:
            state: Current game state, shape (3, 6, 7)
            legal_moves: List of valid column indices
        
        Returns:
            action: Selected column index (0-6)
        """
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.choice(legal_moves)
        
        # Exploitation: best action according to Q-network
        return self._get_best_action(state, legal_moves)
    
    def _get_best_action(self, state: np.ndarray, legal_moves: List[int]) -> int:
        """
        Get action with highest Q-value among legal moves.
        
        Args:
            state: Current game state, shape (3, 6, 7)
            legal_moves: List of valid column indices
        
        Returns:
            action: Column with highest Q-value
        """
        self.q_network.eval()
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get Q-values for all actions
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Mask illegal moves with very negative value
            masked_q_values = np.full(7, -np.inf)
            masked_q_values[legal_moves] = q_values[legal_moves]
            
            # Return action with highest Q-value
            return int(np.argmax(masked_q_values))
    
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: Optional[float],
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: State before action, shape (3, 6, 7)
            action: Action taken (column index 0-6)
            reward: Reward received (1.0 win, -1.0 loss, 0.0 draw, None continuing)
            next_state: State after action, shape (3, 6, 7)
            done: Whether game ended
        """
        # Convert None reward to 0.0 for continuing games
        if reward is None:
            reward = 0.0
        
        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step using experience replay.
        
        Algorithm:
        1. Sample batch from replay buffer
        2. Compute Q-learning targets: y = r + γ * max_a' Q_target(s', a')
        3. Compute loss: MSE(Q(s,a), y)
        4. Backpropagate and update Q-network
        5. Update target network periodically
        
        Returns:
            Dictionary with training metrics (loss, q_values, etc.)
            None if not enough data in buffer
        """
        # Check if we have enough data
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        self.q_network.train()
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use Q-network to select action, target network to evaluate
                next_q_values_online = self.q_network(next_states)
                next_actions = next_q_values_online.argmax(dim=1)
                next_q_values_target = self.target_network(next_states)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            # Compute targets: y = r + γ * max_a' Q_target(s', a') if not done, else r
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Track statistics
        self.total_loss += loss.item()
        self.recent_loss = loss.item()
        self.recent_q = current_q_values.mean().item()
        
        # Return metrics
        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item(),
            'mean_target_q': target_q_values.mean().item(),
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'train_steps': self.train_steps
        }
    
    def decay_epsilon(self) -> None:
        """
        Decay exploration rate after each episode.
        
        Call this at the end of each game to gradually reduce exploration.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_completed += 1
    
    def set_exploration(self, exploration_rate: float) -> None:
        """
        Set exploration rate manually.
        
        Args:
            exploration_rate: New epsilon value (0.0 = greedy, 1.0 = random)
        """
        self.epsilon = max(0.0, min(1.0, exploration_rate))
    
    def reset(self) -> None:
        """Reset agent state for new game (nothing to reset for DQN)."""
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save agent to file.
        
        Args:
            filepath: Path to save agent (e.g., "models/dqn_agent.pth")
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episodes_completed': self.episodes_completed,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load agent from file.
        
        Args:
            filepath: Path to load agent from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episodes_completed = checkpoint['episodes_completed']
        self.games_played = checkpoint['games_played']
        self.wins = checkpoint['wins']
        self.losses = checkpoint['losses']
        self.draws = checkpoint['draws']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get detailed training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        avg_loss = self.total_loss / max(1, self.train_steps)
        return {
            'train_steps': self.train_steps,
            'episodes_completed': self.episodes_completed,
            'epsilon': self.epsilon,
            'avg_loss': avg_loss,
            'buffer_size': len(self.replay_buffer),
            'buffer_capacity': self.replay_buffer.capacity,
            **self.get_stats()
        }
    
    def __str__(self) -> str:
        """String representation with training info."""
        stats = self.get_stats()
        return (f"{self.name} (Player {self.player_id}): "
                f"{stats['wins']}/{stats['games_played']} wins "
                f"({stats['win_rate']:.1%}), "
                f"ε={self.epsilon:.3f}, "
                f"steps={self.train_steps}")