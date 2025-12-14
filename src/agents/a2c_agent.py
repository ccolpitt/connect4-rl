"""
A2C Agent - Advantage Actor-Critic for Connect 4

This module implements the A2C (Advantage Actor-Critic) algorithm, a policy gradient
method that uses both a policy network (actor) and a value network (critic).

Key Concepts:
-------------
1. **Actor (Policy)**: Learns which actions to take → π(a|s)
2. **Critic (Value)**: Learns how good states are → V(s)
3. **Advantage**: A(s,a) = Q(s,a) - V(s) = "how much better is this action than average?"
4. **On-Policy**: Learns from current policy (no replay buffer)

A2C Algorithm:
--------------
1. Collect trajectory: (s₀, a₀, r₀), (s₁, a₁, r₁), ..., (sₙ, aₙ, rₙ)
2. Compute returns: Rₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ... (discounted rewards)
3. Compute advantages: Aₜ = Rₜ - V(sₜ)
4. Update actor: Maximize log π(aₜ|sₜ) · Aₜ (policy gradient)
5. Update critic: Minimize (Rₜ - V(sₜ))² (value prediction)
6. Add entropy bonus: Encourage exploration

References:
-----------
- A3C paper: Mnih et al. (2016) "Asynchronous Methods for Deep RL"
- A2C: Synchronous version of A3C (more stable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from collections import deque

from src.agents.base_agent import BaseAgent
from src.networks.actor_critic_network import ActorCriticNetwork


class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic agent for Connect 4.
    
    This agent learns to play Connect 4 by:
    1. Using a policy network (actor) to select actions
    2. Using a value network (critic) to evaluate states
    3. Computing advantages to reduce variance in policy gradients
    4. Training on-policy (no replay buffer)
    
    Attributes:
        network (ActorCriticNetwork): Shared backbone with actor and critic heads
        optimizer (torch.optim.Optimizer): Optimizer for network
        gamma (float): Discount factor for future rewards
        entropy_coef (float): Entropy bonus coefficient (exploration)
        value_coef (float): Critic loss weight
        n_steps (int): Number of steps before update (or episode end)
        trajectory (deque): Current trajectory buffer
    
    Example Usage:
        # Create agent
        agent = A2CAgent(
            name="A2C-Agent",
            player_id=1,
            learning_rate=3e-4,
            gamma=0.99,
            entropy_coef=0.01,
            value_coef=0.5,
            n_steps=5
        )
        
        # Training loop
        state = env.get_state()
        action = agent.select_action(state, env.get_legal_moves())
        next_state, reward, done = env.play_move(action)
        agent.observe(state, action, reward, next_state, done)
        # Agent trains automatically when trajectory is full or episode ends
    """
    
    def __init__(
        self,
        name: str = "A2C-Agent",
        player_id: int = 1,
        # Network architecture
        conv_channels: List[int] = [32, 64],
        fc_dims: List[int] = [128],
        dropout_rate: float = 0.0,
        # Learning parameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        # A2C specific parameters
        entropy_coef: float = 0.01,  # Entropy bonus for exploration
        value_coef: float = 0.5,     # Critic loss weight
        max_grad_norm: float = 1.0,  # Gradient clipping (increased from 0.5)
        max_log_prob: float = 10.0,  # Log probability clipping (prevents explosion)
        n_steps: int = 5,            # Update every N steps (or episode end)
        # Device
        device: Optional[str] = None
    ):
        """
        Initialize A2C agent.
        
        Args:
            name: Agent name for identification
            player_id: Player identifier (1 or -1)
            
            Network Architecture:
            conv_channels: List of conv layer output channels [32, 64]
            fc_dims: List of shared FC layer dimensions [128]
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
            
            Learning Parameters:
            learning_rate: Adam optimizer learning rate (3e-4 typical for A2C)
            gamma: Discount factor for future rewards (0.99 = value future highly)
            
            A2C Specific:
            entropy_coef: Entropy bonus coefficient (0.01 typical)
                         Higher = more exploration
            value_coef: Weight for critic loss (0.5 typical)
                       Balances actor vs critic learning
            max_grad_norm: Gradient clipping threshold (1.0 typical)
                          Prevents exploding gradients (increased from 0.5)
            n_steps: Steps before update (5 typical)
                    Lower = more frequent updates, higher variance
                    Higher = less frequent updates, lower variance
            
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
        
        # Create actor-critic network
        self.network = ActorCriticNetwork(
            input_channels=3,
            num_actions=7,
            conv_channels=conv_channels,
            fc_dims=fc_dims,
            dropout_rate=dropout_rate,
            board_height=6,
            board_width=7
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.max_log_prob = max_log_prob  # Clip log probs to prevent explosion
        self.n_steps = n_steps
        
        # Trajectory buffer (on-policy)
        # Stores: (state, action, reward, next_state, done, log_prob, value)
        self.trajectory = deque(maxlen=n_steps)
        
        # Training statistics
        self.train_steps = 0
        self.episodes_completed = 0
        self.total_actor_loss = 0.0
        self.total_critic_loss = 0.0
        self.total_entropy = 0.0
        self.recent_actor_loss = 0.0
        self.recent_critic_loss = 0.0
        self.recent_entropy = 0.0
        self.recent_advantage = 0.0
        self.recent_grad_norm = 0.0
    
    def select_action(self, state: np.ndarray, legal_moves: List[int]) -> int:
        """
        Select action using the current policy (actor network).
        
        The agent samples from the policy distribution, with illegal moves masked out.
        
        Args:
            state: Current game state, shape (3, 6, 7)
            legal_moves: List of valid column indices
        
        Returns:
            action: Selected column index (0-6)
        """
        self.network.eval()
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities and state value
            action_probs, state_value = self.network(state_tensor)
            action_probs = action_probs.cpu().numpy()[0]
            
            # Mask illegal moves
            masked_probs = np.zeros(7)
            masked_probs[legal_moves] = action_probs[legal_moves]
            
            # Renormalize
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # Fallback: uniform over legal moves
                masked_probs[legal_moves] = 1.0 / len(legal_moves)
            
            # Sample action from policy
            action = np.random.choice(7, p=masked_probs)
            
            # Store for training (will be added to trajectory in observe())
            self._last_action_prob = masked_probs[action]
            self._last_state_value = state_value.cpu().item()
        
        return action
    
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: Optional[float],
        next_state: np.ndarray,
        done: bool,
        player_id: Optional[int] = None
    ) -> None:
        """
        Store experience in trajectory buffer and train when ready.
        
        Args:
            state: State before action, shape (3, 6, 7)
            action: Action taken (column index 0-6)
            reward: Reward received (1.0 win, -1.0 loss, 0.0 draw, None continuing)
            next_state: State after action, shape (3, 6, 7)
            done: Whether game ended
            player_id: ID of player who made this move (for alternating rewards)
        """
        # Convert None reward to 0.0 for continuing games
        if reward is None:
            reward = 0.0
        
        # Compute log probability for this action (clip to avoid log(0))
        log_prob = np.log(np.clip(self._last_action_prob, 1e-10, 1.0))
        
        # Add to trajectory (including player_id for alternating rewards)
        self.trajectory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': self._last_state_value,
            'player_id': player_id if player_id is not None else self.player_id
        })
        
        # Train if trajectory is full or episode ended
        if len(self.trajectory) >= self.n_steps or done:
            self.train()
            self.trajectory.clear()
            
            if done:
                self.episodes_completed += 1
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        Update actor and critic using collected trajectory.
        
        Algorithm:
        1. Compute returns (discounted rewards)
        2. Compute advantages: A = returns - V(s)
        3. Actor loss: -log π(a|s) · A (policy gradient)
        4. Critic loss: (returns - V(s))²
        5. Entropy bonus: -H(π) (encourages exploration)
        6. Total loss = actor_loss + value_coef*critic_loss - entropy_coef*entropy
        
        Returns:
            Dictionary with training metrics
            None if trajectory is empty
        """
        if len(self.trajectory) == 0:
            return None
        
        # Extract trajectory data
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        player_ids = []
        
        for step in self.trajectory:
            states.append(step['state'])
            actions.append(step['action'])
            rewards.append(step['reward'])
            dones.append(step['done'])
            log_probs.append(step['log_prob'])
            values.append(step['value'])
            player_ids.append(step['player_id'])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)
        
        # Compute returns with alternating rewards for two-player self-play
        returns = self._compute_returns_alternating(rewards, dones, player_ids)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Get current policy and values
        self.network.train()
        action_logits, state_values = self.network.get_action_logits_and_value(states_tensor)
        state_values = state_values.squeeze(-1)
        
        # Compute advantages
        advantages = returns_tensor - old_values
        
        # Improved advantage normalization (always normalize to prevent explosion)
        # Add small epsilon to prevent division by zero
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # If std is too small, just center the advantages
            advantages = advantages - adv_mean
        
        # Actor loss: policy gradient with advantage
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # CRITICAL FIX: Clip log probabilities to prevent explosion
        # When entropy collapses, log(small_prob) → -∞, causing huge losses
        # Clipping to [-max_log_prob, 0] keeps losses bounded
        selected_log_probs = torch.clamp(selected_log_probs, min=-self.max_log_prob, max=0.0)
        
        actor_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Critic loss: value prediction error
        critic_loss = F.mse_loss(state_values, returns_tensor)
        
        # Entropy bonus (encourages exploration)
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (and track the norm before clipping)
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.recent_grad_norm = grad_norm_before_clip.item()
        
        self.optimizer.step()
        
        # Update statistics
        self.train_steps += 1
        self.total_actor_loss += actor_loss.item()
        self.total_critic_loss += critic_loss.item()
        self.total_entropy += entropy.item()
        self.recent_actor_loss = actor_loss.item()
        self.recent_critic_loss = critic_loss.item()
        self.recent_entropy = entropy.item()
        self.recent_advantage = advantages.mean().item()
        
        # Return metrics
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': state_values.mean().item(),
            'mean_return': returns_tensor.mean().item(),
            'train_steps': self.train_steps
        }
    
    def _compute_returns_alternating(
        self,
        rewards: List[float],
        dones: List[bool],
        player_ids: List[int]
    ) -> np.ndarray:
        """
        Compute discounted returns with alternating rewards for two-player games.
        
        In self-play, when one player wins, the opponent's moves should get negative returns.
        This implements the key insight: opponent's win = my loss.
        
        Example (Player 2 wins with +1):
        Move 5 (P2 wins): reward=+1 → return=+1.0  (P2's move, positive)
        Move 4 (P1):      reward=0  → return=-0.99 (P1's move, negative - led to loss)
        Move 3 (P2):      reward=0  → return=+0.98 (P2's move, positive)
        Move 2 (P1):      reward=0  → return=-0.97 (P1's move, negative)
        Move 1 (P2):      reward=0  → return=+0.96 (P2's move, positive)
        
        Args:
            rewards: List of rewards (only terminal state has non-zero reward)
            dones: List of done flags
            player_ids: List of player IDs who made each move
        
        Returns:
            np.ndarray: Discounted returns for each step, with sign flipped for opponent
        """
        returns = []
        R = 0.0
        
        # Find the winner (player who got the final non-zero reward)
        winner = None
        for i in reversed(range(len(rewards))):
            if rewards[i] != 0.0:
                winner = player_ids[i]
                break
        
        # Compute returns backwards with alternating signs
        for reward, done, player_id in zip(
            reversed(rewards),
            reversed(dones),
            reversed(player_ids)
        ):
            if done:
                R = 0.0  # Reset at episode boundary
            
            # Standard discounted return
            R = reward + self.gamma * R
            
            # CRITICAL: Flip sign if this move was made by the opponent
            # If winner is Player 1, then Player 2's moves get negative returns
            # If winner is Player 2, then Player 1's moves get negative returns
            if winner is not None and player_id != winner:
                R = -R
            
            returns.insert(0, R)
        
        return np.array(returns, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset agent state for new game."""
        # Clear any partial trajectory
        self.trajectory.clear()
    
    def save(self, filepath: str) -> None:
        """
        Save agent to file.
        
        Args:
            filepath: Path to save agent (e.g., "models/a2c_agent.pth")
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_steps = checkpoint['train_steps']
        self.episodes_completed = checkpoint['episodes_completed']
        self.games_played = checkpoint['games_played']
        self.wins = checkpoint['wins']
        self.losses = checkpoint['losses']
        self.draws = checkpoint['draws']
    
    def set_exploration(self, entropy_coef: float) -> None:
        """
        Set exploration rate (entropy coefficient).
        
        Args:
            entropy_coef: New entropy coefficient (higher = more exploration)
        """
        self.entropy_coef = max(0.0, entropy_coef)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get detailed training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        avg_actor_loss = self.total_actor_loss / max(1, self.train_steps)
        avg_critic_loss = self.total_critic_loss / max(1, self.train_steps)
        avg_entropy = self.total_entropy / max(1, self.train_steps)
        
        return {
            'train_steps': self.train_steps,
            'episodes_completed': self.episodes_completed,
            'avg_actor_loss': avg_actor_loss,
            'avg_critic_loss': avg_critic_loss,
            'avg_entropy': avg_entropy,
            'recent_actor_loss': self.recent_actor_loss,
            'recent_critic_loss': self.recent_critic_loss,
            'recent_entropy': self.recent_entropy,
            'recent_advantage': self.recent_advantage,
            **self.get_stats()
        }
    
    def __str__(self) -> str:
        """String representation with training info."""
        stats = self.get_stats()
        return (f"{self.name} (Player {self.player_id}): "
                f"{stats['wins']}/{stats['games_played']} wins "
                f"({stats['win_rate']:.1%}), "
                f"steps={self.train_steps}")