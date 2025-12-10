"""
Base Agent - Abstract Interface for All RL Agents

This module defines the BaseAgent abstract class that serves as a blueprint
for all reinforcement learning agents (DQN, A2C, PPO, AlphaZero).

Key Concepts:
-------------
1. **Abstract Base Class (ABC)**: Forces all agents to implement required methods
2. **Polymorphism**: All agents can be used interchangeably through this interface
3. **Consistency**: Ensures all agents interact with the environment the same way

Every agent MUST implement:
- select_action(): Choose an action given a state
- observe(): Store experience from environment interaction
- train(): Update the agent's policy/value function
- save/load(): Persist trained models

Every agent GETS for free:
- Statistics tracking (wins, losses, draws)
- Player ID management
- Common utility methods
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    This class defines the interface that all agents must implement to work
    with the Connect 4 environment. It provides:
    
    1. **Required Methods** (must be implemented by subclasses):
       - select_action(): How the agent chooses moves
       - observe(): How the agent stores experience
       - train(): How the agent learns from experience
       
    2. **Common Functionality** (inherited by all agents):
       - Game statistics tracking
       - Player identification
       - Model save/load interface
    
    Attributes:
        name (str): Human-readable name for the agent (e.g., "DQN-Agent")
        player_id (int): Which player this agent is (1 or -1)
        games_played (int): Total games played
        wins (int): Number of wins
        losses (int): Number of losses
        draws (int): Number of draws
    
    Example Usage:
        # All agents inherit from BaseAgent
        class DQNAgent(BaseAgent):
            def select_action(self, state, legal_moves):
                # DQN-specific action selection
                pass
            
            def observe(self, state, action, reward, next_state, done):
                # Store in replay buffer
                pass
            
            def train(self):
                # Q-learning update
                pass
    """
    
    def __init__(self, name: str = "Agent", player_id: int = 1):
        """
        Initialize base agent.
        
        Args:
            name: Human-readable name for the agent (e.g., "DQN-Agent", "PPO-Agent")
            player_id: Player identifier (1 or -1 for Connect Four)
                      1 = Player 1 (X), -1 = Player 2 (O)
        """
        self.name = name
        self.player_id = player_id
        
        # Statistics tracking
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    # ============================================================================
    # REQUIRED METHODS - Must be implemented by all subclasses
    # ============================================================================
    
    @abstractmethod
    def select_action(self, state: np.ndarray, legal_moves: List[int]) -> int:
        """
        Select an action given the current state.
        
        This is the core decision-making method. Each agent type implements
        this differently:
        - DQN: Choose action with highest Q-value (with ε-greedy exploration)
        - A2C/PPO: Sample from policy distribution
        - AlphaZero: Use MCTS to select action
        
        Args:
            state: Current game state, shape (3, 6, 7)
                   - Channel 0: Player 1 pieces
                   - Channel 1: Player 2 pieces
                   - Channel 2: Current player indicator
            legal_moves: List of valid column indices [0-6]
        
        Returns:
            action: Column index to play (0-6)
        
        Example:
            state = env.get_state()
            legal_moves = env.get_legal_moves()
            action = agent.select_action(state, legal_moves)
            env.play_move(action)
        """
        pass
    
    @abstractmethod
    def observe(self, 
                state: np.ndarray, 
                action: int, 
                reward: Optional[float],
                next_state: np.ndarray, 
                done: bool) -> None:
        """
        Record the outcome of taking an action.
        
        This method stores experience for later learning. Different agents
        store experience differently:
        - DQN: Add to replay buffer (off-policy)
        - A2C/PPO: Add to trajectory buffer (on-policy)
        - AlphaZero: Store game trajectories for self-play
        
        Args:
            state: State before action, shape (3, 6, 7)
            action: Action taken (column index 0-6)
            reward: Reward received (1.0 for win, -1.0 for loss, 0.0 for draw, None if continuing)
            next_state: State after action, shape (3, 6, 7)
            done: Whether the game ended
        
        Example:
            state = env.get_state()
            action = agent.select_action(state, legal_moves)
            next_state, reward, done = env.play_move(action)
            agent.observe(state, action, reward, next_state, done)
        """
        pass
    
    @abstractmethod
    def train(self) -> Optional[Dict[str, float]]:
        """
        Update the agent's policy/value function using stored experience.
        
        This is where learning happens. Each agent type learns differently:
        - DQN: Sample from replay buffer, minimize TD error
        - A2C: Compute advantages, update policy and value
        - PPO: Multiple epochs with clipped objective
        - AlphaZero: Train on self-play games
        
        Returns:
            Optional dict with training metrics (loss, accuracy, etc.)
            Returns None if not enough data to train yet
        
        Example:
            # After collecting some experience
            if len(agent.memory) >= agent.batch_size:
                metrics = agent.train()
                print(f"Loss: {metrics['loss']}")
        """
        pass
    
    # ============================================================================
    # OPTIONAL METHODS - Can be overridden by subclasses if needed
    # ============================================================================
    
    def reset(self) -> None:
        """
        Reset agent state before starting a new game.
        
        Override this if your agent needs to clear any game-specific state:
        - Clear MCTS tree (AlphaZero)
        - Reset epsilon schedule (DQN)
        - Clear temporary buffers
        
        Default implementation does nothing.
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save agent to file.
        
        Override this to save your agent's neural network weights and
        any other state needed to resume training.
        
        Args:
            filepath: Path to save the agent (e.g., "models/dqn_agent.pth")
        
        Example:
            agent.save("models/my_agent.pth")
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load agent from file.
        
        Override this to load previously saved neural network weights.
        
        Args:
            filepath: Path to load the agent from
        
        Example:
            agent = DQNAgent(config, env)
            agent.load("models/trained_agent.pth")
        """
        pass
    
    def set_exploration(self, exploration_rate: float) -> None:
        """
        Set exploration rate for agents that support it.
        
        Override this for agents with exploration parameters:
        - DQN: Set epsilon for ε-greedy
        - A2C/PPO: Set entropy coefficient
        
        Args:
            exploration_rate: Exploration parameter (0.0 = greedy, 1.0 = random)
        """
        pass
    
    # ============================================================================
    # COMMON FUNCTIONALITY - Available to all agents
    # ============================================================================
    
    def update_game_result(self, result: str) -> None:
        """
        Update win/loss statistics after a game.
        
        Call this at the end of each game to track agent performance.
        
        Args:
            result: One of 'win', 'loss', 'draw'
        
        Example:
            if winner == agent.player_id:
                agent.update_game_result('win')
            elif winner == -agent.player_id:
                agent.update_game_result('loss')
            else:
                agent.update_game_result('draw')
        """
        self.games_played += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        elif result == 'draw':
            self.draws += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics.
        
        Returns:
            Dictionary with games played, wins, losses, draws, win rate
        
        Example:
            stats = agent.get_stats()
            print(f"Win rate: {stats['win_rate']:.1%}")
        """
        win_rate = self.wins / self.games_played if self.games_played > 0 else 0.0
        return {
            'name': self.name,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': win_rate
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        stats = self.get_stats()
        return (f"{self.name} (Player {self.player_id}): "
                f"{stats['wins']}/{stats['games_played']} wins "
                f"({stats['win_rate']:.1%})")
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (f"BaseAgent(name='{self.name}', player_id={self.player_id}, "
                f"games_played={self.games_played})")