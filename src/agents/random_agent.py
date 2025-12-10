"""
Random Agent - Simplest possible agent for testing.

This agent selects actions uniformly at random from legal moves.
It's useful for:
1. Testing the BaseAgent interface
2. Providing a baseline for comparing other agents
3. Quick sanity checks during development

Performance: ~50% win rate against itself (as expected for random play)
"""

import random
import numpy as np
from typing import List, Optional, Dict
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random.
    
    This is the simplest possible agent - it doesn't learn anything,
    just picks random legal moves. Despite its simplicity, it's very useful:
    
    - **Baseline**: Any learning agent should beat this easily
    - **Testing**: Verifies the agent interface works correctly
    - **Debugging**: Quick way to test game mechanics
    
    Example Usage:
        config = Config()
        env = ConnectFourEnvironment(config)
        agent = RandomAgent(name="Random-Bot")
        
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        action = agent.select_action(state, legal_moves)
        env.play_move(action)
    """
    
    def __init__(self, name: str = "Random-Agent", player_id: int = 1, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            name: Human-readable name for the agent
            player_id: Player identifier (1 or -1)
            seed: Random seed for reproducibility (None for random behavior)
        """
        super().__init__(name=name, player_id=player_id)
        
        # Set random seed if provided (useful for reproducible testing)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_action(self, state: np.ndarray, legal_moves: List[int]) -> int:
        """
        Select a random legal action.
        
        Args:
            state: Current game state (ignored - random agent doesn't use it)
            legal_moves: List of valid column indices
        
        Returns:
            action: Randomly selected column index
        """
        return random.choice(legal_moves)
    
    def observe(self, 
                state: np.ndarray, 
                action: int, 
                reward: Optional[float],
                next_state: np.ndarray, 
                done: bool) -> None:
        """
        Random agent doesn't learn, so this does nothing.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether game ended
        """
        # Random agent doesn't learn from experience
        pass
    
    def train(self) -> Optional[Dict[str, float]]:
        """
        Random agent doesn't train, so this does nothing.
        
        Returns:
            None (no training metrics)
        """
        # Random agent doesn't train
        return None
    
    def save(self, filepath: str) -> None:
        """
        Random agent has no parameters to save.
        
        Args:
            filepath: Path to save (ignored)
        """
        print(f"RandomAgent has no parameters to save.")
    
    def load(self, filepath: str) -> None:
        """
        Random agent has no parameters to load.
        
        Args:
            filepath: Path to load from (ignored)
        """
        print(f"RandomAgent has no parameters to load.")