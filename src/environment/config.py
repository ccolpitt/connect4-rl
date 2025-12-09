"""
Configuration class for Connect 4 RL environment and training.

This module contains the Config class which centralizes all hyperparameters
and settings for the Connect 4 game environment and RL agents.
"""

import torch


class Config:
    """
    Configuration class for Connect 4 environment and RL training.
    
    Attributes:
        ROWS (int): Number of rows in Connect 4 board (6)
        COLS (int): Number of columns in Connect 4 board (7)
        ACTION_SIZE (int): Number of possible actions (equal to COLS)
        PLAYER_1 (int): Identifier for player 1 (+1)
        PLAYER_2 (int): Identifier for player 2 (-1)
        DRAW_VALUE (int): Value returned for draw games (0)
        DEVICE (torch.device): Computing device (CPU/CUDA/MPS)
    """
    
    # Game dimensions
    ROWS = 6
    COLS = 7
    ACTION_SIZE = COLS
    
    # Player identifiers
    PLAYER_1 = 1
    PLAYER_2 = -1
    DRAW_VALUE = 0
    
    # Device configuration
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    
    def __repr__(self):
        """String representation of configuration."""
        return (
            f"Config(ROWS={self.ROWS}, COLS={self.COLS}, "
            f"DEVICE={self.DEVICE})"
        )