"""
Connect 4 environment module.

This module provides the game environment and configuration for Connect 4 RL training.
"""

from .config import Config
from .connect4 import ConnectFourEnvironment

__all__ = ['Config', 'ConnectFourEnvironment']