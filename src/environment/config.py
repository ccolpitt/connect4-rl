"""
Configuration class for Connect 4 RL environment and training.

This module contains the Config class which centralizes all hyperparameters
and settings for the Connect 4 game environment and RL agents.

Device selection
----------------
DEVICE is set automatically by src/utils/device.py.
MPS (Apple Silicon GPU) is used if and only if all safety checks pass
(forward-pass parity, no NaN, finite training step).  Otherwise CPU is used.
Run `python src/utils/device.py` to see which device was selected and why.
"""

import torch

from src.utils.device import get_device, get_device_info


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
        DEVICE (str): Compute device — "mps" if safe, else "cpu"
    """
    # Game dimensions
    ROWS = 6
    COLS = 7
    ACTION_SIZE = COLS

    # Player identifiers
    PLAYER_1 = 1
    PLAYER_2 = -1
    DRAW_VALUE = 0

    # Device: auto-selected by src/utils/device.py
    # Uses MPS if all safety checks pass, otherwise CPU.
    # Call `python src/utils/device.py` to see selection reason.
    DEVICE: str = get_device()

    def __repr__(self):
        """String representation of configuration."""
        info = get_device_info()
        return (
            f"Config(ROWS={self.ROWS}, COLS={self.COLS}, "
            f"DEVICE={self.DEVICE!r} [{info['reason']}])"
        )