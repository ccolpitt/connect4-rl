"""
Training hyperparameters for DQN Connect 4 agent.

All training-related configuration lives here, separate from the
environment config (src/environment/config.py) which only holds
game constants (board size, player IDs, etc.).
"""

import torch
from src.utils.device import get_device


class TrainingConfig:
    """
    Hyperparameters for DQN training.
    """

    # Device
    DEVICE: str = get_device()

    # Neural Net
    DROPOUT_RATE = 0.00

    # Training Loop
    NUM_EPISODES = 500
    TRAIN_N_TIMES_PER_GAME = 4
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 100

    # Epsilon Schedule
    EPS_START = 0.5
    EPS_END = 0.2
    EPS_DECAY = 0.9999

    # Replay Buffer
    BATCH_SIZE = 128
    TERMINAL_RATE = 0.3
