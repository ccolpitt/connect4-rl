"""
Training examples for supervised learning verification.
Each example shows the winning move, and the second to last (losing) move.  Instead
of the target policy, we show the second to last move and last move.

training_examples_last_2_moves_20251221.py
"""

import numpy as np


def get_training_examples():
    """
    Returns a list of training examples for supervised learning.
    
    Each example is a dict with:
    - board: 2x6x7 numpy array
    - last_moves: list of 2 ints, showing the second to last move, and last move
    - reward: +1.0 (winning move) or -1.0 (move leading to loss)
    - player: 1 or -1 (current player to move)
    """
    examples = []
    
    # ****** Training Example 1 ******
    board = np.array(
        [[0,  0,  0,  0,  0,  0,  1],
         [0,  0,  0,  0,  0,  0,  1],
         [0,  0,  0,  0,  0,  0,  1],
         [0,  1,  0,  0,  0,  0, -1],
         [0, -1,  1, -1,  1, -1, -1],
         [0,  1, -1,  1, -1, -1,  1]]
    )
    last_moves = [1, 5]
    rewards = [-1.0, 1.0]
    players = [-1, 1]
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Example 1: Player 1 wins diag / at column 5'
    })

    # Flipped version
    examples.append({
        'board': -1 * board,
        'last_moves': last_moves,
        'rewards': rewards,
        'players': -1 * players,
        'description': 'Example 2: Player -1 wins diag / at column 5'
    })

    # ****** Training Example 1.1 ******
    board = np.array(
        [[ 0,  0,  0,  0,  0,  1, 0],
         [ 0,  0,  0,  0,  0,  1, 0],
         [ 0,  0,  0,  0,  0,  1, 0],
         [ 1,  0,  0,  0,  0, -1, 0],
         [-1,  1, -1,  1, -1, -1, 0],
         [ 1, -1,  1, -1, -1,  1, 0]])
    last_moves = [0, 4]
    rewards = [-1.0, 1.0]
    players = [-1, 1]
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Example 1: Player 1 wins diag / at column 5'
    })
    # Flipped version
    examples.append({
        'board': -1 * board,
        'last_moves': last_moves,
        'rewards': rewards,
        'players': -1 * players,
        'description': 'Example 2: Player -1 wins diag / at column 5'
    })
    # ****** Training Example 3 ******
    board = np.array(
        [[0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0, -1, -1, 0, 0],
         [0, 0,  1,  1,  1, 0, 0]])
    last_moves = [2, 1]
    rewards = [-1.0, 1.0]
    players = [-1, 1]
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Example 1: Player 1 wins diag / at column 5'
    })
    # Flipped version
    examples.append({
        'board': -1 * board,
        'last_moves': last_moves,
        'rewards': rewards,
        'players': -1 * players,
        'description': 'Example 2: Player -1 wins diag / at column 5'
    })


    return examples

