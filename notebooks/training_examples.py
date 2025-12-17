"""
Training examples for supervised learning verification.
Each example contains a board position, target policy (probability distribution over actions),
expected reward, current player, and the move that should be taken.
"""

import numpy as np


def get_training_examples():
    """
    Returns a list of training examples for supervised learning.
    
    Each example is a dict with:
    - board: 6x7 numpy array
    - target_policy: list of 7 floats (probability distribution over columns)
    - reward: +1.0 (winning move) or -1.0 (defensive/blocking move)
    - player: 1 or -1 (current player to move)
    - move: int (column index of the correct move)
    - description: str (human-readable description)
    """
    examples = []
    
    # ****** Training Example 1 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, -1, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, -1],
         [0, -1, 1, -1, 1, -1, -1],
         [0, 1, -1, 1, -1, -1, 1]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    reward = 1.0
    player = 1
    move = 5
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 1: Player 1 wins at column 5'
    })
    
    # Flipped version
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 1 (flipped): Player -1 wins at column 5'
    })
    
    # ****** Training Example 1.1 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [-1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, -1, 0],
         [-1, 1, -1, 1, -1, -1, 0],
         [1, -1, 1, -1, -1, 1, 0]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 4
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 1.1: Player 1 wins at column 4'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 1.1 (flipped): Player -1 wins at column 4'
    })
    
    # ****** Training Example 2 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, -1, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, -1],
         [0, -1, 1, -1, 1, -1, -1],
         [0, 1, -1, 1, -1, -1, 1]])
    target_policy = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = -1
    move = 2
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 2: Player -1 wins at column 2'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 2 (flipped): Player 1 wins at column 2'
    })
    
    # ****** Training Example 2.1 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [-1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, -1, 0],
         [-1, 1, -1, 1, -1, -1, 0],
         [1, -1, 1, -1, -1, 1, 0]])
    target_policy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = -1
    move = 1
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 2.1: Player -1 wins at column 1'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 2.1 (flipped): Player 1 wins at column 1'
    })
    
    # ****** Training Example 3 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, -1, -1, 0, 0],
         [0, 0, 1, 1, 1, 0, 0]])
    target_policy = [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
    reward = 1.0
    player = 1
    move = 1  # Can also be 5
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 3: Player 1 can win at column 1 or 5'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 3 (flipped): Player -1 can win at column 1 or 5'
    })
    
    # ****** Training Example 3.1 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, -1, -1, -1, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0]])
    target_policy = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 4  # Can also be 0
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 3.1: Player 1 can win at column 0 or 4'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 3.1 (flipped): Player -1 can win at column 0 or 4'
    })
    
    # ****** Training Example 4 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, -1, 0, 0, 0],
         [0, 1, 1, 1, -1, 0, 0]])
    target_policy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 0
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 4: Player 1 wins at column 0'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 4 (flipped): Player -1 wins at column 0'
    })
    
    # ****** Training Example 4.1 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, -1, 0, 0],
         [0, 0, 1, 1, 1, -1, 0]])
    target_policy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 1
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 4.1: Player 1 wins at column 1'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 4.1 (flipped): Player -1 wins at column 1'
    })
    
    # ****** Training Example 4.2 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, -1, -1, 0],
         [0, 0, 0, 1, 1, 1, -1]])
    target_policy = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 2
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 4.2: Player 1 wins at column 2'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 4.2 (flipped): Player -1 wins at column 2'
    })
    
    # ****** Training Example 5 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0, 0, 0],
         [0, 1, 1, 1, -1, 0, 0]])
    target_policy = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 0
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 5: Player 1 wins at column 0'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 5 (flipped): Player -1 wins at column 0'
    })
    
    # ****** Training Example 6 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0, 0, 0],
         [0, -1, 1, 1, 1, 0, 0]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    reward = 1.0
    player = 1
    move = 5
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 6: Player 1 wins at column 5'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 6 (flipped): Player -1 wins at column 5'
    })
    
    # ****** Training Example 7 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [-1, -1, 1, 1, 1, 0, 0]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    reward = 1.0
    player = 1
    move = 5
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 7: Player 1 wins at column 5'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 7 (flipped): Player -1 wins at column 5'
    })
    
    # ****** Training Example 8 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, -1, -1, 1, -1, 0, 0]])
    target_policy = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 2
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 8: Player 1 wins at column 2 (vertical)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 8 (flipped): Player -1 wins at column 2 (vertical)'
    })
    
    # ****** Training Example 9 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, -1, -1, 1, -1, 0, 0]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 4
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 9: Player 1 wins at column 4 (vertical)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 9 (flipped): Player -1 wins at column 4 (vertical)'
    })
    
    # ****** Training Example 10 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, -1],
         [0, 0, 0, 0, 1, 1, -1],
         [0, -1, 0, 1, -1, -1, 1]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    reward = 1.0
    player = 1
    move = 6
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 10: Player 1 wins at column 6 (diagonal)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 10 (flipped): Player -1 wins at column 6 (diagonal)'
    })
    
    # ****** Training Example 11 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0],
         [0, 0, -1, -1, 1, 0, 0],
         [-1, 0, -1, 1, -1, 1, 0]])
    target_policy = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 2
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 11: Player 1 wins at column 2 (diagonal)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 11 (flipped): Player -1 wins at column 2 (diagonal)'
    })
    
    # ****** Training Example 12 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, -1, 1, 0, 0, 0, 0],
         [-1, -1, 1, 1, -1, -1, 1],
         [1, -1, 1, -1, 1, -1, 1]])
    target_policy = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 2
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 12: Player 1 wins at column 2 (vertical)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 12 (flipped): Player -1 wins at column 2 (vertical)'
    })
    
    # ****** Training Example 13 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, -1, 1, -1, 0, 0],
         [-1, -1, 1, 1, -1, -1, 1],
         [1, -1, 1, -1, 1, -1, 1]])
    target_policy = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 3
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 13: Player 1 wins at column 3 (vertical)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 13 (flipped): Player -1 wins at column 3 (vertical)'
    })
    
    # ****** Training Example 14 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, -1, 1, 0, 0, 0, 0],
         [-1, -1, 1, 1, -1, -1, 1],
         [1, -1, 1, -1, 1, -1, 1]])
    target_policy = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 2  # Can also be 1
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 14: Player 1 can win at column 1 or 2'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 14 (flipped): Player -1 can win at column 1 or 2'
    })
    
    # ****** Training Example 15 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, -1, 0],
         [-1, -1, 1, 1, -1, -1, 1],
         [1, -1, 1, -1, 1, -1, 1]])
    target_policy = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    reward = 1.0
    player = 1
    move = 5
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 15: Player 1 wins at column 5 (diagonal)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 15 (flipped): Player -1 wins at column 5 (diagonal)'
    })
    
    # ****** Training Example 16 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, -1, 0, -1, -1, 1, 0],
         [-1, -1, 1, 1, -1, -1, 1],
         [1, -1, 1, -1, 1, -1, 1]])
    target_policy = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 3
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 16: Player 1 wins at column 3 (diagonal)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 16 (flipped): Player -1 wins at column 3 (diagonal)'
    })
    
    # ****** Training Example 17 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0],
         [0, -1, -1, -1, 1, 0, 0],
         [-1, 1, -1, 1, -1, 0, 0]])
    target_policy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    reward = 1.0
    player = 1
    move = 1
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 17: Player 1 wins at column 1 (diagonal)'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 17 (flipped): Player -1 wins at column 1 (diagonal)'
    })
    
    # ****** Training Example 18 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, -1, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0]])
    target_policy = [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
    reward = -1.0
    player = -1
    move = 5
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 18: Player -1 must block at column 1 or 5'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 18 (flipped): Player 1 must block at column 1 or 5'
    })
    
    # ****** Training Example 19 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0, -1, 0],
         [0, 0, 0, 1, 1, 1, 0]])
    target_policy = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
    reward = -1.0
    player = -1
    move = 6
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 19: Player -1 must block at column 2 or 6'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 19 (flipped): Player 1 must block at column 2 or 6'
    })
    
    # ****** Training Example 20 ******
    board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, -1, 1, 0, 0],
         [-1, 1, -1, -1, 1, 0, 0],
         [1, 1, -1, 1, 1, -1, -1]])
    target_policy = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    reward = -1.0
    player = -1
    move = 4
    examples.append({
        'board': board.copy(),
        'target_policy': target_policy,
        'reward': reward,
        'player': player,
        'move': move,
        'description': 'Example 20: Player -1 must block at column 3 or 4'
    })
    examples.append({
        'board': -1 * board,
        'target_policy': target_policy,
        'reward': reward,
        'player': -1 * player,
        'move': move,
        'description': 'Example 20 (flipped): Player 1 must block at column 3 or 4'
    })
    
    return examples