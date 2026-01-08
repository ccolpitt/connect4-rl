"""
Training examples for supervised learning verification.
Each example shows the winning move, and the second to last (losing) move.  Instead
of the target policy, we show the second to last move and last move.

training_examples_last_2_moves_20251221.py

The purpose of this file is to create a plausible replay buffer, given states
that could occur in the course of self play.  Task one is to construct a sample replay
buffer, and prove that it could occur during the course of actual game play.

Task 2 is to use the sample replay buffer, of maybe 16 or 32 examples, and verify that a
neural net will actually learn the rewards, which in these terminal cases, will be the
same thing as the q-value targets.
"""

import numpy as np

def str_to_board(s):
    """Helper to turn a string layout into a 6x7 numpy array."""
    # Removes whitespace and converts chars to ints
    rows = s.strip().split('\n')
    return np.array([[int(c.replace('.', '0').replace('X', '1').replace('O', '-1')) 
                     for c in row.split()] for row in rows])




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
         [0,  1,  0,  0,  0,  0, -1], # <--- col 5 is winner for player 1; blocks for player 2
         [0, -1,  1, -1,  1, -1, -1],
         [0,  1, -1,  1, -1, -1,  1]]
    )
    last_moves = [1, 5]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Example 1: Player 1 wins diag / at column 5'
    })

    # ****** Training Example 1.1 ******
    board = np.array(
        [[ 0,  0,  0,  0,  0,  1, 0],
         [ 0,  0,  0,  0,  0,  1, 0],
         [ 0,  0,  0,  0,  0,  1, 0],
         [ 1,  0,  0,  0,  0, -1, 0], # same as before, but shifted left
         [-1,  1, -1,  1, -1, -1, 0],
         [ 1, -1,  1, -1, -1,  1, 0]])
    last_moves = [0, 4]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Example 1: Player 1 wins diag / at column 5'
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
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Fail to block, then win'
    })


    # ****** Training Example 3 ******
    board = np.array(
        [[0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0, -1, -1,  0, 0, 0],  # Real attempt to block, then win
         [0, 0,  1,  1,  1, 0, 0]])
    last_moves = [5, 1]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'Attempt at block, then win'
    })

    board = np.array(
        [[0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0,  0,  0,  0, 0, 0],
         [0, 0, -1, -1,  0, 0, 0],
         [0, 1,  1,  1, -1, 0, 0]])
    last_moves = [4,0]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'fail to block, then win'
    })

    # ****** Training Example 4.1 ******
    board = np.array(
        [[0, 0, 0,  0,  0,  0, 0],
         [0, 0, 0,  0,  0,  0, 0],
         [0, 0, 0,  0,  0,  0, 0],
         [0, 0, 0,  0,  0,  0, 0],
         [0, 0, 0, -1, -1,  0, 0],
         [0, 0, 1,  1,  1, -1, 0]])
    last_moves = [5,1]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'next case'
    })

    # ****** Training Example 4.2 ******
    board = np.array(
        [[0, 0, 0, 0,  0,  0, 0],
         [0, 0, 0, 0,  0,  0, 0],
         [0, 0, 0, 0,  0,  0, 0],
         [0, 0, 0, 0,  0,  0, 0],
         [0, 0, 0, 0, -1, -1, 0],
         [0, 0, 0, 1,  1,  1, -1]])
    last_moves = [6,2]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'next case'
    })

    # ****** Training Example 5 ******
    board = np.array(
        [[0, 0, 0,  0,  0, 0, 0],
         [0, 0, 0,  0,  0, 0, 0],
         [0, 0, 0,  0,  0, 0, 0],
         [0, 0, 0,  0,  0, 0, 0],
         [0, 0, 0, -1,  0, 0, 0],
         [0, 1, 1,  1, -1, 0, 0]])
    last_moves = [3,0]
    rewards = [-1.0, 1.0]
    players = np.array([-1, 1])
    examples.append({
        'board': board.copy(),
        'last_moves': last_moves,
        'rewards': rewards,
        'players': players,
        'description': 'next case'
    })

    return examples

# ####################################################################################3
# simulate the self-play game play.  Start by loading an initial state and player.
# Then pretend that eps-greedy made the moves outlined in the example
# Use the environment to assign reward vs. the example.
# Verify that the resulting replay buffer all makes sense.

import sys
import os
from pathlib import Path

#root_dir = Path(__file__).resolve().parent.parent.parent
#if str(root_dir) not in sys.path:
#    sys.path.insert(0, str(root_dir))

# Get the absolute path of the current script
#current_file = Path(__file__).resolve()
# Add project root to path to import from src
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Go up three levels to get to the project root
# .parent is training/ | .parent.parent is src/ | .parent.parent.parent is project_root/
#project_root = current_file.parent.parent.parent
#project_root = current_file.parent.parent.parent

sys.path.append('..')

from src.environment.config import Config
from src.environment.connect4 import ConnectFourEnvironment
from src.utils.dqn_replay_buffer import DQNReplayBuffer

def get_action_mask(legal_moves):
    """Converts [0, 1, 3] into [1, 1, 0, 1, 0, 0, 0]"""
    mask = np.zeros(7, dtype=np.float32)
    mask[legal_moves] = 1.0
    return mask

#config = Config()
#env = ConnectFourEnvironment(config)
#replay_buffer = DQNReplayBuffer(capacity=10000)

def generate_artificial_replay_buffer_for_training():
    config = Config()
    env = ConnectFourEnvironment(config)
    # Larger capacity to handle symmetry
    replay_buffer = DQNReplayBuffer(capacity=1000) 
    examples = get_training_examples()

    for example in examples:
        env.reset()
        env.set_state(example['board'], example['players'][0])
        
        for action in example['last_moves']:
            state = env.get_state()
            next_state, reward, done = env.play_move(action)
            
            if not done:
                next_mask = get_action_mask(env.get_legal_moves())
            else:
                next_mask = np.zeros(7, dtype=np.float32)

            # Use symmetric add: adds 2 entries (Original + Mirrored)
            replay_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
            
            # If this move ended the game, the opponent's previous move (2 entries ago)
            # must be updated to a loss. Since we added 2 entries, we hit -3 and -4.
            if reward != 0:
                replay_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
                replay_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)
                
    return replay_buffer

"""
def generate_artificial_replay_buffer_for_training():
    config = Config()
    env = ConnectFourEnvironment(config)
    replay_buffer = DQNReplayBuffer(capacity=10000)
    examples = get_training_examples()
    env.reset()
    done = False
    #moves = 0
    #eps = config.EPS

    # loop through test examples
    for example in examples:
        # Set initial environment state
        env.set_state( example['board'], example['players'][0] )
        # Artificially select actions from example scenarios
        for action in example['last_moves']:
            # Calculate current state (from curr player perspective)
            state = env.get_state()
            # make move
            next_state, reward, done = env.play_move( action )
            # Get mask for the NEXT player ---
            if not done:
                next_legal_moves = env.get_legal_moves()
                next_mask = get_action_mask(next_legal_moves)
            else:
                next_mask = np.zeros(7, dtype=np.float32)
            # Add to replay buffer
            replay_buffer.add( state, action, reward, next_state, done, next_mask)
            # Update the reward of the second to last move, if not a draw
            if( reward != 0 ):
                replay_buffer.update_penalty(-2, -1, True )
    return replay_buffer
"""

#replay_buffer = generate_artificial_replay_buffer_for_training()
#print( "Length of synthetic replay buffer: ", len( replay_buffer) )

"""
idx = 1
for entry in replay_buffer.buffer:
    print( "#"*20)
    print( "EXAMPLE: ", idx)
    print( "#"*20)
    #print( "State:" )
    #print( entry.state )
    #print( "Action:" )
    #print( entry.action )
    print( "Reward:" )
    print( entry.reward )
    #print( "Next State:" )
    #print( entry.next_state )
    print( "Is Done:" )
    print( entry.done )
    #print( "Next move mask:" )
    #print( entry.next_mask )
    idx += 1
"""
# ####################################################################################3

            




# Baseline self-play function - Pulled from train_dqn_20251221
def play_self_play_game():
    env.reset()
    done = False
    moves = 0
    eps = config.EPS

    # loop through moves
    while not done and moves < 42:
        state = env.get_state()     # This needs to return a tensor
        unique_states_seen.add(hash_state(state)) # TRACKING UNIQUE STATES
        legal_moves = env.get_legal_moves()
        
        # Eps greedy
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(state_tensor, legal_moves, eps)
        # Make the move
        next_state, reward, done = env.play_move( action )
        # Get mask for the NEXT player ---
        if not done:
            next_legal_moves = env.get_legal_moves()
            next_mask = get_action_mask(next_legal_moves)
        else:
            next_mask = np.zeros(7, dtype=np.float32)

        # Add to replay buffer
        replay_buffer.add( state, action, reward, next_state, done, next_mask)
        moves += 1
    
    # Update the reward of the second to last move, if not a draw
    if( reward != 0 ):
        replay_buffer.update_penalty(-2, -1, True )
        #print( "SECOND TO LAST")
        #print( replay_buffer.buffer[-2] )

