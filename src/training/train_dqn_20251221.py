# *****************************************************************
# Full Training - Build it up step by step
# *****************************************************************


"""
Docstring for train_dqn_20251221

This will iteratively build up DQN training.
1: Create environment
2: Create manual network based on manual test
3: Play a single self-play game.  Use Eps Greedy with constant eps.
    If greedy, do inference on state, and sample from the response - softmax
    At the end of the episode, if there was a winner, make sure is_done is true
    Also, override second to last reward to -1, and is_done to True
"""


import sys
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from src.environment import ConnectFourEnvironment, Config
from src.utils import DQNReplayBuffer

# *****************************************************************
# Constants

# *****************************************************************
# Create environment, Replay Buffer
# *****************************************************************
config = Config()
env = ConnectFourEnvironment(config)
replay_buffer = DQNReplayBuffer(capacity=10000)

replay_buffer.add( [1,2,3], 0, 0, [2,3,4], False)
replay_buffer.add( [2,3,4], 0, 0, [3,4,5], False)

"""
print( replay_buffer )
print( "Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-1]))
print( "Second Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-2]))
"""

# *****************************************************************
# Create Network
# *****************************************************************
device = torch.device('cpu')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(32)
dr1 = nn.Dropout2d(p=config.DROPOUT_RATE)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)
dr2 = nn.Dropout2d(p=config.DROPOUT_RATE)
fc1 = nn.Linear(64 * 6 * 7, 128)
dr3 = nn.Dropout(p=config.DROPOUT_RATE)
output = nn.Linear(128, 7)

# He initialization
for layer in [conv1, conv2, fc1, output]:
    if hasattr(layer, 'weight'):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)

def forward(x):
    # Ensure input is a Float Tensor and has a batch dimension
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 3:
        x = x.unsqueeze(0) # Change (2,6,7) to (1,2,6,7)
        
    x = torch.relu(bn1(conv1(x)))
    x = dr1(x)
    x = torch.relu(bn2(conv2(x)))
    x = dr2(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(fc1(x))
    x = dr3(x)
    return output(x)

# *****************************************************************
# Test inference works with the function defined.  Pull examples from the notebook dir.
# *****************************************************************

# Get the absolute path of the current script
current_file = Path(__file__).resolve()
# Add project root to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Go up three levels to get to the project root
# .parent is training/ | .parent.parent is src/ | .parent.parent.parent is project_root/
project_root = current_file.parent.parent.parent
# import training examples
from notebooks.training_examples_last_2_moves_20251221 import get_training_examples

examples = get_training_examples()
initial_board = examples[0]['board']
players = examples[0]['players']
env.set_state( initial_board, players[0] )

q_values = forward(env.get_state())

"""
print( env.get_state() )
print( q_values )
"""

# *****************************************************************
# Self Play.  Use eps-greedy

def play_self_play_game():
    env.reset()
    done = False
    moves = 0

    # Track previous player's state and action for loser experience
    prev_state = None
    prev_action = None

    # loop through moves
    while not done and moves < 42:
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        print( state )
        print( legal_moves )
        break