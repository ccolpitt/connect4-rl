# *****************************************************************
# Full Training - Build it up step by step
# *****************************************************************

"""
Docstring for train_dqn_20251221

This will iteratively build up DQN training.
1: Create environment -- DONE
2: Create manual network based on what worked in manual_nn_training_test_20251221_v2 -- DONE
3: Play a single self-play game.  Use Eps Greedy with constant eps. -- DONE
    If greedy, do inference on state, and sample from the response - softmax
    At the end of the episode, if there was a winner, make sure is_done is true
    Also, override second to last reward to -1, and is_done to True
4: Play an ensemble of games to populate the replay buffer -- DONE
5: Train based on the replay buffer -- DONE
    Train 100 times on static replay buffer, sampling independently; verify loss goes down
    Once replay buffer is ready, train X times per Y games, samping Z samples; verify loss decreases
6: Implement training tracking - DONE
    Avg Abs Q Value prediction
    Avg Abs Q Value of Win/Loss position prediction (only where is_done = True)
    NN Loss by training event
    Win rate vs. random
    Unique States Explored
6: Perspective test - test with three in a row from play 1 and 2 perspective.  Verify that 
    Q values also change a lot.
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
import matplotlib.pyplot as plt

# *****************************************************************
# Constants
# *****************************************************************
num_episodes                = 500   # Number of games to train on
batch_size                  = 16
learning_rate               = 0.00001
training_iterations         = 1     # Training per game
eval_vs_random_game_count   = 50
gamma                       = 0.99
evaluation_frequency        = 10    # Evaluate every 10 episodes
#eval_games: int = 100
#health_check_freq: int = 500
#save_freq: int = 5000
#model_save_path: str = None
#plot_save_path: str = None


# *****************************************************************
# Create environment, Replay Buffer
# *****************************************************************
config = Config()
env = ConnectFourEnvironment(config)
replay_buffer = DQNReplayBuffer(capacity=10000)

#replay_buffer.add( [1,2,3], 0, 0, [2,3,4], False)
#replay_buffer.add( [2,3,4], 0, 1, [3,4,5], False)

"""
print( replay_buffer )
print( "Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-1]))
print( "Second Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-2]))
# Adjust reward of second to last entry
# PLACEHOLDER
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
example_test_case = 5

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

initial_board = examples[example_test_case]['board']
players = examples[example_test_case]['players']
env.set_state( initial_board, players[0] )

print( "Initial board ")
print( initial_board)
print( "Initial Player: ", players[0] )
print( "Initial Board as stored in environment" )
print( env.get_state() )
print( "Test Inference ")
q_values = forward(env.get_state())
print( "Initial Q Values: ", q_values )

print( "Flipped Board")
initial_board = -1 * initial_board
print( initial_board)
env.set_state( initial_board, players[0] )
print( "Flipped Board as stored in environment" )
print( env.get_state() )
print( "Test Inference ")
q_values = forward(env.get_state())
print( "Q Values on flipped state: ", q_values )


# *****************************************************************
# Select Action - Keep it simple to start
# *****************************************************************
def select_action(state, legal_moves, eps) -> int:
    """
    Implements epsilon-greedy action selection with illegal move masking.
    
    Args:
        state (torch.Tensor): 2x6x7 board state.
        legal_moves (list/np.array): List of valid column indices.
        eps (float): Probability of choosing a random action.
        
    Returns:
        int: The chosen action (0-6).
    """
    # 1. EXPLORATION: Randomly choose from legal moves
    if np.random.random() < eps:
        return int(np.random.choice(legal_moves))
    # 2. EXPLOITATION: Choose best action based on Q-values
    # Ensure we don't track gradients during inference
    with torch.no_grad():
        # Add batch dimension: (2, 6, 7) -> (1, 2, 6, 7)
        # Squeeze the output back to (7,)
        q_values = forward(state.unsqueeze(0)).squeeze(0)
        
        # MASKING: We want to ignore moves that are illegal.
        # We create a copy and set illegal moves to a very small value.
        masked_q = q_values.clone()
        
        # Create a set of all possible actions and find the illegal ones
        all_actions = set(range(7))
        illegal_actions = list(all_actions - set(legal_moves))
        
        # Set illegal moves to -infinity (or a very large negative number)
        # so they can never be the argmax.
        masked_q[illegal_actions] = -1e9
        
        # 3. Choose the action with the highest estimated Q-value
        best_action = torch.argmax(masked_q).item()
        
        return int(best_action)
    

# *****************************************************************
# Self Play.  Use eps-greedy
# *****************************************************************
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

# Test out Self Play Game
#play_self_play_game()

# After self play game, print result
#print( "Final State:" )
#print( env.get_state() )
#print( env.get_current_player() )

#print( "Last Replay Buffer Entry: ")
#print( replay_buffer.buffer[-1] )

#print( "Second to Last Replay Buffer Entry: ")
#replay_buffer.buffer[-2]["reward"] = -1 # Update reward
#print( replay_buffer.buffer[-2] )


# *****************************************************************
# Helper functions, for training and metric tracking
# *****************************************************************
# Helper to hash states for Metric 4
def hash_state(state):
    return state.tobytes()

# Helper for Metric 6 (Placeholder - ensure you have a random agent function)
def evaluate_vs_random(num_games=20):
    wins = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = env.get_state()
            legal = env.get_legal_moves()
            # Agent move (Greedy)
            action = select_action(torch.tensor(state, dtype=torch.float32), legal, eps=0.0)
            _, reward, done = env.play_move(action)
            if done and reward == 1:
                wins += 1
                break
            if done: break # Draw or loss
            
            # Random move
            legal = env.get_legal_moves()
            action = np.random.choice(legal)
            _, reward, done = env.play_move(action)
            if done and reward == 1: # Random agent won
                break 
    return wins / num_games

def get_next_q_masked(next_states, next_legal_masks):
    with torch.no_grad():
        # next_q_values shape: (batch_size, 7)
        next_q_values = forward(next_states)
        
        # Apply mask: set illegal moves to -1e9
        # next_legal_masks should be a tensor (batch_size, 7) where 1=legal, 0=illegal
        masked_q = next_q_values.masked_fill(next_legal_masks == 0, -1e9)
        
        return masked_q.max(dim=1)[0]

def get_action_mask(legal_moves):
    """Converts [0, 1, 3] into [1, 1, 0, 1, 0, 0, 0]"""
    mask = np.zeros(7, dtype=np.float32)
    mask[legal_moves] = 1.0
    return mask

# *****************************************************************
# Training Loop, Function: 
# *****************************************************************
# prelearn: Set to training mode
conv1.train()
bn1.train()
dr1.train()
conv2.train()
bn2.train()
dr2.train()
fc1.train()
dr3.train()
output.train()

# prelearn: I think this is to track NN weight and gradient health    
all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
    list(conv2.parameters()) + list(bn2.parameters()) + \
    list(fc1.parameters()) + list(output.parameters())
    
# prelearn: Back propagation optimizer
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

# Initialize learning result metrics
loss_history = []           # 1: Should decrease
q_magnitude_history = []    # 2: Store range of Q values- should trend towards [0-1]
q_magnitude_history_win_loss = []   #3: should trend towards 1
unique_states_history = []  # 4: Should increase
win_rate_vs_rand_hist = []  # 5: Should increase from 50% to 100%
dead_neuron_cnt_hist = []   # 6: Should stay constant
exploding_neuron_cnt_hist = []  # 7: Should stay constant
grad_history = []           # 8: Should stay constant, or decrease
unique_states_seen = set()

def train_dqn_agent():
    for episode in range(1, num_episodes + 1):
        # Play a single game, add moves to replay buffer
        play_self_play_game()
        if( episode % evaluation_frequency == 0 ):
            print( "Episode ", episode, ".  ReplayBuffLen: ", replay_buffer.__len__())

        # Check if enough data to train the network.  Train if enough
        if replay_buffer.is_ready( batch_size ):
            # Sample from replay buffer
            states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(batch_size)
            
            # Convert samples to pytorch tensors
            start_state_tensor_batch = torch.FloatTensor(states).to('cpu')
            actions_batch = torch.LongTensor(actions).to('cpu')
            rewards_tensor_batch = torch.FloatTensor(rewards).to('cpu')
            next_state_tensor_batch = torch.FloatTensor(next_states).to('cpu')
            dones_tensor_batch = torch.FloatTensor(dones).to('cpu')
            next_masks_tensor_batch = torch.FloatTensor(next_masks).to('cpu')
            
            #print( "REPLAY BUFFER SAMPLE")
            #print( "Start State Tensor Batch", start_state_tensor_batch )
            #break

            # Calculate target Q values - note this stays the same for each batch
            # Bellman Equation, with negamax logic
            with torch.no_grad():
                #next_q_values = forward(next_state_tensor_batch)
                #next_q_max = next_q_values.max(dim=1)[0]
                #target_q = rewards_tensor_batch - (1 - dones_tensor_batch) * gamma * next_q_max

                # Use the masked max
                next_q_max = get_next_q_masked(next_state_tensor_batch, next_masks_tensor_batch)
                target_q = rewards_tensor_batch - (1 - dones_tensor_batch) * gamma * next_q_max

            #print( "SAMPLE DONES")
            #print( dones_tensor_batch )
            #print( "SAMPLE TARGET Q VALUES:")
            #print( target_q )
            #print( "SAMPLE SUM OF TARGET Q's")
            #print( torch.sum( target_q))
            #break

            # Train training_iterations times    
            for iteration in range(training_iterations):
                # Forward pass
                q_values = forward( start_state_tensor_batch ) # To Do: Make this dtype=torch.float32
                predicted_qs = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                #print( "SAMPLE PREDICTED Q VALUES")
                #print( predicted_qs )
                #print( "SAMPLE SUM PREDICTED Q's")
                #print( torch.sum( predicted_qs ))
                #break

                # Calculate loss between actual Q-value predictions (predicted_qs), and targets (which stay the same
                # over all episodes)
                # METRIC 1
                loss = nn.functional.mse_loss(predicted_qs, target_q )

                # Backward pass --> I think this is where the real learning happens
                optimizer.zero_grad()   # Zero the gradients
                loss.backward()         # Back propagate loss gradients through the NN - ie - calculate gradients

                # Update NN weights -- THIS IS LEARNING - GREY MATTER UPDATE
                optimizer.step()
                
                # TRAINING METRICS
                # METRIC 2: average magnitude of all Q-values
                q_magnitude = torch.mean(torch.abs(q_values[0])).item()
                # q_magnitude = torch.mean(torch.abs(q_values)).item()

                # METRIC 3: Calculate avg magnitude of Q vals where state is win or loss
                terminal_indices = (dones_tensor_batch == 1).nonzero(as_tuple=True)[0]
                if len(terminal_indices) > 0:
                    terminal_q_avg = torch.mean(torch.abs(predicted_qs[terminal_indices])).item()
                else:
                    terminal_q_avg = 0.0
                
                # METRIC 4: Unique States
                unique_states = len( unique_states_seen )

                # METRIC 5: Win vs. Random Agent
                if episode % evaluation_frequency == 0:
                    win_rate_vs_random_pct = evaluate_vs_random(eval_vs_random_game_count)
                
                # METRIC 6, 7: Dead/Exploding Neurons (using 'output' layer weights as proxy)
                # To Do: Include all neurons
                with torch.no_grad():
                    weights = output.weight.data
                    dead_neuron_cnt = torch.sum(torch.abs(weights) < 0.001).item()
                    exploding_neuron_cnt = torch.sum(torch.abs(weights) > 10.0).item()

                # Calculate average gradient across all parameters
                total_grad = 0.0
                total_params = 0
                for param in all_params:
                    if param.grad is not None:
                        total_grad += torch.sum(torch.abs(param.grad)).item()
                        total_params += param.grad.numel()
                avg_grad = total_grad / total_params if total_params > 0 else 0.0

                # STORE RESULTS
                loss_history.append(loss.item())            # 1: scalar
                q_magnitude_history.append(q_magnitude)     # 2: scalar
                q_magnitude_history_win_loss.append(terminal_q_avg)   #3: scalar
                unique_states_history.append(unique_states)  # 4: Should increase
                if episode % evaluation_frequency == 0:
                    win_rate_vs_rand_hist.append(win_rate_vs_random_pct)     # 5: Should increase from 50% to 100%
                dead_neuron_cnt_hist.append(dead_neuron_cnt)   # 6: Should stay constant
                exploding_neuron_cnt_hist.append(exploding_neuron_cnt)  # 7: Should stay constant
                grad_history.append(avg_grad)                       # 8 Scalar

# Synthetic training with a contrived replay buffer - prove that the network will learn
# win and loss

from notebooks.training_examples_last_2_moves_20251221 import generate_artificial_replay_buffer_for_training

def train_on_synthetic_replay_buffer():
    # load replay buffer
    replay_buffer = generate_artificial_replay_buffer_for_training()
    if replay_buffer.is_ready( 16 ):
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(batch_size)
            
        # Convert samples to pytorch tensors
        start_state_tensor_batch = torch.FloatTensor(states).to('cpu')
        actions_batch = torch.LongTensor(actions).to('cpu')
        rewards_tensor_batch = torch.FloatTensor(rewards).to('cpu')
        next_state_tensor_batch = torch.FloatTensor(next_states).to('cpu')
        dones_tensor_batch = torch.FloatTensor(dones).to('cpu')
        next_masks_tensor_batch = torch.FloatTensor(next_masks).to('cpu')

def train_on_synthetic_replay_buffer():
    for episode in range(1, num_episodes + 1): # fake game - we just pre-load replay buffer
        replay_buffer = generate_artificial_replay_buffer_for_training()

        # Check if enough data to train the network.  Train if enough
        if replay_buffer.is_ready( batch_size ):
            # Sample from replay buffer
            states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(batch_size)
            
            # Convert samples to pytorch tensors
            start_state_tensor_batch = torch.FloatTensor(states).to('cpu')
            actions_batch = torch.LongTensor(actions).to('cpu')
            rewards_tensor_batch = torch.FloatTensor(rewards).to('cpu')
            next_state_tensor_batch = torch.FloatTensor(next_states).to('cpu')
            dones_tensor_batch = torch.FloatTensor(dones).to('cpu')
            next_masks_tensor_batch = torch.FloatTensor(next_masks).to('cpu')

            # Calculate target Q values - note this stays the same for each batch
            # Bellman Equation, with negamax logic
            with torch.no_grad():
                next_q_max = get_next_q_masked(next_state_tensor_batch, next_masks_tensor_batch)
                target_q = rewards_tensor_batch - (1 - dones_tensor_batch) * gamma * next_q_max

            # Train training_iterations times    
            for iteration in range(training_iterations):
                # Forward pass
                q_values = forward( start_state_tensor_batch ) # To Do: Make this dtype=torch.float32
                predicted_qs = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                #print( "SAMPLE PREDICTED Q VALUES")
                #print( predicted_qs )
                #print( "SAMPLE SUM PREDICTED Q's")
                #print( torch.sum( predicted_qs ))
                #break

                # Calculate loss between actual Q-value predictions (predicted_qs), and targets (which stay the same
                # over all episodes)
                # METRIC 1
                loss = nn.functional.mse_loss(predicted_qs, target_q )

                # Backward pass --> I think this is where the real learning happens
                optimizer.zero_grad()   # Zero the gradients
                loss.backward()         # Back propagate loss gradients through the NN - ie - calculate gradients

                # Update NN weights -- THIS IS LEARNING - GREY MATTER UPDATE
                optimizer.step()
                
                # TRAINING METRICS
                # METRIC 2: average magnitude of all Q-values
                q_magnitude = torch.mean(torch.abs(q_values[0])).item()
                # q_magnitude = torch.mean(torch.abs(q_values)).item()

                # METRIC 3: Calculate avg magnitude of Q vals where state is win or loss
                terminal_indices = (dones_tensor_batch == 1).nonzero(as_tuple=True)[0]
                if len(terminal_indices) > 0:
                    terminal_q_avg = torch.mean(torch.abs(predicted_qs[terminal_indices])).item()
                else:
                    terminal_q_avg = 0.0
                
                # METRIC 4: Unique States
                unique_states = len( unique_states_seen )

                # METRIC 5: Win vs. Random Agent
                if episode % evaluation_frequency == 0:
                    win_rate_vs_random_pct = evaluate_vs_random(eval_vs_random_game_count)
                
                # METRIC 6, 7: Dead/Exploding Neurons (using 'output' layer weights as proxy)
                # To Do: Include all neurons
                with torch.no_grad():
                    weights = output.weight.data
                    dead_neuron_cnt = torch.sum(torch.abs(weights) < 0.001).item()
                    exploding_neuron_cnt = torch.sum(torch.abs(weights) > 10.0).item()

                # Calculate average gradient across all parameters
                total_grad = 0.0
                total_params = 0
                for param in all_params:
                    if param.grad is not None:
                        total_grad += torch.sum(torch.abs(param.grad)).item()
                        total_params += param.grad.numel()
                avg_grad = total_grad / total_params if total_params > 0 else 0.0

                # STORE RESULTS
                loss_history.append(loss.item())            # 1: scalar
                q_magnitude_history.append(q_magnitude)     # 2: scalar
                q_magnitude_history_win_loss.append(terminal_q_avg)   #3: scalar
                unique_states_history.append(unique_states)  # 4: Should increase
                if episode % evaluation_frequency == 0:
                    win_rate_vs_rand_hist.append(win_rate_vs_random_pct)     # 5: Should increase from 50% to 100%
                dead_neuron_cnt_hist.append(dead_neuron_cnt)   # 6: Should stay constant
                exploding_neuron_cnt_hist.append(exploding_neuron_cnt)  # 7: Should stay constant
                grad_history.append(avg_grad)                       # 8 Scalar


# *****************************************************************
# Function to show result metrics 
# *****************************************************************
def plot_training_metrics(loss_hist, q_hist, q_terminal_hist, states_hist, 
                          win_rate_hist, dead_hist, exploding_hist, grad_hist, 
                          eval_freq=10):
    """
    Simplified plotting function for DQN training.
    6 subplots consolidating 8 metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Connect-4 DQN Training Dashboard', fontsize=16, fontweight='bold')
    
    # Generate x-axis indices
    episodes = np.arange(len(loss_hist))
    eval_episodes = np.arange(len(win_rate_hist)) * eval_freq

    # --- Plot 1: Loss ---
    ax = axes[0, 0]
    ax.plot(episodes, loss_hist, color='blue', alpha=0.3)
    if len(loss_hist) > 10:
        smoothed = np.convolve(loss_hist, np.ones(10)/10, mode='valid')
        ax.plot(episodes[9:], smoothed, color='blue', label='Smoothed Loss')
    ax.set_title('Training Loss (MSE)')
    ax.set_yscale('log') # Log scale is often better for seeing convergence
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Q-Value Magnitudes (Combined) ---
    ax = axes[0, 1]
    ax.plot(episodes, q_hist, label='Avg All States', alpha=0.8)
    ax.plot(episodes, q_terminal_hist, label='Avg Win/Loss States', alpha=0.8, linestyle='--')
    ax.set_title('Mean |Q| Predictions')
    ax.axhline(y=1.0, color='r', linestyle=':', label='Target (1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Exploration (Unique States) ---
    ax = axes[0, 2]
    ax.plot(episodes, states_hist, color='green', linewidth=2)
    ax.set_title('Unique States Explored')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Win Rate vs Random ---
    ax = axes[1, 0]
    ax.plot(eval_episodes, win_rate_hist, marker='o', color='gold', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    ax.set_ylim(0, 1.1)
    ax.set_title('Win Rate vs Random Agent')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Plot 5: NN Health (Combined Dead/Exploding) ---
    ax = axes[1, 1]
    ax.plot(episodes, dead_hist, label='Dead (<0.001)', color='black')
    ax.plot(episodes, exploding_hist, label='Exploding (>10)', color='red')
    ax.set_title('Neuron Health (Weight Counts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Gradients ---
    ax = axes[1, 2]
    ax.plot(episodes, grad_hist, color='purple')
    ax.set_title('Mean Absolute Gradient')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# *****************************************************************
# Train, show results
# *****************************************************************
#train_dqn_agent()                   # <--- REAL SELF PLAY
train_on_synthetic_replay_buffer()  # <--- SYNTHETIC EXPERIENCE
plot_training_metrics(
    loss_history, q_magnitude_history, q_magnitude_history_win_loss,
    unique_states_history, win_rate_vs_rand_hist, dead_neuron_cnt_hist,
    exploding_neuron_cnt_hist, grad_history, eval_freq=10)

# *****************************************************************
# Save the policy 
# *****************************************************************


# *****************************************************************
# Test the policy 
# *****************************************************************
# Test 1: Three in a row - see if Q-values switch depending on whose move it is
examples = get_training_examples()

initial_board = examples[example_test_case]['board']
players = examples[example_test_case]['players']
env.set_state( initial_board, players[0] )

print( "Initial board ")
print( initial_board)
print( "Initial Player: ", players[0] )
print( "Initial Board as stored in environment" )
print( env.get_state() )
print( "Test Inference ")
q_values = forward(env.get_state())
print( "Initial Q Values: ", q_values )

print( "Switched player")
env.set_state( initial_board, players[1] )

print( "Flipped Board")
initial_board = -1 * initial_board
print( initial_board)
env.set_state( initial_board, players[0] )
print( "Flipped Board as stored in environment" )
print( env.get_state() )
print( "Test Inference ")
q_values = forward(env.get_state())
print( "Q Values on flipped state: ", q_values )