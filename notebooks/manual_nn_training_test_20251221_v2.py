# *****************************************************************
# Learning rate comparison test - same initial weights, different LRs
# Look at last 2 moves.  Last move = +1 reward; 2nd to last = -1 reward
# *****************************************************************

"""
Docstring for notebooks.manual_nn_training_test_20251221_v2

LEARNING: This is a true POC
    1/ When we want to manually override the reward, then we have to put is_done = TRUE as well
    
TO DO:
    1/ L2 Regularization
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import os
from collections import deque
from training_examples_last_2_moves_20251221 import get_training_examples
from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config

# *****************************************************************
# Variables
test_example = 4
training_iterations = 1000 #100
learning_rates = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
dropout_rate = 0.0
gamma = 0.99


# *****************************************************************
# Load test case
examples = get_training_examples()
initial_board = examples[test_example]['board']
actions_list = examples[test_example]['last_moves']
rewards_list = examples[test_example]['rewards']
players_list = examples[test_example]['players']

# *****************************************************************
# Create environment and Replay Buffer
config = Config()
env = ConnectFourEnvironment(config)

# A Replay Buffer stores (state, action, reward, next_state, done)
# These are rewards in the real world, what we observe from the environment
# So, rewards in the replay buffer should be + / - 1 for these training sets
replay_buffer = deque()

# Set state to initial state, based on states loaded from training
print( "Initial State")
print( initial_board )
print( "initial player: ", players_list[0])

env.set_state(initial_board, players_list[0] )

# Populate Replay Buffer
for i in range(len(actions_list)):
    state_obs = env.get_state_from_perspective(players_list[i])
    action = actions_list[i]
    reward = rewards_list[i] # NOTE WE ARE PULLING REWARD FROM DATA, RATHER THAN BELLMAN EQ
    next_board, move_reward, done = env.play_move(action) # Play move
    next_state_obs = env.get_state_from_perspective(-1 * players_list[i])
    replay_buffer.append({
        'state': state_obs,
        'action': action,
        'reward': reward, 
        'next_state': next_state_obs,
        'done': done,
        'move': i,
        'player_who_moved': players_list[i]
        })
    # Done creating replay buffer.

# *****************************************************************
# Print replay buffer --> looks ok, as of 12/21/2025
"""
print( "*"*60)
print( "Move: ", replay_buffer[0]["move"])
print( "Replay Buffer entry 1:")
print( replay_buffer[0]["state"])
print( "Action: ", replay_buffer[0]["action"])
print( "Reward: ", replay_buffer[0]["reward"])
print( "Next State:" )
print( replay_buffer[0]["next_state"])
print( "Done: ", replay_buffer[0]["done"])
print( "Player who moved:", replay_buffer[0]["player_who_moved"] )


print()
print( "Move: ", replay_buffer[1]["move"])
print( "Replay Buffer entry 1:")
print( replay_buffer[1]["state"])
print( "Action: ", replay_buffer[1]["action"])
print( "Reward: ", replay_buffer[1]["reward"])
print( "Next State:" )
print( replay_buffer[1]["next_state"])
print( "Done: ", replay_buffer[1]["done"])
print( "Player who moved:", replay_buffer[1]["player_who_moved"] )
"""

# *****************************************************************
# Create INITIAL network
device = torch.device('cpu')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

conv1_init = nn.Conv2d(2, 32, kernel_size=3, padding=1)
bn1_init = nn.BatchNorm2d(32)
dropout1_init = nn.Dropout2d(p=dropout_rate)
conv2_init = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2_init = nn.BatchNorm2d(64)
dropout2_init = nn.Dropout2d(p=dropout_rate)
fc1_init = nn.Linear(64 * 6 * 7, 128)
dropout3_init = nn.Dropout(p=dropout_rate)
output_init = nn.Linear(128, 7)

# He initialization
for layer in [conv1_init, conv2_init, fc1_init, output_init]:
    if hasattr(layer, 'weight'):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)

# *****************************************************************
# Create Replay Buffer in Pytorch - no sampling - just convert the whole thing.
start_state_tensor_batch = torch.stack([torch.tensor(e['state'], dtype=torch.float32) for e in replay_buffer])
next_state_tensor_batch = torch.stack([torch.tensor(e['next_state'], dtype=torch.float32) for e in replay_buffer])
actions_batch = torch.tensor([e['action'] for e in replay_buffer], dtype=torch.long)
rewards_tensor_batch = torch.tensor([e['reward'] for e in replay_buffer], dtype=torch.float32)
dones_tensor_batch = torch.tensor([float(e['done']) for e in replay_buffer], dtype=torch.float32)

# OVERRIDE dones_batch - second to last should be set to true
# This is crucial, because if we don't do this, the bellman equation, where we ahve
# (1-done) will bring in the network-calculated Q value, which is not what we want, because
# we know actually what the reward is!
dones_tensor_batch[0] = True

"""
# Looks good 20251221
print( "Initial States (pytorch)")
print( start_state_tensor_batch )
print( "actions (pytorch):")
print( actions_batch )
print( "Next States (pytorch):")
print( next_state_tensor_batch )
print( "Rewards (pytorch):")
print( rewards_batch )
print( "Dones (pytorch):")
print( dones_tensor_batch )
"""

# *****************************************************************
# This is the key part - train with 100 episodes for each learning rate.  
# For each episode for a LR, calculate q-values with inference.  Calculate
# the Q-value error for each state
# For each learning rate, and each
# Episode, state:
    # 

# Storage for all results across learning rates and episodes.  There's an entry per learning rate.
results_dict = {}

for lr in learning_rates:
    print(f"\nTraining with LR: {lr}")
    
    # prelearn: Copy originally initiated neural net, so all LR trainings start at the same place
    conv1, bn1, dr1 = copy.deepcopy(conv1_init), copy.deepcopy(bn1_init), copy.deepcopy(dropout1_init)
    conv2, bn2, dr2 = copy.deepcopy(conv2_init), copy.deepcopy(bn2_init), copy.deepcopy(dropout2_init)
    fc1, dr3, output = copy.deepcopy(fc1_init), copy.deepcopy(dropout3_init), copy.deepcopy(output_init)

    # prelearn: Define forward inference - not sure why it's in a for loop
    def forward(x):
        x = torch.relu(bn1(conv1(x)))
        x = dr1(x)
        x = torch.relu(bn2(conv2(x)))
        x = dr2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(fc1(x))
        x = dr3(x)
        return output(x)
    
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
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # prelearn: Storage for this learning rate
    q_values_history = []
    loss_history = []
    grad_history = []
    q_magnitude_history = []

     # Calculate target Q-value.  Target Q value is the same as the reward, if we know what the reward
    # is.  Q val is the discounted reward
    with torch.no_grad():
        # next_q_values = forward(next_state_tensor_batch.unsqueeze(0))
        next_q_values = forward(next_state_tensor_batch)
        next_q_max = next_q_values.max(dim=1)[0]
        target_q = rewards_tensor_batch - (1 - dones_tensor_batch) * gamma * next_q_max

    """
    print( "Learning Loop iteration: ", iteration, ": Bellman Values" )
    print( "Next Q Values after next state tensor:")
    print( next_q_values )
    print( "Max of next Q value estimates: " )
    print( next_q_max )
    print( "Dones: ")
    print( dones_tensor_batch )
    print( "Target Q tensor: ")
    print( target_q)
    """

    # AUDIT DATA INITIALIZATION - for this learning rate
    q_values_2nd_last_move_history = []   # History of what the network calculated - for loss [lr x num_episodes]
    q_values_last_move_history = []  # History of what the network calculated - for win [lr x num_episodes]  
    loss_history = []       # NN loss tracker [lr x num_episodes]
    grad_history = []       # Avg Grad [lr x num_episodes ]
    q_magnitude_history = []# Avg Q Val [lr x num_episodes ]


    # Training loop - Do real learning, log metrics here
    for iteration in range(training_iterations):
        # Forward pass
        q_values = forward( start_state_tensor_batch ) # To Do: Make this dtype=torch.float32
        predicted_qs = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

        """
        # This looks ok
        print( "Raw Q Values: ")
        print( q_values )
        print( "Predicted Specific Move Q Values (loss input 1): ")
        print( predicted_qs )
        print( "Target Q Values (loss input 2):")
        print( target_q )
        print( "Raw scenario Rewards (should be the same as targets): ")
        print( rewards_tensor_batch )
        """

        # Calculate loss between actual Q-value predictions (predicted_qs), and targets (which stay the same
        # over all episodes)
        loss = nn.functional.mse_loss(predicted_qs, target_q )

        # Backward pass --> I think this is where the real learning happens
        optimizer.zero_grad()   # Zero the gradients
        loss.backward()         # Back propagate loss gradients through the NN - ie - calculate gradients

        # Calculate average of abs gradient across all parameters - this should stay constant
        total_grad = 0.0
        total_params = 0
        for param in all_params:
            if param.grad is not None:
                total_grad += torch.sum(torch.abs(param.grad)).item()
                total_params += param.grad.numel()
        avg_grad = total_grad / total_params if total_params > 0 else 0.0

        """
        print( "Total Parameters: ", total_params)
        print( "Total Abs Gradient: ", total_grad)
        print( "Average Gradient: ", avg_grad )
        """

        # Update NN weights -- this is the apex of the learning pyramid
        optimizer.step()
        
        # Calculate average magnitude of all Q-values
        q_magnitude = torch.mean(torch.abs(q_values[0])).item()

        # Store metrics for the current episode
        q_values_2nd_last_move_history.append( predicted_qs[0].item() )
        q_values_last_move_history.append( predicted_qs[1].item() )
        #q_values_history.append(predicted_qs.item())        # Two items per episode
        loss_history.append(loss.item())                    # A scalar
        grad_history.append(avg_grad)                       # Scalar
        q_magnitude_history.append(q_magnitude)             # Scalar

    # END OF EPISODE FOR LOOP
    
    # Store results for this learning rate
    results_dict[lr] = {
        'q_values_2nd_to_last': q_values_2nd_last_move_history,
        'q_values_last': q_values_last_move_history,
        'loss': loss_history,
        'grad_history': grad_history,
        'q_magnitudes': q_magnitude_history,
        
        'final_q_loss': q_values_2nd_last_move_history[-1],
        'final_q_win': q_values_last_move_history[-1],
        'final_nn_loss': loss_history[-1],
        'final_q_magnitude': q_magnitude_history[-1]
    }

    #print(f"  Final Loser Q-value: {q_values_2nd_last_move_history[-1]:.4f} (target: {target_q[0].item():.4f})")
    #print(f"  Final Winner Q-value: {q_values_last_move_history[-1]:.4f} (target: {target_q[1].item():.4f})")
    #print(f"  Final NN loss: {loss_history[-1]:.6f}")
    #print(f"  Distance from Loser Q-target: {abs(q_values_2nd_last_move_history[-1] - target_q[0].item()):.4f}")
    #print(f"  Distance from Winner Q-target: {abs(q_values_last_move_history[-1] - target_q[1].item()):.4f}")



# *****************************************************************
# Visualization
# *****************************************************************
print("\n" + "="*80)
print("CREATING COMPARISON VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

colors = ['blue', 'green', 'red', 'purple', 'orange']

# Plot 1: Q-value convergence for LOSING move
ax = axes[0, 0]
for i, lr in enumerate(learning_rates):
    ax.plot(results_dict[lr]['q_values_2nd_to_last'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.axhline(target_q[0].item(), color='black', linestyle='--', linewidth=2,
           label=f'Target ({target_q[0].item():.2f})')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Q-Value')
ax.set_title('Loss Q-Value Convergence Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 1: Q-value convergence for WINNING move
ax = axes[0, 1]
for i, lr in enumerate(learning_rates):
    ax.plot(results_dict[lr]['q_values_last'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.axhline(target_q[1].item(), color='black', linestyle='--', linewidth=2,
           label=f'Target ({target_q[1].item():.2f})')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Q-Value')
ax.set_title('Win Q-Value Convergence Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Loss over time
ax = axes[1, 0]
for i, lr in enumerate(learning_rates):
    ax.plot(results_dict[lr]['loss'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training Loss Trajectory')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Average gradients
ax = axes[1, 1]
for i, lr in enumerate(learning_rates):
    ax.plot(results_dict[lr]['grad_history'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Average Absolute Gradient')
ax.set_title('Gradient Magnitude Evolution')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Average Q-value magnitude
ax = axes[2, 0]
for i, lr in enumerate(learning_rates):
    ax.plot(results_dict[lr]['q_magnitudes'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Average |Q-value|')
ax.set_title('Avg Abs Q-Value Trend')
ax.legend()
ax.grid(True, alpha=0.3)


plt.show()
