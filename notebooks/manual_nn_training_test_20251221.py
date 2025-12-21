# *****************************************************************
# Learning rate comparison test - same initial weights, different LRs
# Look at last 2 moves.  Last move = +1 reward; 2nd to last = -1 reward
# *****************************************************************

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
test_example = 0
training_iterations = 100
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
replay_buffer = deque()

print("Processing move sequence into Replay Buffer...")
current_board = initial_board.copy()
for i in range(len(actions_list)):
    player = players_list[i]
    action = actions_list[i]
    reward = rewards_list[i]
    
    # 1. Set environment to the state BEFORE the move
    env.set_state(current_board, player)
    state_obs = env.get_state_from_perspective(player)
    
    # 2. Play the move
    next_board, move_reward, done = env.play_move(action)
    
    # 3. Get next state from the NEXT player's perspective
    # (In DQN, next_state is used to estimate future value)
    next_state_obs = env.get_state_from_perspective(env.get_current_player())
    
    # Store in buffer
    replay_buffer.append({
        'state': state_obs,
        'action': action,
        'reward': reward, 
        'next_state': next_state_obs,
        'done': done,
        'desc': f"Move {i+1} (Player {player})"
    })
    
    current_board = next_board
    print(f"Added {replay_buffer[-1]['desc']} to buffer. Reward: {reward}, Done: {done}")

# *****************************************************************
# Print replay buffer --> looks ok, as of 12/21/2025
"""
print( "*"*60)
print( "Replay Buffer entry 1:")
print( replay_buffer[0]["state"])
print( "Action: ", replay_buffer[0]["action"])
print( "Reward: ", replay_buffer[0]["reward"])
print( "Next State:" )
print( replay_buffer[0]["next_state"])
print( "Done: ", replay_buffer[0]["done"])
print()
print( "Replay Buffer entry 2:")
print( replay_buffer[1]["state"])
print( "Action: ", replay_buffer[1]["action"])
print( "Reward: ", replay_buffer[1]["reward"])
print( "Next State:" )
print( replay_buffer[1]["next_state"])
print( "Done: ", replay_buffer[1]["done"])
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

# --- STEP 1: PREPARE DATA TENSORS (OUTSIDE BOTH LOOPS) ---
# This converts your list of dictionaries into fixed batch tensors
states_b = torch.stack([torch.tensor(e['state'], dtype=torch.float32) for e in replay_buffer])
next_states_b = torch.stack([torch.tensor(e['next_state'], dtype=torch.float32) for e in replay_buffer])
actions_b = torch.tensor([e['action'] for e in replay_buffer], dtype=torch.long)
rewards_b = torch.tensor([e['reward'] for e in replay_buffer], dtype=torch.float32)
dones_b = torch.tensor([float(e['done']) for e in replay_buffer], dtype=torch.float32)

all_results = {}

# *****************************************************************
# Train with each learning rate
# *****************************************************************
for lr in learning_rates:
    print(f"\nTraining with LR: {lr}")
    
    # Model Setup (using your initial layers)
    conv1, bn1, dr1 = copy.deepcopy(conv1_init), copy.deepcopy(bn1_init), copy.deepcopy(dropout1_init)
    conv2, bn2, dr2 = copy.deepcopy(conv2_init), copy.deepcopy(bn2_init), copy.deepcopy(dropout2_init)
    fc1, dr3, output = copy.deepcopy(fc1_init), copy.deepcopy(dropout3_init), copy.deepcopy(output_init)
    
    def forward(x):
        x = torch.relu(bn1(conv1(x)))
        x = dr1(x)
        x = torch.relu(bn2(conv2(x)))
        x = dr2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(fc1(x))
        x = dr3(x)
        return output(x)

    all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
                 list(conv2.parameters()) + list(bn2.parameters()) + \
                 list(fc1.parameters()) + list(output.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # History storage for metrics
    lr_history = {
        'win_q': [], 'loss_q': [], 'loss': [], 'grads': [], 'avg_abs_q': []
    }

    for iteration in range(training_iterations):
        optimizer.zero_grad()

        # 1. BATCH TARGET CALCULATION (Negamax Bellman)
        # Target = Reward - gamma * max_q(S') * (1 - done)
        with torch.no_grad():
            next_q_batch = forward(next_states_b)
            max_next_q = next_q_batch.max(dim=1)[0]
            targets = rewards_b - (gamma * max_next_q * (1 - dones_b))

        # 2. BATCH FORWARD PASS
        current_q_batch = forward(states_b)
        
        # Extract predicted Q for the specific actions taken
        # (Batch Size, 7) -> (Batch Size, 1) -> (Batch Size)
        predicted_qs = current_q_batch.gather(1, actions_b.unsqueeze(1)).squeeze(1)

        # 3. BATCH LOSS & UPDATE
        loss = nn.functional.mse_loss(predicted_qs, targets)
        loss.backward()
        
        # Calculate Gradient Magnitude (avg absolute grad)
        total_grad = sum(p.grad.abs().sum().item() for p in all_params if p.grad is not None)
        total_params = sum(p.numel() for p in all_params if p.grad is not None)
        
        optimizer.step()

        # 4. RECORD METRICS
        lr_history['win_q'].append(predicted_qs[1].item())  # Last move
        lr_history['loss_q'].append(predicted_qs[0].item()) # 2nd to last
        lr_history['loss'].append(loss.item())
        lr_history['grads'].append(total_grad / total_params)
        # Average Absolute Q value across all 7 actions for both samples
        lr_history['avg_abs_q'].append(torch.mean(torch.abs(current_q_batch)).item())

    all_results[lr] = lr_history

# *****************************************************************
# Visualization
# *****************************************************************
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Connect 4 DQN Training Analysis (Gamma={gamma})", fontsize=16)

colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, lr in enumerate(learning_rates):
    res = all_results[lr]
    
    # Top Left: Winning Move Q
    axes[0,0].plot(res['win_q'], label=f'LR={lr}', color=colors[i])
    axes[0,0].set_title("Winning Move Q-Value (Target -> 1.0)")
    axes[0,0].axhline(1.0, color='black', linestyle='--', alpha=0.5)

    # Top Right: Losing Move Q
    axes[0,1].plot(res['loss_q'], label=f'LR={lr}', color=colors[i])
    axes[0,1].set_title("Losing Move Q-Value (Negamax Target)")
    # Showing common target lines for context
    axes[0,1].axhline(-1.0, color='red', linestyle=':', alpha=0.4, label="-1.0 Reference")
    axes[0,1].axhline(-1.99, color='darkred', linestyle='--', alpha=0.4, label="Bellman -1.99")

    # Bottom Left: Batch Loss
    axes[1,0].plot(res['loss'], label=f'LR={lr}', color=colors[i])
    axes[1,0].set_title("Batch MSE Loss")
    axes[1,0].set_yscale('log')

    # Bottom Right: Gradient Magnitude
    axes[1,1].plot(res['grads'], label=f'LR={lr}', color=colors[i])
    axes[1,1].set_title("Average Gradient Magnitude")
    axes[1,1].set_yscale('log')

for ax in axes.flat:
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Extra Plot: Average Q Magnitude (confidence/stability check)
plt.figure(figsize=(8, 5))
for i, lr in enumerate(learning_rates):
    plt.plot(all_results[lr]['avg_abs_q'], label=f'LR={lr}', color=colors[i])
plt.title("Average Absolute Q-Value (All 7 Actions)")
plt.xlabel("Episode")
plt.ylabel("Mean(|Q|)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()