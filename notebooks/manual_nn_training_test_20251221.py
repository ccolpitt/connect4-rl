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
# Print replay buffer
print( "Replay Buffer entry 1:")
print( replay_buffer[0]["state"])
print( "Action: ", replay_buffer[0]["action"])
print( "Reward: ", replay_buffer[0]["reward"])
print( "Next State:" )
print( replay_buffer[0]["next_state"])
print( "Done: ", replay_buffer[0]["done"])

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
# Train with each learning rate
# *****************************************************************
all_results = {}

for lr in learning_rates:
    print(f"\nTesting LR: {lr}")
    
    # Deepcopy initialized layers
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

    optimizer = torch.optim.Adam(
        list(conv1.parameters()) + list(bn1.parameters()) + 
        list(conv2.parameters()) + list(bn2.parameters()) + 
        list(fc1.parameters()) + list(output.parameters()), lr=lr
    )

    # History tracking
    history = {i: [] for i in range(len(replay_buffer))}
    loss_history = []

    for iteration in range(training_iterations):
        optimizer.zero_grad()
        total_loss = 0
        
        # Process the entire batch (the replay buffer)
        for i, entry in enumerate(replay_buffer):
            s = torch.tensor(entry['state'], dtype=torch.float32).unsqueeze(0)
            ns = torch.tensor(entry['next_state'], dtype=torch.float32).unsqueeze(0)
            a = entry['action']
            r = entry['reward']
            d = entry['done']
            
            # Calculate Target Q
            with torch.no_grad():
                if d:
                    target_q = torch.tensor(r, dtype=torch.float32)
                else:
                    # Bellman Equation: r + gamma * max(Q(ns))
                    next_q = forward(ns)
                    target_q = r + gamma * torch.max(next_q)
            
            # Forward pass
            current_q_values = forward(s)
            predicted_q = current_q_values[0, a]
            
            loss = nn.functional.mse_loss(predicted_q, target_q)
            total_loss += loss
            history[i].append(predicted_q.item())
            
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

    all_results[lr] = {'history': history, 'loss': loss_history}

# *****************************************************************
# Visualization
# *****************************************************************
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, lr in enumerate(learning_rates):
    # Plot Loss
    ax1.plot(all_results[lr]['loss'], label=f'LR {lr}')
    
    # Plot Q-values for both moves for the best LR (or first one)
    if lr == learning_rates[0]:
        for move_idx in range(len(replay_buffer)):
            ax2.plot(all_results[lr]['history'][move_idx], 
                    linestyle='--', label=f'Move {move_idx+1} Q (LR {lr})')

ax1.set_title("Total Batch Loss")
ax1.set_yscale('log')
ax1.legend()

ax2.set_title("Q-Value Convergence (Sample LR)")
ax2.axhline(1.0, color='g', alpha=0.3, label="Win Target")
ax2.axhline(-1.0, color='r', alpha=0.3, label="Loss Target")
ax2.legend()

plt.show()