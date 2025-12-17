# *****************************************************************
# Import libraries
import sys
sys.path.append('..')

import numpy as np
from training_examples import get_training_examples
from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config

# *****************************************************************
# Variables
test_example = 1

# *****************************************************************
# Load all test case
examples = get_training_examples()
state = examples[test_example]['board']
target_policy = examples[test_example]['target_policy']
reward = examples[test_example]['reward']
player = examples[test_example]['player']

action = np.random.choice(7, p=target_policy)

# *****************************************************************
# Create environment
config = Config()
env = ConnectFourEnvironment(config)

# Print Start
env.set_state( state, player )
print( "Starting State:")
env.render()
print( "Start player: ", env.get_current_player() )

# *****************************************************************
# Generate Replay Buffer Entries
start_state = env.get_state_from_perspective( env.get_current_player() )
end_state, reward, done = env.play_move( action )

'''
print()
print( "start state:")
print( start_state )
print( "end state: ")
print( end_state )

print( "End player: ", env.get_current_player() )
'''

# *****************************************************************
# Create Policy NN, initialize weights
import torch
import torch.nn as nn
import os

# Use CPU, to avoid MPS math errors
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
device = torch.device('cpu')  # Force CPU device
torch.set_default_device('cpu')
print(f"Using device: {device}")

# Create network layers manually (default: conv=[32,64], fc=[128])
conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(32)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)
fc1 = nn.Linear(64 * 6 * 7, 128)  # 64 channels * 6 rows * 7 cols = 2688
output = nn.Linear(128, 7)

# Initialize weights (He initialization for ReLU)
nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(conv1.bias, 0)
nn.init.constant_(bn1.weight, 1)
nn.init.constant_(bn1.bias, 0)
nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(conv2.bias, 0)
nn.init.constant_(bn2.weight, 1)
nn.init.constant_(bn2.bias, 0)
nn.init.kaiming_normal_(fc1.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(fc1.bias, 0)
nn.init.kaiming_normal_(output.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(output.bias, 0)

# Forward pass function
def forward(x):
    x = torch.relu(bn1(conv1(x)))
    x = torch.relu(bn2(conv2(x)))
    x = x.view(x.size(0), -1)  # Flatten
    x = torch.relu(fc1(x))
    return output(x)

# *****************************************************************
# Test inference on the NN
#start_state_tensor = torch.tensor(start_state, dtype=torch.float32)

# Add batch dimension if needed (network expects shape: [batch, channels, height, width])
#if start_state_tensor.dim() == 3:  # If shape is (2, 6, 7) -- needs batch
#    start_state_tensor = start_state_tensor.unsqueeze(0)  # Make it (1, 2, 6, 7)



# *****************************************************************
# Test one training loop.
# Calculate Q value, then calculate target, then loss.  Backpropagate.
# Recalculate loss, and then calc Q
# start_state, action, reward end_state, done

start_state_tensor = torch.tensor(start_state, dtype=torch.float32).unsqueeze(0)
action_tensor = torch.tensor(action, dtype=torch.long)  # Must be long for indexing
reward_tensor = torch.tensor(reward, dtype=torch.float32)
end_state_tensor = torch.tensor(end_state, dtype=torch.float32)
done_tensor = torch.tensor(float(done), dtype=torch.float32)  # Convert bool to float

# Hyperparameters
gamma = 0.99
learning_rate = 0.0001

# Create optimizer
all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
             list(conv2.parameters()) + list(bn2.parameters()) + \
             list(fc1.parameters()) + list(output.parameters())
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

print("\n" + "="*60)
print("BEFORE TRAINING")
print("="*60)

# 1. Calculate initial Q-values
q_values = forward(start_state_tensor)  # Shape: (1, 7)
current_q = q_values[0, action].item()  # Q-value for the action taken
print(f"Q-values: {q_values[0].detach().numpy()}")
print(f"Q-value for action {action}: {current_q:.4f}")

# 2. Calculate target Q-value (negamax formula)
with torch.no_grad():
    next_q_values = forward(end_state_tensor.unsqueeze(0))  # Add batch dim
    next_q_max = next_q_values.max(dim=1)[0]  # Max Q-value in next state
    target_q = reward_tensor - (1 - done_tensor) * gamma * next_q_max
    target_q = target_q.item()

print(f"Target Q-value: {target_q:.4f}")
print(f"Reward: {reward_tensor.item():.4f}, Done: {done_tensor.item()}")

# 3. Calculate initial loss
predicted_q = q_values[0, action]
loss_before = nn.functional.mse_loss(predicted_q, torch.tensor(target_q, dtype=torch.float32))
print(f"Loss (before): {loss_before.item():.6f}")

print("\n" + "="*60)
print("TRAINING (1 step)")
print("="*60)

# 4. Backpropagation and weight update
optimizer.zero_grad()
loss = nn.functional.mse_loss(predicted_q, torch.tensor(target_q, dtype=torch.float32))
loss.backward()
optimizer.step()

print(f"Gradients computed and weights updated")

print("\n" + "="*60)
print("AFTER TRAINING")
print("="*60)

# 5. Re-calculate Q-values after training
q_values_after = forward(start_state_tensor)
current_q_after = q_values_after[0, action].item()
print(f"Q-values: {q_values_after[0].detach().numpy()}")
print(f"Q-value for action {action}: {current_q_after:.4f}")

# 6. Calculate final loss
predicted_q_after = q_values_after[0, action]
loss_after = nn.functional.mse_loss(predicted_q_after, torch.tensor(target_q, dtype=torch.float32))
print(f"Loss (after): {loss_after.item():.6f}")

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"Q-value change: {current_q:.4f} → {current_q_after:.4f} (target: {target_q:.4f})")
print(f"Loss change: {loss_before.item():.6f} → {loss_after.item():.6f}")
print(f"Q-value moved toward target: {abs(current_q_after - target_q) < abs(current_q - target_q)}")
print(f"Loss decreased: {loss_after.item() < loss_before.item()}")