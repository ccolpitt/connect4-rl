# *****************************************************************
# Import libraries
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from training_examples import get_training_examples
from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config

# *****************************************************************
# Variables
test_example = 1

# *****************************************************************
# Load all test cases
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
env.set_state(state, player)
print("Starting State:")
env.render()
print("Start player: ", env.get_current_player())

# *****************************************************************
# Generate Replay Buffer Entries
start_state = env.get_state_from_perspective(env.get_current_player())
end_state, reward, done = env.play_move(action)

# *****************************************************************
# Create Policy NN, initialize weights
import torch
import torch.nn as nn
import os

# Use CPU, to avoid MPS math errors
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
device = torch.device('cpu')
torch.set_default_device('cpu')
print(f"Using device: {device}")

# Create network layers manually (default: conv=[32,64], fc=[128])
conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # [2, 3, 3]
bn1 = nn.BatchNorm2d(32)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)
fc1 = nn.Linear(64 * 6 * 7, 128)
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
# Prepare tensors
start_state_tensor = torch.tensor(start_state, dtype=torch.float32).unsqueeze(0)
action_tensor = torch.tensor(action, dtype=torch.long)
reward_tensor = torch.tensor(reward, dtype=torch.float32)
end_state_tensor = torch.tensor(end_state, dtype=torch.float32)
done_tensor = torch.tensor(float(done), dtype=torch.float32)

# Hyperparameters
gamma = 0.99
learning_rate = 0.001

# Create optimizer
all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
             list(conv2.parameters()) + list(bn2.parameters()) + \
             list(fc1.parameters()) + list(output.parameters())
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

print("\n" + "="*60)
print("BEFORE TRAINING")
print("="*60)

# 1. Calculate initial Q-values
q_values = forward(start_state_tensor)
current_q = q_values[0, action].item()
print(f"Q-values: {q_values[0].detach().numpy()}")
print(f"Q-value for action {action}: {current_q:.4f}")

# 2. Calculate target Q-value (negamax formula)
with torch.no_grad():
    next_q_values = forward(end_state_tensor.unsqueeze(0))
    next_q_max = next_q_values.max(dim=1)[0]
    target_q = reward_tensor - (1 - done_tensor) * gamma * next_q_max
    target_q_value = target_q.item()

print(f"Target Q-value: {target_q_value:.4f}")
print(f"Reward: {reward_tensor.item():.4f}, Done: {done_tensor.item()}")

# 3. Calculate initial loss
predicted_q = q_values[0, action]
loss_before = nn.functional.mse_loss(predicted_q, torch.tensor(target_q_value, dtype=torch.float32))
print(f"Loss (before): {loss_before.item():.6f}")

print("\n" + "="*60)
print("TRAINING (1 step)")
print("="*60)

# 4. Backpropagation
optimizer.zero_grad()
loss = nn.functional.mse_loss(predicted_q, torch.tensor(target_q_value, dtype=torch.float32))
loss.backward()

# *****************************************************************
# GRADIENT ANALYSIS
# *****************************************************************
print("\n" + "="*60)
print("GRADIENT ANALYSIS")
print("="*60)

# Collect all gradients
all_gradients = []
layer_names = []
layer_stats = []

for name, param in [('conv1.weight', conv1.weight), ('conv1.bias', conv1.bias),
                    ('bn1.weight', bn1.weight), ('bn1.bias', bn1.bias),
                    ('conv2.weight', conv2.weight), ('conv2.bias', conv2.bias),
                    ('bn2.weight', bn2.weight), ('bn2.bias', bn2.bias),
                    ('fc1.weight', fc1.weight), ('fc1.bias', fc1.bias),
                    ('output.weight', output.weight), ('output.bias', output.bias)]:
    if param.grad is not None:
        grad_np = param.grad.detach().numpy().flatten()
        all_gradients.extend(grad_np)
        
        # Calculate statistics per layer
        grad_mean = np.mean(np.abs(grad_np))
        grad_max = np.max(np.abs(grad_np))
        grad_std = np.std(grad_np)
        
        layer_names.append(name)
        layer_stats.append({
            'mean_abs': grad_mean,
            'max_abs': grad_max,
            'std': grad_std,
            'size': len(grad_np)
        })
        
        print(f"{name:20s}: mean_abs={grad_mean:.6f}, max_abs={grad_max:.6f}, std={grad_std:.6f}")

# Convert to numpy array
all_gradients = np.array(all_gradients)

print(f"\nOverall gradient statistics:")
print(f"  Total parameters: {len(all_gradients)}")
print(f"  Mean absolute gradient: {np.mean(np.abs(all_gradients)):.6f}")
print(f"  Max absolute gradient: {np.max(np.abs(all_gradients)):.6f}")
print(f"  Std of gradients: {np.std(all_gradients):.6f}")
print(f"  Gradient range: [{np.min(all_gradients):.6f}, {np.max(all_gradients):.6f}]")

# Check for exploding gradients
if np.max(np.abs(all_gradients)) > 10.0:
    print("\n⚠️  WARNING: EXPLODING GRADIENTS DETECTED! (max > 10.0)")
elif np.max(np.abs(all_gradients)) > 1.0:
    print("\n⚠️  WARNING: Large gradients detected (max > 1.0)")
else:
    print("\n✓ Gradients are in reasonable range")

# Update weights
optimizer.step()
print(f"\nWeights updated with learning rate: {learning_rate}")

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
loss_after = nn.functional.mse_loss(predicted_q_after, torch.tensor(target_q_value, dtype=torch.float32))
print(f"Loss (after): {loss_after.item():.6f}")

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"Q-value change: {current_q:.4f} → {current_q_after:.4f} (target: {target_q_value:.4f})")
print(f"Loss change: {loss_before.item():.6f} → {loss_after.item():.6f}")
print(f"Q-value moved toward target: {abs(current_q_after - target_q_value) < abs(current_q - target_q_value)}")
print(f"Loss decreased: {loss_after.item() < loss_before.item()}")

# *****************************************************************
# VISUALIZATION
# *****************************************************************
print("\n" + "="*60)
print("CREATING GRADIENT VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram of all gradients
ax = axes[0, 0]
ax.hist(all_gradients, bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Gradient Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of All Gradients')
ax.axvline(0, color='red', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3)

# Plot 2: Gradient magnitude by layer
ax = axes[0, 1]
layer_means = [stats['mean_abs'] for stats in layer_stats]
ax.barh(range(len(layer_names)), layer_means)
ax.set_yticks(range(len(layer_names)))
ax.set_yticklabels(layer_names, fontsize=8)
ax.set_xlabel('Mean Absolute Gradient')
ax.set_title('Gradient Magnitude by Layer')
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Max gradient by layer
ax = axes[1, 0]
layer_maxs = [stats['max_abs'] for stats in layer_stats]
ax.barh(range(len(layer_names)), layer_maxs, color='orange')
ax.set_yticks(range(len(layer_names)))
ax.set_yticklabels(layer_names, fontsize=8)
ax.set_xlabel('Max Absolute Gradient')
ax.set_title('Maximum Gradient by Layer')
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Q-value comparison
ax = axes[1, 1]
q_before = q_values[0].detach().numpy()
q_after = q_values_after[0].detach().numpy()
x = np.arange(7)
width = 0.35
ax.bar(x - width/2, q_before, width, label='Before', alpha=0.7)
ax.bar(x + width/2, q_after, width, label='After', alpha=0.7)
ax.axhline(target_q_value, color='red', linestyle='--', label=f'Target ({target_q_value:.2f})')
ax.set_xlabel('Action (Column)')
ax.set_ylabel('Q-Value')
ax.set_title('Q-Values Before vs After Training')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/gradient_analysis_20251216.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: notebooks/gradient_analysis_20251216.png")
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)