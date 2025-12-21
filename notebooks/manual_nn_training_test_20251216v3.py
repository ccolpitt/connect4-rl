# *****************************************************************
# Learning rate comparison test - same initial weights, different LRs
# *****************************************************************
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import copy
from training_examples import get_training_examples
from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config

# *****************************************************************
# Variables
test_example = 1
training_iterations = 100
learning_rates = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
dropout_rate = 0.0

# *****************************************************************
# Load test case
examples = get_training_examples()
states = examples[test_example]['board']
target_policy = examples[test_example]['target_policy']
reward = examples[test_example]['reward']
player = examples[test_example]['player']

action = np.random.choice(7, p=target_policy)

# *****************************************************************
# Create environment
config = Config()
env = ConnectFourEnvironment(config)

env.set_state(state, player)
print("Starting State:")
env.render()
print(f"Start player: {env.get_current_player()}")
print(f"Action: {action}")
print(f"Target reward: {reward}")

# *****************************************************************
# Generate Replay Buffer Entry
start_state = env.get_state_from_perspective(env.get_current_player())
end_state, reward, done = env.play_move(action)

# *****************************************************************
# Create INITIAL network (will be copied for each learning rate)
import torch
import torch.nn as nn
import os

# Force CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
device = torch.device('cpu')
torch.set_default_device('cpu')
print(f"\nUsing device: {device}")
print(f"Dropout rate: {dropout_rate}")
print(f"Training iterations: {training_iterations}")
print(f"Learning rates to test: {learning_rates}")

# Create initial network layers
conv1_init = nn.Conv2d(2, 32, kernel_size=3, padding=1)
bn1_init = nn.BatchNorm2d(32)
dropout1_init = nn.Dropout2d(p=dropout_rate)
conv2_init = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2_init = nn.BatchNorm2d(64)
dropout2_init = nn.Dropout2d(p=dropout_rate)
fc1_init = nn.Linear(64 * 6 * 7, 128)
dropout3_init = nn.Dropout(p=dropout_rate)
output_init = nn.Linear(128, 7)

# Initialize weights ONCE
nn.init.kaiming_normal_(conv1_init.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(conv1_init.bias, 0)
nn.init.constant_(bn1_init.weight, 1)
nn.init.constant_(bn1_init.bias, 0)
nn.init.kaiming_normal_(conv2_init.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(conv2_init.bias, 0)
nn.init.constant_(bn2_init.weight, 1)
nn.init.constant_(bn2_init.bias, 0)
nn.init.kaiming_normal_(fc1_init.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(fc1_init.bias, 0)
nn.init.kaiming_normal_(output_init.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(output_init.bias, 0)

print("\n✓ Initial weights created and will be reused for all learning rates")

# *****************************************************************
# Prepare tensors
start_state_tensor = torch.tensor(start_state, dtype=torch.float32).unsqueeze(0)
action_tensor = torch.tensor(action, dtype=torch.long)
reward_tensor = torch.tensor(reward, dtype=torch.float32)
end_state_tensor = torch.tensor(end_state, dtype=torch.float32)
done_tensor = torch.tensor(float(done), dtype=torch.float32)

gamma = 0.99

# *****************************************************************
# Train with each learning rate
# *****************************************************************
print("\n" + "="*80)
print("TRAINING WITH DIFFERENT LEARNING RATES")
print("="*80)

# Storage for all results
all_results = {}

for lr in learning_rates:
    print(f"\n{'='*80}")
    print(f"Testing learning rate: {lr}")
    print(f"{'='*80}")
    
    # Create COPIES of initial network for this learning rate
    conv1 = copy.deepcopy(conv1_init)
    bn1 = copy.deepcopy(bn1_init)
    dropout1 = copy.deepcopy(dropout1_init)
    conv2 = copy.deepcopy(conv2_init)
    bn2 = copy.deepcopy(bn2_init)
    dropout2 = copy.deepcopy(dropout2_init)
    fc1 = copy.deepcopy(fc1_init)
    dropout3 = copy.deepcopy(dropout3_init)
    output = copy.deepcopy(output_init)
    
    # Forward pass function
    def forward(x):
        x = torch.relu(bn1(conv1(x)))
        x = dropout1(x)
        x = torch.relu(bn2(conv2(x)))
        x = dropout2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(fc1(x))
        x = dropout3(x)
        return output(x)
    
    # Set to training mode
    conv1.train()
    bn1.train()
    dropout1.train()
    conv2.train()
    bn2.train()
    dropout2.train()
    fc1.train()
    dropout3.train()
    output.train()
    
    # Create optimizer with this learning rate
    all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
                 list(conv2.parameters()) + list(bn2.parameters()) + \
                 list(fc1.parameters()) + list(output.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
    # Calculate target Q-value (same for all)
    with torch.no_grad():
        next_q_values = forward(end_state_tensor.unsqueeze(0))
        next_q_max = next_q_values.max(dim=1)[0]
        target_q = reward_tensor - (1 - done_tensor) * gamma * next_q_max
        target_q_value = target_q.item()
    
    # Storage for this learning rate
    q_values_history = []
    loss_history = []
    grad_history = []
    q_magnitude_history = []
    
    # Training loop
    for iteration in range(training_iterations):
        # Forward pass
        q_values = forward(start_state_tensor)
        predicted_q = q_values[0, action]
        
        # Calculate loss
        loss = nn.functional.mse_loss(predicted_q, torch.tensor(target_q_value, dtype=torch.float32))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Calculate average gradient across all parameters
        total_grad = 0.0
        total_params = 0
        for param in all_params:
            if param.grad is not None:
                total_grad += torch.sum(torch.abs(param.grad)).item()
                total_params += param.grad.numel()
        avg_grad = total_grad / total_params if total_params > 0 else 0.0
        
        # Update weights
        optimizer.step()
        
        # Calculate average magnitude of all Q-values
        q_magnitude = torch.mean(torch.abs(q_values[0])).item()
        
        # Store metrics
        q_values_history.append(predicted_q.item())
        loss_history.append(loss.item())
        grad_history.append(avg_grad)
        q_magnitude_history.append(q_magnitude)
    
    # Store results for this learning rate
    all_results[lr] = {
        'q_values': q_values_history,
        'loss': loss_history,
        'gradients': grad_history,
        'q_magnitudes': q_magnitude_history,
        'final_q': q_values_history[-1],
        'final_loss': loss_history[-1],
        'final_q_magnitude': q_magnitude_history[-1]
    }
    
    print(f"  Final Q-value: {q_values_history[-1]:.4f} (target: {target_q_value:.4f})")
    print(f"  Final loss: {loss_history[-1]:.6f}")
    print(f"  Distance from target: {abs(q_values_history[-1] - target_q_value):.4f}")

# *****************************************************************
# Visualization
# *****************************************************************
print("\n" + "="*80)
print("CREATING COMPARISON VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = ['blue', 'green', 'red', 'purple', 'orange']

# Plot 1: Q-value convergence
ax = axes[0, 0]
for i, lr in enumerate(learning_rates):
    ax.plot(all_results[lr]['q_values'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.axhline(target_q_value, color='black', linestyle='--', linewidth=2,
           label=f'Target ({target_q_value:.2f})')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Q-Value')
ax.set_title('Q-Value Convergence Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Loss over time
ax = axes[0, 1]
for i, lr in enumerate(learning_rates):
    ax.plot(all_results[lr]['loss'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training Loss Comparison')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Average gradients
ax = axes[1, 0]
for i, lr in enumerate(learning_rates):
    ax.plot(all_results[lr]['gradients'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Average Absolute Gradient')
ax.set_title('Gradient Magnitude Comparison')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Average Q-value magnitude
ax = axes[1, 1]
for i, lr in enumerate(learning_rates):
    ax.plot(all_results[lr]['q_magnitudes'], linewidth=2,
            label=f'LR={lr}', color=colors[i])
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Average |Q-value|')
ax.set_title('Q-Value Magnitude Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_comparison_20251216v3.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: learning_rate_comparison_20251216v3.png")
plt.show()

# *****************************************************************
# Summary table
# *****************************************************************
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"\n{'Learning Rate':>15} | {'Final Q-Value':>15} | {'Final Loss':>15} | {'Avg |Q|':>12} | {'Distance to Target':>20}")
print("-" * 95)
for lr in learning_rates:
    final_q = all_results[lr]['final_q']
    final_loss = all_results[lr]['final_loss']
    final_q_mag = all_results[lr]['final_q_magnitude']
    distance = abs(final_q - target_q_value)
    print(f"{lr:15.6f} | {final_q:15.4f} | {final_loss:15.6f} | {final_q_mag:12.4f} | {distance:20.4f}")

# Find best learning rate
best_lr = min(learning_rates, key=lambda lr: abs(all_results[lr]['final_q'] - target_q_value))
print(f"\n✓ Best learning rate: {best_lr} (closest to target)")

print("\n" + "="*80)