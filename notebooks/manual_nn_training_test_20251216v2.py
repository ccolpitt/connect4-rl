# *****************************************************************
# Multi-iteration training test with gradient tracking
# *****************************************************************
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from training_examples import get_training_examples
from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config

# *****************************************************************
# Variables
test_example = 20
training_iterations = 100
learning_rate = 0.00001
dropout_rate = 0.0  # 0.0 = no dropout, 0.5 = 50% dropout, 1.0 = 100% dropout (no training)

# *****************************************************************
# Load test case
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
# Create Policy NN
import torch
import torch.nn as nn
import os

# Force CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
device = torch.device('cpu')
torch.set_default_device('cpu')
print(f"\nUsing device: {device}")
print(f"Learning rate: {learning_rate}")
print(f"Dropout rate: {dropout_rate}")
print(f"Training iterations: {training_iterations}")

# Create network layers
conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(32)
dropout1 = nn.Dropout2d(p=dropout_rate)  # Dropout after first conv block
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)
dropout2 = nn.Dropout2d(p=dropout_rate)  # Dropout after second conv block
fc1 = nn.Linear(64 * 6 * 7, 128)
dropout3 = nn.Dropout(p=dropout_rate)  # Dropout after FC layer
output = nn.Linear(128, 7)

# Initialize weights
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
    x = dropout1(x)
    x = torch.relu(bn2(conv2(x)))
    x = dropout2(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(fc1(x))
    x = dropout3(x)
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

# Create optimizer (dropout layers have no parameters)
all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
             list(conv2.parameters()) + list(bn2.parameters()) + \
             list(fc1.parameters()) + list(output.parameters())
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

# Set network to training mode (important for dropout and batchnorm)
conv1.train()
bn1.train()
dropout1.train()
conv2.train()
bn2.train()
dropout2.train()
fc1.train()
dropout3.train()
output.train()

# Calculate target Q-value (constant throughout training)
with torch.no_grad():
    next_q_values = forward(end_state_tensor.unsqueeze(0))
    next_q_max = next_q_values.max(dim=1)[0] # Note this is never used if done=True
    target_q = reward_tensor - (1 - done_tensor) * gamma * next_q_max
    target_q_value = target_q.item()

print(f"\nTarget Q-value: {target_q_value:.4f}")

# *****************************************************************
# Training loop with tracking
# *****************************************************************
print("\n" + "="*80)
print("TRAINING LOOP")
print("="*80)

# Storage for metrics
q_values_history = []
loss_history = []
conv1_grad_history = []
output_grad_history = []

# Print header
print(f"\n{'Iter':>6} | {'Q-value':>10} | {'Loss':>12} | {'Conv1 Grad':>12} | {'Output Grad':>12}")
print("-" * 80)

for iteration in range(training_iterations):
    # Forward pass
    q_values = forward(start_state_tensor)
    predicted_q = q_values[0, action]
    
    # Calculate loss
    loss = nn.functional.mse_loss(predicted_q, torch.tensor(target_q_value, dtype=torch.float32))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Extract gradients
    conv1_grad = torch.mean(torch.abs(conv1.weight.grad)).item()
    output_grad = torch.mean(torch.abs(output.weight.grad)).item()
    
    # Update weights
    optimizer.step()
    
    # Store metrics
    q_values_history.append(predicted_q.item())
    loss_history.append(loss.item())
    conv1_grad_history.append(conv1_grad)
    output_grad_history.append(output_grad)
    
    # Print progress (every 10 iterations or last iteration)
    if iteration % 10 == 0 or iteration == training_iterations - 1:
        print(f"{iteration:6d} | {predicted_q.item():10.4f} | {loss.item():12.6f} | "
              f"{conv1_grad:12.6f} | {output_grad:12.6f}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

# Final Q-values
final_q_values = forward(start_state_tensor)
print(f"\nFinal Q-values: {final_q_values[0].detach().numpy()}")
print(f"Final Q-value for action {action}: {final_q_values[0, action].item():.4f}")
print(f"Target Q-value: {target_q_value:.4f}")
print(f"Final loss: {loss_history[-1]:.6f}")

# *****************************************************************
# Visualization
# *****************************************************************
print("\n" + "="*80)
print("CREATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Q-value convergence
ax = axes[0, 0]
ax.plot(q_values_history, linewidth=2, label='Q-value')
ax.axhline(target_q_value, color='red', linestyle='--', linewidth=2, label=f'Target ({target_q_value:.2f})')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Q-Value')
ax.set_title(f'Q-Value Convergence (LR={learning_rate}, Dropout={dropout_rate})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Loss over time
ax = axes[0, 1]
ax.plot(loss_history, linewidth=2, color='orange')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training Loss')
ax.set_yscale('log')  # Log scale to see convergence better
ax.grid(True, alpha=0.3)

# Plot 3: Conv1 gradients
ax = axes[1, 0]
ax.plot(conv1_grad_history, linewidth=2, color='green')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Mean Absolute Gradient')
ax.set_title('Conv1 Weight Gradients')
ax.grid(True, alpha=0.3)

# Plot 4: Output gradients
ax = axes[1, 1]
ax.plot(output_grad_history, linewidth=2, color='purple')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Mean Absolute Gradient')
ax.set_title('Output Layer Weight Gradients')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_convergence_20251216v2.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: training_convergence_20251216v2.png")
plt.show()

# *****************************************************************
# Summary statistics
# *****************************************************************
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nQ-value:")
print(f"  Initial: {q_values_history[0]:.4f}")
print(f"  Final: {q_values_history[-1]:.4f}")
print(f"  Target: {target_q_value:.4f}")
print(f"  Distance from target: {abs(q_values_history[-1] - target_q_value):.4f}")

print(f"\nLoss:")
print(f"  Initial: {loss_history[0]:.6f}")
print(f"  Final: {loss_history[-1]:.6f}")
print(f"  Reduction: {(1 - loss_history[-1]/loss_history[0])*100:.2f}%")

print(f"\nGradients:")
print(f"  Conv1 - Mean: {np.mean(conv1_grad_history):.6f}, Max: {np.max(conv1_grad_history):.6f}")
print(f"  Output - Mean: {np.mean(output_grad_history):.6f}, Max: {np.max(output_grad_history):.6f}")

# Check for convergence
converged = abs(q_values_history[-1] - target_q_value) < 0.1
print(f"\nConvergence status: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}")
print(f"  (Q-value within 0.1 of target: {converged})")

print("\n" + "="*80)