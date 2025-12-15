# ============================================================
# DEEP DEBUG: Analyze Single Training Step
# ============================================================

import sys
import numpy as np
import torch

sys.path.append('..')

from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config
from src.agents.dqn_agent import DQNAgent

print("="*80)
print("DEEP DEBUG: Single Training Step Analysis")
print("="*80)
print()

# ============================================================
# 1. CREATE AGENT
# ============================================================

agent = DQNAgent(
    name="DQN-Debug",
    player_id=1,
    conv_channels=[16, 32],
    fc_dims=[64],
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=1,
    min_buffer_size=1,
    buffer_size=1000,
    target_update_freq=1000,
    use_double_dqn=True
)

print("Agent created")
print(f"Device: {agent.device}")
print()

# ============================================================
# 2. CREATE SIMPLE WINNING SCENARIO
# ============================================================

config = Config()
env = ConnectFourEnvironment(config)

# Setup: X X X . O O .
env.reset()
env.play_move(0)  # Player 1
env.play_move(4)  # Player -1
env.play_move(1)  # Player 1
env.play_move(5)  # Player -1
env.play_move(2)  # Player 1
env.current_player = 1  # Ensure Player 1's turn

print("Board setup (Player 1 can win at column 3):")
env.render()
print()

# ============================================================
# 3. GET STATE AND MAKE WINNING MOVE
# ============================================================

state_before = env.get_state()
action = 3
next_state, reward, done = env.play_move(action)

print("Move details:")
print(f"  Action: {action}")
print(f"  Reward: {reward}")
print(f"  Done: {done}")
print(f"  State shape: {state_before.shape}")
print(f"  Next state shape: {next_state.shape}")
print()

# ============================================================
# 4. ANALYZE INITIAL Q-VALUES
# ============================================================

print("="*80)
print("INITIAL Q-VALUES")
print("="*80)

q_initial = agent.get_q_values(state_before)
print("Q-values for all actions:")
for col in range(7):
    marker = " <-- WINNING ACTION" if col == action else ""
    print(f"  Column {col}: {q_initial[col]:+.6f}{marker}")
print()

# ============================================================
# 5. MANUALLY COMPUTE TARGET
# ============================================================

print("="*80)
print("TARGET COMPUTATION")
print("="*80)

with torch.no_grad():
    state_tensor = torch.FloatTensor(state_before).unsqueeze(0).to(agent.device)
    next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
    
    # Current Q-value
    current_q_all = agent.q_network(state_tensor)[0]
    current_q = current_q_all[action].item()
    
    print(f"Current Q(s,a): {current_q:+.6f}")
    
    # Next Q-value (from target network)
    if agent.use_double_dqn:
        next_q_online = agent.q_network(next_tensor)[0]
        next_action = next_q_online.argmax().item()
        next_q_target = agent.target_network(next_tensor)[0]
        next_q_max = next_q_target[next_action].item()
        print(f"Next Q (Double DQN):")
        print(f"  Online network best action: {next_action}")
        print(f"  Target network Q for that action: {next_q_max:+.6f}")
    else:
        next_q_max = agent.target_network(next_tensor)[0].max().item()
        print(f"Next Q max: {next_q_max:+.6f}")
    
    # Compute target using negamax
    if done:
        target = reward
        print(f"\nGame is done, so target = reward")
    else:
        target = reward - agent.gamma * next_q_max
        print(f"\nGame continues, so target = reward - gamma * next_q_max")
    
    print(f"\nTarget computation:")
    print(f"  Reward: {reward:+.6f}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Next Q max: {next_q_max:+.6f}")
    print(f"  Target: {target:+.6f}")
    
    td_error = target - current_q
    expected_loss = td_error ** 2
    
    print(f"\nTD Error: {td_error:+.6f}")
    print(f"Expected Loss: {expected_loss:.6f}")
print()

# ============================================================
# 6. STORE EXPERIENCE AND CHECK BUFFER
# ============================================================

print("="*80)
print("REPLAY BUFFER")
print("="*80)

agent.observe(state_before, action, reward, next_state, done)
print(f"Stored experience in buffer")
print(f"Buffer size: {len(agent.replay_buffer)}")

# Verify what's in the buffer
last_exp = agent.replay_buffer.buffer[-1]
print(f"\nLast experience in buffer:")
print(f"  State shape: {last_exp[0].shape}")
print(f"  Action: {last_exp[1]}")
print(f"  Reward: {last_exp[2]}")
print(f"  Next state shape: {last_exp[3].shape}")
print(f"  Done: {last_exp[4]}")
print()

# ============================================================
# 7. PERFORM ONE TRAINING STEP WITH DETAILED LOGGING
# ============================================================

print("="*80)
print("TRAINING STEP")
print("="*80)

# Get weights before
first_param_before = list(agent.q_network.parameters())[0][0, 0, 0, 0].item()
print(f"Sample weight before training: {first_param_before:.8f}")

# Train
metrics = agent.train(sample_indices=[-1])

print(f"\nTraining metrics:")
print(f"  Loss: {metrics['loss']:.6f}")
print(f"  Mean Q-value: {metrics['mean_q_value']:+.6f}")
print(f"  Mean target Q: {metrics['mean_target_q']:+.6f}")

# Get weights after
first_param_after = list(agent.q_network.parameters())[0][0, 0, 0, 0].item()
weight_change = first_param_after - first_param_before
print(f"\nSample weight after training: {first_param_after:.8f}")
print(f"Weight change: {weight_change:+.8f}")
print()

# ============================================================
# 8. ANALYZE Q-VALUES AFTER TRAINING
# ============================================================

print("="*80)
print("Q-VALUES AFTER TRAINING")
print("="*80)

q_after = agent.get_q_values(state_before)
print("Q-values for all actions:")
for col in range(7):
    change = q_after[col] - q_initial[col]
    marker = " <-- WINNING ACTION" if col == action else ""
    print(f"  Column {col}: {q_after[col]:+.6f} (change: {change:+.6f}){marker}")
print()

# ============================================================
# 9. ANALYSIS AND DIAGNOSIS
# ============================================================

print("="*80)
print("ANALYSIS")
print("="*80)

q_change = q_after[action] - q_initial[action]
print(f"\nWinning action Q-value:")
print(f"  Before: {q_initial[action]:+.6f}")
print(f"  After:  {q_after[action]:+.6f}")
print(f"  Change: {q_change:+.6f}")
print(f"  Target: {target:+.6f}")

# Check direction
moved_toward_target = abs(q_after[action] - target) < abs(q_initial[action] - target)
print(f"\nMoved toward target: {'✓ YES' if moved_toward_target else '✗ NO'}")

# Check for explosion
if abs(q_after[action]) > 2.0:
    print(f"\n🚨 EXPLOSION DETECTED!")
    print(f"Q-value exploded to {q_after[action]:+.6f}")
    print(f"This indicates a fundamental bug!")
    
    print(f"\nPossible causes:")
    print(f"1. Target computation is wrong")
    print(f"2. Loss computation is wrong")
    print(f"3. Gradient is wrong")
    print(f"4. Learning rate is too high")
    print(f"5. Network has no output bounds")
    
    # Check if target is reasonable
    if abs(target) > 1.5:
        print(f"\n⚠️  Target value {target:+.6f} is outside expected range [-1, 1]!")
        print(f"   This suggests the negamax formula or reward is wrong")
    
    # Check if loss is reasonable
    if metrics['loss'] > 10.0:
        print(f"\n⚠️  Loss {metrics['loss']:.6f} is very high!")
        print(f"   Expected loss for this scenario: ~{expected_loss:.6f}")
        print(f"   Actual loss is {metrics['loss']/expected_loss:.1f}x higher than expected")

print()
print("="*80)
print("DEBUG COMPLETE")
print("="*80)