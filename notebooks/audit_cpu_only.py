# Force CPU to avoid MPS issues
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append('..')

from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config
from src.agents.dqn_agent import DQNAgent

print("="*80)
print("AUDIT: CPU Only (avoiding MPS issues)")
print("="*80)
print()

# Create agent with CPU device explicitly
agent = DQNAgent(
    name="DQN-CPU",
    player_id=1,
    conv_channels=[16, 32],
    fc_dims=[64],
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=1,
    min_buffer_size=1,
    buffer_size=1000,
    target_update_freq=1000,
    use_double_dqn=True,
    device='cpu'  # Force CPU
)

print(f"Device: {agent.device}")
print()

config = Config()
env = ConnectFourEnvironment(config)

# Setup board
env.reset()
env.play_move(0)  # Player 1
env.play_move(4)  # Player -1
env.play_move(1)  # Player 1
env.play_move(5)  # Player -1
env.play_move(2)  # Player 1

print("Board prior to loser making move:")
env.render()
print()

# Make losing move
loser_state_before = env.get_state()
loser_action = 6
loser_next_state, loser_reward, loser_done = env.play_move(loser_action)

print("Board after loser makes losing move, prior to winner making winning move:")
env.render()
print()

# Player 1 makes winning move
winner_state_before = env.get_state()
winner_action = 3
winner_next_state, winner_reward, winner_done = env.play_move(winner_action)

print("Final board:")
env.render()
print()

# ============================================================
# TEST WINNER EXPERIENCE (20 iterations)
# ============================================================

print("="*80)
print("WINNER EXPERIENCE (20 iterations on CPU)")
print("="*80)

q_winner_before = agent.get_q_values(winner_state_before)[winner_action]
print(f"Initial Q-value: {q_winner_before:+.4f}")
print(f"Target: +1.0")

print( " *** Information going into the winning replay buffer: ***")
print( "Winner State Before Winning Move")
print( winner_state_before )
print( "Winner Action: ", winner_action )
print( "Winner reward: ", winner_reward )
print( "Winner Next State:")
print( winner_next_state )
print( "Winner done: ", winner_done )

agent.observe(winner_state_before, winner_action, winner_reward, winner_next_state, winner_done)

winner_q_progression = [q_winner_before]
winner_losses = []

print("\nTraining 20 iterations...")
for i in range(20):
    metrics = agent.train(sample_indices=[-1])
    q_current = agent.get_q_values(winner_state_before)[winner_action]
    winner_q_progression.append(q_current)
    winner_losses.append(metrics['loss'])
    
    if i % 5 == 0 or i == 19:
        print(f"  Iteration {i+1:2d}: Loss={metrics['loss']:8.4f}, Q={q_current:+.4f}")

q_winner_after = winner_q_progression[-1]
dist_before = abs(q_winner_before - 1.0)
dist_after = abs(q_winner_after - 1.0)
winner_success = dist_after < dist_before

print(f"\nFinal Q-value: {q_winner_after:+.4f}")
print(f"Distance from +1.0: {dist_before:.4f} → {dist_after:.4f}")
print(f"Result: {'✓ SUCCESS' if winner_success else '✗ FAILURE'}")
print()

# ============================================================
# TEST LOSER EXPERIENCE (20 iterations)
# ============================================================

print("="*80)
print("LOSER EXPERIENCE (20 iterations on CPU)")
print("="*80)

q_loser_before = agent.get_q_values(loser_state_before)[loser_action]
print(f"Initial Q-value: {q_loser_before:+.4f}")
print(f"Target: -1.0")

agent.observe(loser_state_before, loser_action, -1.0, loser_next_state, True)

loser_q_progression = [q_loser_before]
loser_losses = []

print("\nTraining 20 iterations...")
for i in range(20):
    metrics = agent.train(sample_indices=[-1])
    q_current = agent.get_q_values(loser_state_before)[loser_action]
    loser_q_progression.append(q_current)
    loser_losses.append(metrics['loss'])
    
    if i % 5 == 0 or i == 19:
        print(f"  Iteration {i+1:2d}: Loss={metrics['loss']:8.4f}, Q={q_current:+.4f}")

q_loser_after = loser_q_progression[-1]
dist_before = abs(q_loser_before - (-1.0))
dist_after = abs(q_loser_after - (-1.0))
loser_success = dist_after < dist_before

print(f"\nFinal Q-value: {q_loser_after:+.4f}")
print(f"Distance from -1.0: {dist_before:.4f} → {dist_after:.4f}")
print(f"Result: {'✓ SUCCESS' if loser_success else '✗ FAILURE'}")
print()

# ============================================================
# RESULTS
# ============================================================

print("="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nWinner: {q_winner_before:+.4f} → {q_winner_after:+.4f} (target: +1.0) {'✓' if winner_success else '✗'}")
print(f"Loser:  {q_loser_before:+.4f} → {q_loser_after:+.4f} (target: -1.0) {'✓' if loser_success else '✗'}")
print(f"\nOverall: {'✓✓ BOTH PASSED' if winner_success and loser_success else '✗✗ ONE OR MORE FAILED'}")
print("="*80)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(range(len(winner_q_progression)), winner_q_progression, marker='o', markersize=4, linewidth=2)
ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Target (+1.0)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Q-Value')
ax.set_title('Winner Q-Value Progression (CPU)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(range(1, len(winner_losses)+1), winner_losses, marker='o', markersize=4, linewidth=2, color='orange')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Winner Loss')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(range(len(loser_q_progression)), loser_q_progression, marker='o', markersize=4, linewidth=2, color='purple')
ax.axhline(y=-1.0, color='r', linestyle='--', linewidth=2, label='Target (-1.0)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Q-Value')
ax.set_title('Loser Q-Value Progression (CPU)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(range(1, len(loser_losses)+1), loser_losses, marker='o', markersize=4, linewidth=2, color='brown')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Loser Loss')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()