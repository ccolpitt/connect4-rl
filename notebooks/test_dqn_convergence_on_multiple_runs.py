# Test convergence over multiple runs to see if it's consistent
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append('..')

from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config
from src.agents.dqn_agent import DQNAgent

print("="*80)
print("CONVERGENCE TEST: Multiple Runs")
print("="*80)
print("Testing if Q-values consistently converge to targets")
print()

num_runs = 5
num_iterations = 100

all_winner_progressions = []
all_loser_progressions = []

for run in range(num_runs):
    print(f"\n{'='*80}")
    print(f"RUN {run + 1}/{num_runs}")
    print(f"{'='*80}")
    
    # Create fresh agent
    agent = DQNAgent(
        name=f"DQN-Run{run+1}",
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
        device='cpu'
    )
    
    # Setup environment
    config = Config()
    env = ConnectFourEnvironment(config)
    env.reset()
    env.play_move(0)  # Player 1
    env.play_move(4)  # Player -1
    env.play_move(1)  # Player 1
    env.play_move(5)  # Player -1
    env.play_move(2)  # Player 1
    
    # Player -1 makes losing move
    loser_state_before = env.get_state()
    loser_action = 6
    loser_next_state, loser_reward, loser_done = env.play_move(loser_action)
    
    # Player 1 makes winning move
    winner_state_before = env.get_state()
    winner_action = 3
    winner_next_state, winner_reward, winner_done = env.play_move(winner_action)
    
    # Test winner
    q_winner_initial = agent.get_q_values(winner_state_before)[winner_action]
    agent.observe(winner_state_before, winner_action, winner_reward, winner_next_state, winner_done)
    
    winner_progression = [q_winner_initial]
    for i in range(num_iterations):
        agent.train(sample_indices=[-1])
        q_current = agent.get_q_values(winner_state_before)[winner_action]
        winner_progression.append(q_current)
    
    all_winner_progressions.append(winner_progression)
    
    # Test loser
    q_loser_initial = agent.get_q_values(loser_state_before)[loser_action]
    agent.observe(loser_state_before, loser_action, -1.0, loser_next_state, True)
    
    loser_progression = [q_loser_initial]
    for i in range(num_iterations):
        agent.train(sample_indices=[-1])
        q_current = agent.get_q_values(loser_state_before)[loser_action]
        loser_progression.append(q_current)
    
    all_loser_progressions.append(loser_progression)
    
    print(f"Winner: {q_winner_initial:+.4f} → {winner_progression[-1]:+.4f} (target: +1.0)")
    print(f"Loser:  {q_loser_initial:+.4f} → {loser_progression[-1]:+.4f} (target: -1.0)")

# Analyze convergence
print(f"\n{'='*80}")
print("CONVERGENCE ANALYSIS")
print(f"{'='*80}")

winner_finals = [prog[-1] for prog in all_winner_progressions]
loser_finals = [prog[-1] for prog in all_loser_progressions]

print(f"\nWinner Q-values after {num_iterations} iterations:")
for i, q in enumerate(winner_finals):
    print(f"  Run {i+1}: {q:+.4f} (distance from +1.0: {abs(q-1.0):.4f})")
print(f"  Mean: {np.mean(winner_finals):+.4f}")
print(f"  Std:  {np.std(winner_finals):.4f}")

print(f"\nLoser Q-values after {num_iterations} iterations:")
for i, q in enumerate(loser_finals):
    print(f"  Run {i+1}: {q:+.4f} (distance from -1.0: {abs(q-(-1.0)):.4f})")
print(f"  Mean: {np.mean(loser_finals):+.4f}")
print(f"  Std:  {np.std(loser_finals):.4f}")

# Check if converging
print(f"\n{'='*80}")
print("CONVERGENCE CHECK")
print(f"{'='*80}")

# Check last 10 iterations for stability
for run_idx, prog in enumerate(all_winner_progressions):
    last_10 = prog[-10:]
    variance = np.var(last_10)
    mean_last_10 = np.mean(last_10)
    print(f"Winner Run {run_idx+1}: Last 10 mean={mean_last_10:+.4f}, variance={variance:.6f}")

print()
for run_idx, prog in enumerate(all_loser_progressions):
    last_10 = prog[-10:]
    variance = np.var(last_10)
    mean_last_10 = np.mean(last_10)
    print(f"Loser Run {run_idx+1}: Last 10 mean={mean_last_10:+.4f}, variance={variance:.6f}")

# Plot all runs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Winner progressions
ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Target (+1.0)', zorder=1)
for i, prog in enumerate(all_winner_progressions):
    ax1.plot(range(len(prog)), prog, alpha=0.7, linewidth=2, label=f'Run {i+1}', zorder=2)
ax1.set_xlabel('Training Iteration')
ax1.set_ylabel('Q-Value')
ax1.set_title(f'Winner Q-Value Convergence ({num_runs} runs)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loser progressions
ax2.axhline(y=-1.0, color='r', linestyle='--', linewidth=2, label='Target (-1.0)', zorder=1)
for i, prog in enumerate(all_loser_progressions):
    ax2.plot(range(len(prog)), prog, alpha=0.7, linewidth=2, label=f'Run {i+1}', zorder=2)
ax2.set_xlabel('Training Iteration')
ax2.set_ylabel('Q-Value')
ax2.set_title(f'Loser Q-Value Convergence ({num_runs} runs)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_analysis.png', dpi=150)
print(f"\nPlot saved to convergence_analysis.png")
plt.show()

# Final verdict
print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")

winner_converged = all(abs(q - 1.0) < 0.2 for q in winner_finals)
loser_converged = all(abs(q - (-1.0)) < 0.2 for q in loser_finals)

if winner_converged and loser_converged:
    print("✓ Both winner and loser Q-values converge reasonably close to targets")
    print(f"  Winner within 0.2 of +1.0: {winner_converged}")
    print(f"  Loser within 0.2 of -1.0: {loser_converged}")
else:
    print("✗ Q-values do NOT converge properly")
    print(f"  Winner within 0.2 of +1.0: {winner_converged}")
    print(f"  Loser within 0.2 of -1.0: {loser_converged}")
    print("\nThis suggests a fundamental issue with the training logic!")