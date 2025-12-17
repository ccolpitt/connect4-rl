"""
Neural Network Learning Verification Test

This script tests whether the DQN network can learn from hand-picked examples.
We'll populate the replay buffer with known good/bad moves and verify:
1. Replay buffer contains correct data
2. Training samples the right experiences
3. Loss decreases over iterations
4. Q-values move in the correct direction (toward +1 for wins, -1 for losses)

This is a prerequisite before attempting full self-play training.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append('..')

from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config
from src.agents.dqn_agent import DQNAgent
from training_examples import get_training_examples

print("="*80)
print("NEURAL NETWORK LEARNING VERIFICATION")
print("="*80)
print("Testing if NN can learn from hand-picked examples")
print()

# ============================================================
# TRAINING EXAMPLES
# ============================================================
# Load all training examples
examples = get_training_examples()

print("="*80)
print("TRAINING EXAMPLES VISUALIZATION")
print("="*80)
print(f"Total examples loaded: {len(examples)}")
print()

# Select which example to visualize (change this number to see different examples)
EXAMPLE_INDEX = 38

# Get the selected example
example = examples[EXAMPLE_INDEX]

print(f"Example {EXAMPLE_INDEX}:")
print(f"Description: {example['description']}")
print(f"Player to move: {example['player']}")
print(f"Correct move(s): {example['move']}")
print(f"Target policy: {example['target_policy']}")
print(f"Expected reward: {example['reward']}")
print()

# Create environment and set the board state
config = Config()
env = ConnectFourEnvironment(config)

# Set the board state directly
env.board = example['board'].copy()
env.current_player = example['player']

# Render the board
print("Board state:")
env.render()
print()

# Show which columns are the correct moves
print("Correct move(s) highlighted:")
for col in range(7):
    if example["target_policy"][col]  > 0:
        print(f"  Column {col}: *** CORRECT MOVE ***")
    else:
        print(f"  Column {col}:")

print()
print("="*80)
print("To view other examples, change EXAMPLE_INDEX at the top of this script")
print(f"Valid indices: 0 to {len(examples)-1}")
print("="*80)

sdfd

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def board_to_canonical_state(board, player):
    """
    Convert board to canonical state from player's perspective.
    
    Args:
        board: 2D numpy array (6x7) with 1=Player1, -1=Player2, 0=empty
        player: 1 or -1 (whose perspective)
    
    Returns:
        state: 3D numpy array (2, 6, 7) canonical representation
    """
    state = np.zeros((2, 6, 7), dtype=np.float32)
    
    if player == 1:
        # Player 1's perspective
        state[0] = (board == 1).astype(np.float32)   # My pieces
        state[1] = (board == -1).astype(np.float32)  # Opponent's pieces
    else:
        # Player 2's perspective
        state[0] = (board == -1).astype(np.float32)  # My pieces
        state[1] = (board == 1).astype(np.float32)   # Opponent's pieces
    
    return state

def setup_environment_from_board(env, board, player):
    """Set environment to match the given board state."""
    env.board = board.copy()
    env.current_player = player
    env.last_move = None
    return env

def create_experiences_from_example(example, env):
    """
    Create replay buffer experiences from a training example.
    If there are multiple correct moves, creates one experience for each.
    
    Returns:
        List of (state, action, reward, next_state, done) tuples
    """
    experiences = []
    
    # Setup environment
    board = example['board']
    player = example['player']
    correct_moves = example['correct_moves']  # Now a list
    reward = example['reward']
    
    # Get state before move (from current player's perspective)
    state = board_to_canonical_state(board, player)
    
    # Create an experience for each correct move
    for action in correct_moves:
        setup_environment_from_board(env, board, player)
        
        # Make the move
        next_state_env, reward_env, done = env.play_move(action)
        
        # For winning moves, use the provided reward
        # For losing moves, we need to simulate opponent winning
        if reward == 1.0:
            # This is a winning move
            next_state = next_state_env
            done = True
        else:
            # This is a losing move (opponent will win)
            next_state = next_state_env
            done = True
            reward = -1.0
        
        experiences.append((state, action, reward, next_state, done))
    
    return experiences

# ============================================================
# TEST SETUP
# ============================================================

print("="*80)
print("SETUP")
print("="*80)

# Create agent with CPU (to avoid MPS issues)
agent = DQNAgent(
    name="DQN-Supervised",
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

config = Config()
env = ConnectFourEnvironment(config)

print(f"Agent: {agent.name}")
print(f"Device: {agent.device}")
print(f"Network: {sum(p.numel() for p in agent.q_network.parameters())} parameters")
print()

# ============================================================
# POPULATE REPLAY BUFFER
# ============================================================

print("="*80)
print("POPULATING REPLAY BUFFER")
print("="*80)

experience_count = 0
for i, example in enumerate(training_examples):
    experiences = create_experiences_from_example(example, env)
    
    print(f"Example {i+1}: {example['description']}")
    print(f"  Correct moves: {example['correct_moves']}")
    print(f"  Creating {len(experiences)} experience(s)")
    
    for exp_idx, (state, action, reward, next_state, done) in enumerate(experiences):
        agent.observe(state, action, reward, next_state, done)
        print(f"    Experience {exp_idx+1}: action={action}, reward={reward:+.1f}, done={done}")
        experience_count += 1
    print()

print(f"Replay buffer size: {len(agent.replay_buffer)}")
print()

# ============================================================
# VERIFY BUFFER CONTENTS
# ============================================================

print("="*80)
print("VERIFY BUFFER CONTENTS")
print("="*80)

for i in range(len(agent.replay_buffer)):
    exp = agent.replay_buffer.buffer[i]
    state, action, reward, next_state, done = exp
    print(f"Buffer[{i}]: action={action}, reward={reward:+.1f}, done={done}, state_shape={state.shape}")

print()

# ============================================================
# TRAINING TEST
# ============================================================

print("="*80)
print("TRAINING TEST")
print("="*80)

num_iterations = 50
print(f"Training for {num_iterations} iterations on all examples...")
print()

# Track metrics for each experience in the buffer
experience_metrics = []
buffer_idx = 0

for example_idx, example in enumerate(training_examples):
    experiences = create_experiences_from_example(example, env)
    
    print(f"Example {example_idx+1}: {example['description']}")
    
    # Train on each experience from this example
    for exp_idx, (state, action, reward, next_state, done) in enumerate(experiences):
        print(f"  Training on action {action} (buffer index {buffer_idx})")
        print(f"    Target: Q(state, action={action}) → {reward:+.1f}")
        
        # Get initial Q-value
        q_initial = agent.get_q_values(state)[action]
        print(f"    Initial Q-value: {q_initial:+.4f}")
        
        # Train on this specific experience
        q_progression = [q_initial]
        losses = []
        
        for iter in range(num_iterations):
            metrics = agent.train(sample_indices=[buffer_idx])
            q_current = agent.get_q_values(state)[action]
            q_progression.append(q_current)
            losses.append(metrics['loss'])
        
        q_final = q_progression[-1]
        print(f"    Final Q-value: {q_final:+.4f}")
        print(f"    Change: {q_final - q_initial:+.4f}")
        print(f"    Distance from target: {abs(q_final - reward):.4f}")
        print(f"    Final loss: {losses[-1]:.6f}")
        
        # Check if moved toward target
        moved_toward_target = abs(q_final - reward) < abs(q_initial - reward)
        print(f"    Result: {'✓ MOVED TOWARD TARGET' if moved_toward_target else '✗ MOVED AWAY'}")
        print()
        
        experience_metrics.append({
            'description': f"{example['description']} (action {action})",
            'q_initial': q_initial,
            'q_final': q_final,
            'target': reward,
            'q_progression': q_progression,
            'losses': losses,
            'success': moved_toward_target
        })
        
        buffer_idx += 1

# ============================================================
# RESULTS
# ============================================================

print("="*80)
print("RESULTS")
print("="*80)

success_count = sum(1 for m in experience_metrics if m['success'])
total_count = len(experience_metrics)

print(f"\nSuccess rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
print()

for i, metrics in enumerate(experience_metrics):
    status = "✓" if metrics['success'] else "✗"
    print(f"{status} Experience {i+1}: {metrics['q_initial']:+.4f} → {metrics['q_final']:+.4f} (target: {metrics['target']:+.1f})")
    print(f"   {metrics['description']}")

# ============================================================
# VISUALIZATION
# ============================================================

if len(experience_metrics) > 0:
    fig, axes = plt.subplots(len(experience_metrics), 2, figsize=(14, 4*len(experience_metrics)))
    
    if len(experience_metrics) == 1:
        axes = axes.reshape(1, -1)
    
    for i, metrics in enumerate(experience_metrics):
        # Q-value progression
        ax = axes[i, 0]
        ax.plot(range(len(metrics['q_progression'])), metrics['q_progression'], 
                marker='o', markersize=3, linewidth=2)
        ax.axhline(y=metrics['target'], color='r', linestyle='--', linewidth=2, 
                   label=f"Target ({metrics['target']:+.1f})")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Q-Value')
        ax.set_title(f"Experience {i+1}: Q-Value Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss progression
        ax = axes[i, 1]
        ax.plot(range(1, len(metrics['losses'])+1), metrics['losses'], 
                marker='o', markersize=3, linewidth=2, color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f"Example {i+1}: Loss")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_learning_verification.png', dpi=150)
    print(f"\nPlot saved to nn_learning_verification.png")
    plt.show()

# ============================================================
# VERDICT
# ============================================================

print()
print("="*80)
print("VERDICT")
print("="*80)

if success_count == total_count:
    print("✓✓ ALL EXAMPLES LEARNED SUCCESSFULLY")
    print("The neural network can learn from supervised examples!")
    print("Ready to proceed with RL training.")
else:
    print(f"✗✗ ONLY {success_count}/{total_count} EXAMPLES LEARNED")
    print("The neural network is NOT learning properly.")
    print("Need to debug the training logic before proceeding.")

print("="*80)