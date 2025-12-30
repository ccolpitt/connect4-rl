# *****************************************************************
# Full Training - Build it up step by step
# *****************************************************************

"""
Docstring for train_dqn_20251221

This will iteratively build up DQN training.
1: Create environment -- DONE
2: Create manual network based on what worked in manual_nn_training_test_20251221_v2 -- DONE
3: Play a single self-play game.  Use Eps Greedy with constant eps. -- DONE
    If greedy, do inference on state, and sample from the response - softmax
    At the end of the episode, if there was a winner, make sure is_done is true
    Also, override second to last reward to -1, and is_done to True
4: Play an ensemble of games to populate the replay buffer -- DONE
5: Train based on the replay buffer -- DONE
    Train 100 times on static replay buffer, sampling independently; verify loss goes down
    Once replay buffer is ready, train X times per Y games, samping Z samples; verify loss decreases
6: Implement training tracking
    Avg Abs Q Value prediction
    Avg Abs Q Value of Win/Loss position prediction (only where is_done = True)
    NN Loss by training event
    Win rate vs. random
    Unique States Explored
    
6: Play against the agent
"""


import sys
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from src.environment import ConnectFourEnvironment, Config
from src.utils import DQNReplayBuffer

# *****************************************************************
# Constants
# *****************************************************************


# *****************************************************************
# Create environment, Replay Buffer
# *****************************************************************
config = Config()
env = ConnectFourEnvironment(config)
replay_buffer = DQNReplayBuffer(capacity=10000)

#replay_buffer.add( [1,2,3], 0, 0, [2,3,4], False)
#replay_buffer.add( [2,3,4], 0, 1, [3,4,5], False)

"""
print( replay_buffer )
print( "Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-1]))
print( "Second Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-2]))
# Adjust reward of second to last entry
# PLACEHOLDER
"""

# *****************************************************************
# Create Network
# *****************************************************************
device = torch.device('cpu')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(32)
dr1 = nn.Dropout2d(p=config.DROPOUT_RATE)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(64)
dr2 = nn.Dropout2d(p=config.DROPOUT_RATE)
fc1 = nn.Linear(64 * 6 * 7, 128)
dr3 = nn.Dropout(p=config.DROPOUT_RATE)
output = nn.Linear(128, 7)

# He initialization
for layer in [conv1, conv2, fc1, output]:
    if hasattr(layer, 'weight'):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)

def forward(x):
    # Ensure input is a Float Tensor and has a batch dimension
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 3:
        x = x.unsqueeze(0) # Change (2,6,7) to (1,2,6,7)
        
    x = torch.relu(bn1(conv1(x)))
    x = dr1(x)
    x = torch.relu(bn2(conv2(x)))
    x = dr2(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(fc1(x))
    x = dr3(x)
    return output(x)

# *****************************************************************
# Test inference works with the function defined.  Pull examples from the notebook dir.
# *****************************************************************

# Get the absolute path of the current script
current_file = Path(__file__).resolve()
# Add project root to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Go up three levels to get to the project root
# .parent is training/ | .parent.parent is src/ | .parent.parent.parent is project_root/
project_root = current_file.parent.parent.parent
# import training examples
from notebooks.training_examples_last_2_moves_20251221 import get_training_examples

examples = get_training_examples()
initial_board = examples[0]['board']
players = examples[0]['players']
env.set_state( initial_board, players[0] )

q_values = forward(env.get_state())

#print( "Test Inference ")
#print( env.get_state() )
#print( q_values )


# *****************************************************************
# Select Action - Keep it simple to start
# *****************************************************************
def select_action(state, legal_moves, eps) -> int:
    """
    Implements epsilon-greedy action selection with illegal move masking.
    
    Args:
        state (torch.Tensor): 2x6x7 board state.
        legal_moves (list/np.array): List of valid column indices.
        eps (float): Probability of choosing a random action.
        
    Returns:
        int: The chosen action (0-6).
    """
    # 1. EXPLORATION: Randomly choose from legal moves
    if np.random.random() < eps:
        return int(np.random.choice(legal_moves))
    # 2. EXPLOITATION: Choose best action based on Q-values
    # Ensure we don't track gradients during inference
    with torch.no_grad():
        # Add batch dimension: (2, 6, 7) -> (1, 2, 6, 7)
        # Squeeze the output back to (7,)
        q_values = forward(state.unsqueeze(0)).squeeze(0)
        
        # MASKING: We want to ignore moves that are illegal.
        # We create a copy and set illegal moves to a very small value.
        masked_q = q_values.clone()
        
        # Create a set of all possible actions and find the illegal ones
        all_actions = set(range(7))
        illegal_actions = list(all_actions - set(legal_moves))
        
        # Set illegal moves to -infinity (or a very large negative number)
        # so they can never be the argmax.
        masked_q[illegal_actions] = -1e9
        
        # 3. Choose the action with the highest estimated Q-value
        best_action = torch.argmax(masked_q).item()
        
        return int(best_action)
    

# *****************************************************************
# Self Play.  Use eps-greedy
# *****************************************************************
def play_self_play_game():
    env.reset()
    done = False
    moves = 0
    eps = config.EPS

    # loop through moves
    while not done and moves < 42:
        state = env.get_state()     # This needs to return a tensor
        unique_states_seen.add(hash_state(state)) # TRACKING UNIQUE STATES
        legal_moves = env.get_legal_moves()
        
        # Eps greedy
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(state_tensor, legal_moves, eps)
        # Make the move
        next_state, reward, done = env.play_move( action )
        # Add to replay buffer
        replay_buffer.add( state, action, reward, next_state, done)
        moves += 1
    
    # Update the reward of the second to last move, if not a draw
    if( reward != 0 ):
        replay_buffer.update_penalty(-2, -1, True )
        #print( "SECOND TO LAST")
        #print( replay_buffer.buffer[-2] )

# Test out Self Play Game
#play_self_play_game()

# After self play game, print result
#print( "Final State:" )
#print( env.get_state() )
#print( env.get_current_player() )

#print( "Last Replay Buffer Entry: ")
#print( replay_buffer.buffer[-1] )

#print( "Second to Last Replay Buffer Entry: ")
#replay_buffer.buffer[-2]["reward"] = -1 # Update reward
#print( replay_buffer.buffer[-2] )


# *****************************************************************
# Helper functions, for training and metric tracking
# *****************************************************************
# Helper to hash states for Metric 4
def hash_state(state):
    return state.tobytes()

# Helper for Metric 6 (Placeholder - ensure you have a random agent function)
def evaluate_vs_random(num_games=20):
    wins = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = env.get_state()
            legal = env.get_legal_moves()
            # Agent move (Greedy)
            action = select_action(torch.tensor(state, dtype=torch.float32), legal, eps=0.0)
            _, reward, done = env.play_move(action)
            if done and reward == 1:
                wins += 1
                break
            if done: break # Draw or loss
            
            # Random move
            legal = env.get_legal_moves()
            action = np.random.choice(legal)
            _, reward, done = env.play_move(action)
            if done and reward == 1: # Random agent won
                break 
    return wins / num_games

# *****************************************************************
# Training Loop, Function: 
# *****************************************************************
num_episodes = 5  # Number of games
batch_size = 64
learning_rate = 0.00001
training_iterations = 100
gamma = 0.99
#eval_games: int = 100
#health_check_freq: int = 500
#save_freq: int = 5000
#model_save_path: str = None
#plot_save_path: str = None

# prelearn: Set to training mode
conv1.train()
bn1.train()
dr1.train()
conv2.train()
bn2.train()
dr2.train()
fc1.train()
dr3.train()
output.train()

# prelearn: I think this is to track NN weight and gradient health    
all_params = list(conv1.parameters()) + list(bn1.parameters()) + \
    list(conv2.parameters()) + list(bn2.parameters()) + \
    list(fc1.parameters()) + list(output.parameters())
    
# prelearn: Back propagation optimizer
optimizer = torch.optim.Adam(all_params, lr=learning_rate)

# Initialize learning result metrics
loss_history = []
grad_history = []
q_magnitude_history = []  # Store range of Q values- should trend towards +/- 1
q_magnitude_history_win_loss_positions = []
win_rate_vs_random = []
exploding_neurons = []
dead_neurons = []
unique_states_seen = set()
evaluation_frequency = 10 # Evaluate every 10 episodes


def train_dqn_agent():
    for episode in range(1, num_episodes + 1):
        # Play a single game, add moves to replay buffer
        play_self_play_game()
        print( "Episode ", episode, ".  ReplayBuffLen: ", replay_buffer.__len__())

        # Check if enough data to train the network.  Train if enough
        if replay_buffer.is_ready( batch_size ):
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Convert samples to pytorch tensors
            start_state_tensor_batch = torch.FloatTensor(states).to('cpu')
            actions_batch = torch.LongTensor(actions).to('cpu')
            rewards_tensor_batch = torch.FloatTensor(rewards).to('cpu')
            next_state_tensor_batch = torch.FloatTensor(next_states).to('cpu')
            dones_tensor_batch = torch.FloatTensor(dones).to('cpu')
            
            #print( "REPLAY BUFFER SAMPLE")
            #print( "Start State Tensor Batch", start_state_tensor_batch )
            #break

            # Calculate target Q values - note this stays the same for each batch
            # Discounted Bellman, with negamax logic
            with torch.no_grad():
                next_q_values = forward(next_state_tensor_batch)
                next_q_max = next_q_values.max(dim=1)[0]
                target_q = rewards_tensor_batch - (1 - dones_tensor_batch) * gamma * next_q_max

            #print( "SAMPLE DONES")
            #print( dones_tensor_batch )
            #print( "SAMPLE TARGET Q VALUES:")
            #print( target_q )
            #print( "SAMPLE SUM OF TARGET Q's")
            #print( torch.sum( target_q))
            #break

            # Train on the batch training_iterations times    
            for iteration in range(training_iterations):
                # Forward pass
                q_values = forward( start_state_tensor_batch ) # To Do: Make this dtype=torch.float32
                predicted_qs = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                #print( "SAMPLE PREDICTED Q VALUES")
                #print( predicted_qs )
                #print( "SAMPLE SUM PREDICTED Q's")
                #print( torch.sum( predicted_qs ))
                #break

                # Calculate loss between actual Q-value predictions (predicted_qs), and targets (which stay the same
                # over all episodes)
                loss = nn.functional.mse_loss(predicted_qs, target_q )

                # Backward pass --> I think this is where the real learning happens
                optimizer.zero_grad()   # Zero the gradients
                loss.backward()         # Back propagate loss gradients through the NN - ie - calculate gradients

                # Calculate average of abs gradient across all parameters - this should stay constant
                total_grad = 0.0
                total_params = 0
                for param in all_params:
                    if param.grad is not None:
                        total_grad += torch.sum(torch.abs(param.grad)).item()
                        total_params += param.grad.numel()
                avg_grad = total_grad / total_params if total_params > 0 else 0.0

                # Update NN weights -- THIS IS LEARNING - GREY MATTER UPDATE
                optimizer.step()
                
                # Calculate average magnitude of all Q-values
                q_magnitude = torch.mean(torch.abs(q_values[0])).item()

                # Calculate avg magnitude of Q vals where state is win or loss
                q_win_or_loss_magnitude = torch.mean(torch.abs(q_values[0])).item()

                # TRACK TRAINING STATS
                # Appears to work -- avg grad, loss, and q-value decrease
                print( "Episode: ", episode, "Episode Training Iteration: ", iteration)
                #print( "Total Parameters: ", total_params)
                #print( "Total Abs Gradient: ", total_grad)
                #print( "Average Grad: ", avg_grad, "Mean predicted Q value: ", q_magnitude, "Loss: ", loss )
                #print( "Mean predicted Q value: ", q_magnitude )

                # Store metrics for the current training episode
                loss_history.append(loss.item())                    # A scalar
                grad_history.append(avg_grad)                       # Scalar
                q_magnitude_history.append(q_magnitude)           # Scalar
                # Initialize learning result metrics

                """
                loss_history = [] # DONE
                grad_history = [] # DONE
                q_magnitude_history = []  # Store range of Q values- should trend towards +/- 1
                q_magnitude_history_win_loss_positions = []
                win_rate_vs_random = []
                exploding_neurons = []
                dead_neurons = []
                """




# test the train function
#train_dqn_agent()

# *****************************************************************
# Function to show result metrics 
# *****************************************************************

def plot_training_metrics( ):
    """
    Plot comprehensive training metrics.
    
    Args:
        metrics: TrainingMetrics object with collected data
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('DQN Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    if len(metrics.losses) > 0:
        ax.plot(metrics.episodes, metrics.losses, alpha=0.3, label='Loss')
        if len(metrics.losses) > 100:
            window = 100
            smoothed = np.convolve(metrics.losses, np.ones(window)/window, mode='valid')
            ax.plot(metrics.episodes[window-1:], smoothed, label=f'Smoothed (window={window})', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean Q-Values
    ax = axes[0, 1]
    if len(metrics.mean_q_values) > 0:
        ax.plot(metrics.episodes, metrics.mean_q_values, alpha=0.3, label='Mean Q')
        if len(metrics.mean_q_values) > 100:
            window = 100
            smoothed = np.convolve(metrics.mean_q_values, np.ones(window)/window, mode='valid')
            ax.plot(metrics.episodes[window-1:], smoothed, label=f'Smoothed (window={window})', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Q-Value')
        ax.set_title('Mean Q-Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    ax = axes[0, 2]
    if len(metrics.epsilons) > 0:
        ax.plot(metrics.episodes, metrics.epsilons, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (ε)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Win Rate vs Random
    ax = axes[1, 0]
    if len(metrics.win_rates) > 0:
        ax.plot(metrics.eval_episodes, metrics.win_rates, marker='o', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Random Agent')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Unique States Explored
    ax = axes[1, 1]
    if len(metrics.unique_states_counts) > 0:
        ax.plot(metrics.unique_states_episodes, metrics.unique_states_counts, marker='o', linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Unique States')
        ax.set_title('Unique States Explored')
        ax.grid(True, alpha=0.3)
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Plot 6: Dead Neurons
    ax = axes[1, 2]
    if len(metrics.dead_neuron_percentages) > 0:
        ax.plot(metrics.health_episodes, metrics.dead_neuron_percentages, linewidth=2, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Dead Neurons (%)')
        ax.set_title('Dead Neurons (|w| < 1e-3)')
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Exploding Weights
    ax = axes[2, 0]
    if len(metrics.exploding_weight_percentages) > 0:
        ax.plot(metrics.health_episodes, metrics.exploding_weight_percentages, linewidth=2, color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploding Weights (%)')
        ax.set_title('Exploding Weights (|w| > 100)')
        ax.grid(True, alpha=0.3)
    
    # Plot 8: Gradient Norm
    ax = axes[2, 1]
    if len(metrics.gradient_norms) > 0:
        ax.plot(metrics.health_episodes, metrics.gradient_norms, linewidth=2, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 9: Summary Statistics
    ax = axes[2, 2]
    ax.axis('off')
    if len(metrics.win_rates) > 0:
        final_win_rate = metrics.win_rates[-1]
        best_win_rate = max(metrics.win_rates)
        avg_loss = metrics.get_moving_average_loss()
        avg_q = metrics.get_moving_average_q()
        
        summary_text = f"""
        Training Summary
        ═══════════════════
        
        Final Win Rate: {final_win_rate:.1%}
        Best Win Rate: {best_win_rate:.1%}
        
        Avg Loss (last 100): {avg_loss:.4f}
        Avg Q-value (last 100): {avg_q:.2f}
        
        Final Epsilon: {metrics.epsilons[-1]:.4f}
        """
        
        # Add unique states if available
        if len(metrics.unique_states_counts) > 0:
            summary_text += f"""
        Unique States: {metrics.unique_states_counts[-1]:,}
        """
        
        # Add health metrics if available
        if len(metrics.dead_neuron_percentages) > 0:
            summary_text += f"""
        Dead Neurons: {metrics.dead_neuron_percentages[-1]:.2f}%
        Exploding Weights: {metrics.exploding_weight_percentages[-1]:.2f}%
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Training plots saved to: {save_path}")
    
    plt.show()

# *****************************************************************
# Train, show results
# *****************************************************************
