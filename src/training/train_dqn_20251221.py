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
6: Implement training tracking - DONE
    Avg Abs Q Value prediction
    Avg Abs Q Value of Win/Loss position prediction (only where is_done = True)
    NN Loss by training event
    Win rate vs. random
    Unique States Explored
6: Perspective test - test with three in a row from play 1 and 2 perspective.  Verify that 
    Q values also change a lot.
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
import matplotlib.pyplot as plt

# *****************************************************************
# Constants
# *****************************************************************
num_episodes                = 200   # Number of games to train on
batch_size                  = 16
learning_rate               = 0.00001
training_iterations         = 1     # Training per game
eval_vs_random_game_count   = 50
gamma                       = 0.99
evaluation_frequency        = 10    # Evaluate every 10 episodes
#eval_games: int = 100
#health_check_freq: int = 500
#save_freq: int = 5000
#model_save_path: str = None
#plot_save_path: str = None


# *****************************************************************
# Create environment, Replay Buffer
# *****************************************************************
config = Config()
env = ConnectFourEnvironment(config)
replay_buffer = DQNReplayBuffer(capacity=20000)

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
import torch
import torch.nn as nn
import copy

class Connect4Net(nn.Module):
    def __init__(self, device, dropout_rate=0.2):
        super(Connect4Net, self).__init__()
        self.device = device
        
        # Define layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dr1 = nn.Dropout2d(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dr2 = nn.Dropout2d(p=dropout_rate)
        
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.dr3 = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(128, 7)

        # Apply He initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Move the entire model to the specified device immediately
        self.to(self.device)

    def forward(self, x):
        # 1. Convert NumPy to Tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # 2. Ensure input is on the SAME device as the model layers
        x = x.to(self.device)
        
        # 3. Handle batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dr1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dr2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dr3(x)
        return self.output(x)
    
# Initialize models
my_device = torch.device("cpu")
policy_net = Connect4Net(device=my_device, dropout_rate=config.DROPOUT_RATE)
target_net = Connect4Net(device=my_device, dropout_rate=config.DROPOUT_RATE)

# Sync weights
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Simplified Optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

# *****************************************************************
# Test inference works with the function defined.  Pull examples from the notebook dir.
# *****************************************************************
example_test_case = 5

# Get the absolute path of the current script
current_file = Path(__file__).resolve()
# Add project root to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Go up three levels to get to the project root
# .parent is training/ | .parent.parent is src/ | .parent.parent.parent is project_root/
project_root = current_file.parent.parent.parent
# import training examples
from notebooks.training_examples_last_2_moves_20251221 import get_training_examples

"""
examples = get_training_examples()

initial_board = examples[example_test_case]['board']
players = examples[example_test_case]['players']
env.set_state( initial_board, players[0] )

print( "Initial board ")
print( initial_board)
print( "Initial Player: ", players[0] )
print( "Initial Board as stored in environment" )
print( env.get_state() )
print( "Test Inference ")
state_tensor = torch.FloatTensor(env.get_state()).to(Config.DEVICE)
q_values = target_net(state_tensor)
print( "Initial Q Values: ", q_values )
"""
# *****************************************************************
# Helper functions, for training and metric tracking
# *****************************************************************
# Helper to hash states for Metric 4
def hash_state(state):
    return state.tobytes()

# Helper for Metric 6 (Placeholder - ensure you have a random agent function)
def evaluate_vs_random(policy_net = policy_net, num_games=20):
    wins = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = env.get_state()
            legal = env.get_legal_moves()
            # Agent move (Greedy)
            action = select_action(policy_net, torch.tensor(state, dtype=torch.float32), legal, eps=0.0)
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

def get_next_q_masked(next_states, next_legal_masks):
    with torch.no_grad():
        # next_q_values shape: (batch_size, 7)
        next_q_values = forward(next_states)
        
        # Apply mask: set illegal moves to -1e9
        # next_legal_masks should be a tensor (batch_size, 7) where 1=legal, 0=illegal
        masked_q = next_q_values.masked_fill(next_legal_masks == 0, -1e9)
        
        return masked_q.max(dim=1)[0]

def get_action_mask(legal_moves):
    """Converts [0, 1, 3] into [1, 1, 0, 1, 0, 0, 0]"""
    mask = np.zeros(7, dtype=np.float32)
    mask[legal_moves] = 1.0
    return mask


# *****************************************************************
# Select Action - Keep it simple to start
# *****************************************************************
def select_action(policy_net, state, legal_moves, eps) -> int:
    # 1. EXPLORATION: Randomly choose from legal moves
    if np.random.random() < eps:
        return int(np.random.choice(legal_moves))
        
    # 2. EXPLOITATION: 
    # CRITICAL: Switch to eval mode to disable Dropout/BatchNorm noise
    policy_net.eval() 
    
    with torch.no_grad():
        # The class handles unsqueeze and .to(device) internally now, 
        # but calling it here is fine for clarity.
        q_values = policy_net(state).squeeze(0)
        
        # MASKING
        masked_q = q_values.clone()
        all_actions = set(range(7))
        illegal_actions = list(all_actions - set(legal_moves))
        
        # Using a very large negative ensures illegal moves aren't picked
        masked_q[illegal_actions] = -1e9
        
        best_action = torch.argmax(masked_q).item()
        
    # Note: We don't call .train() here because this function is also 
    # used during evaluation. We let the training loop handle switching 
    # back to .train() when it's ready to update weights.
    
    return int(best_action)
    

# *****************************************************************
# Self Play.  Use eps-greedy
# *****************************************************************
def play_self_play_game(policy_net, eps=0.5):
    env.reset()
    done = False
    moves_count = 0
    game_states = [] # To return to the training loop for unique state tracking

    while not done and moves_count < 42:
        state = env.get_state()
        game_states.append(state)
        
        legal_moves = env.get_legal_moves()
        
        # 1. Select Action (Note: Pass policy_net, state, and legal_moves)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(policy_net, state_tensor, legal_moves, eps)
        
        # 2. Execute Move
        next_state, reward, done = env.play_move(action)
        
        # 3. Handle Masking for the NEXT player
        if not done:
            next_legal_moves = env.get_legal_moves()
            next_mask = get_action_mask(next_legal_moves)
        else:
            # No legal moves after game ends
            next_mask = np.zeros(7, dtype=np.float32)

        # 4. Add to Replay Buffer
        replay_buffer.add(state, action, reward, next_state, done, next_mask)
        moves_count += 1
    
    # 5. THE BELLMAN NEGATIVE REWARD FIX
    # In Connect 4, if the last move resulted in 1.0 (Win), 
    # the move immediately before that by the opponent was a Loss (-1.0).
    if reward == 1.0:
        # We reach into the buffer and update the transition for the opponent
        # -2 refers to the second-to-last item added to the buffer
        replay_buffer.update_penalty(index=-2, new_reward=-1.0, is_done=True)
    
    return game_states
"""
def play_self_play_game(policy_net, eps = 0.5):
    env.reset()
    done = False
    moves = 0

    # loop through moves
    while not done and moves < 42:
        state = env.get_state()     # This needs to return a tensor
        unique_states_seen.add(hash_state(state)) # TRACKING UNIQUE STATES
        legal_moves = env.get_legal_moves()
        
        # Eps greedy
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(policy_net, state_tensor, legal_moves, eps)
        # Make the move
        next_state, reward, done = env.play_move( action )
        # Get mask for the NEXT player ---
        if not done:
            next_legal_moves = env.get_legal_moves()
            next_mask = get_action_mask(next_legal_moves)
        else:
            next_mask = np.zeros(7, dtype=np.float32)

        # Add to replay buffer
        replay_buffer.add( state, action, reward, next_state, done, next_mask)
        moves += 1
    
    # Update the reward of the second to last move, if not a draw
    if( reward != 0 ):
        replay_buffer.update_penalty(-2, -1, True )
        #print( "SECOND TO LAST")
        #print( replay_buffer.buffer[-2] )
"""
"""
# Test out Self Play Game
play_self_play_game(policy_net, eps=0.5)

# After self play game, print result
print( "Final State:" )
print( env.get_state() )
print( env.get_current_player() )

print( "Last Replay Buffer Entry: ")
print( replay_buffer.buffer[-1] )

print( "Second to Last Replay Buffer Entry: ")
replay_buffer.buffer[-2]["reward"] = -1 # Update reward
print( replay_buffer.buffer[-2] )
"""

# *****************************************************************
# Training Loop, Function: 
# *****************************************************************

# Initialize learning result metrics
loss_history = []           # 1: Should decrease
q_magnitude_history = []    # 2: Store range of Q values- should trend towards [0-1]
q_magnitude_history_win_loss = []   #3: should trend towards 1
unique_states_history = []  # 4: Should increase
win_rate_vs_rand_hist = []  # 5: Should increase from 50% to 100%
dead_neuron_cnt_hist = []   # 6: Should stay constant
exploding_neuron_cnt_hist = []  # 7: Should stay constant
grad_history = []           # 8: Should stay constant, or decrease
terminal_pct_history = []   # Initialize new history list
unique_states_seen = set()

def train_dqn_agent(policy_net, optimizer, config):
    # 1. SETUP
    target_net = copy.deepcopy(policy_net) 
    target_net.eval()
    
    eps = config.EPS_START 
    
    for episode in range(1, config.NUM_EPISODES + 1):
        # 2. SELF PLAY
        new_states_seen = play_self_play_game(policy_net, eps) 
        
        for s in new_states_seen:
            unique_states_seen.add(s.tobytes()) 
            
        # 3. EPSILON DECAY
        eps = max(config.EPS_END, eps * config.EPS_DECAY)

        # 4. TRAINING LOOP
        if replay_buffer.is_ready(config.BATCH_SIZE):
            policy_net.train()
            batch_terminal_counts = []
            
            # We track the 'last' values of the loop to append to history
            last_loss = 0
            last_q_values = None
            last_predicted_qs = None
            last_dones = None

            for _ in range(config.TRAIN_N_TIMES_PER_GAME): 
                #states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(config.BATCH_SIZE)
                states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(
                    config.BATCH_SIZE, 
                    terminal_ratio=0.25 # Our target ratio
                )


                s_batch = torch.tensor(states, dtype=torch.float32).to(my_device)
                a_batch = torch.tensor(actions, dtype=torch.long).to(my_device)
                r_batch = torch.tensor(rewards, dtype=torch.float32).to(my_device)
                ns_batch = torch.tensor(next_states, dtype=torch.float32).to(my_device)
                d_batch = torch.tensor(dones, dtype=torch.float32).to(my_device)
                m_batch = torch.tensor(next_masks, dtype=torch.float32).to(my_device)

                with torch.no_grad():
                    next_q_values = target_net(ns_batch) 
                    masked_next_q = next_q_values.masked_fill(m_batch == 0, -1e9)
                    next_q_max = masked_next_q.max(dim=1)[0]
                    target_q = r_batch + (config.GAMMA * next_q_max * (1 - d_batch))

                optimizer.zero_grad()
                q_values = policy_net(s_batch)
                predicted_qs = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                loss = nn.functional.mse_loss(predicted_qs, target_q)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                
                # Capture data for metrics
                last_loss = loss.item()
                last_q_values = q_values.detach()
                last_predicted_qs = predicted_qs.detach()
                last_dones = d_batch.detach()

                actual_pct = np.mean(dones) 
                batch_terminal_counts.append(actual_pct)

            # --- SYNCHRONIZED METRIC LOGGING ---
            # All these must be appended together so the list lengths match
            loss_history.append(last_loss)
            q_magnitude_history.append(torch.mean(torch.abs(last_q_values)).item())
            unique_states_history.append(len(unique_states_seen))

            # Metric 3: Terminal Q-Mag (Win/Loss states)
            # Find terminal states in the last batch
            terminal_mask = (last_dones == 1)
            if terminal_mask.any():
                term_q = torch.mean(torch.abs(last_predicted_qs[terminal_mask])).item()
            else:
                # Fallback if no terminal state in batch: use batch average
                term_q = q_magnitude_history[-1]
            q_magnitude_history_win_loss.append(term_q)

            # Metric 6 & 7: Neuron Health
            with torch.no_grad():
                all_w = torch.cat([p.view(-1) for p in policy_net.parameters()])
                dead_neuron_cnt_hist.append((torch.abs(all_w) < 0.01).sum().item() / all_w.numel())
                exploding_neuron_cnt_hist.append((torch.abs(all_w) > 10.0).sum().item() / all_w.numel())

            # Metric 8: Gradients
            total_grad = 0.0
            grad_count = 0
            for p in policy_net.parameters():
                if p.grad is not None:
                    total_grad += p.grad.abs().sum().item()
                    grad_count += p.grad.numel()
            grad_history.append(total_grad / grad_count if grad_count > 0 else 0)

            # Metric 9: Terminal % of Batch
            terminal_pct_history.append(np.mean(batch_terminal_counts))

            # 5. SYNC TARGET NETWORK
            if episode % config.TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # 6. EVALUATION
            if episode % 100 == 0:
                win_rate = evaluate_vs_random(policy_net, num_games=20)
                win_rate_vs_rand_hist.append(win_rate)
                print(f"Ep {episode} | Eps: {eps:.2f} | Unique: {len(unique_states_seen)} | WinRate: {win_rate:.2f}")

    return policy_net

"""
def train_dqn_agent(policy_net, optimizer, config):
    # 1. SETUP
    target_net = copy.deepcopy(policy_net) 
    target_net.eval()
    
    # Epsilon setup: start high (random), end low (expert)
    eps = config.EPS_START 
    
    for episode in range(1, config.NUM_EPISODES + 1):
        # 2. SELF PLAY
        # Pass current epsilon to the game player
        # play_self_play_game should return the moves or just add them to replay_buffer
        new_states_seen = play_self_play_game(policy_net, eps) 
        
        # Track unique states
        for s in new_states_seen:
            # Convert 2x6x7 board to a string or tuple to make it hashable
            unique_states_seen.add(s.tobytes()) 
            
        # 3. EPSILON DECAY
        eps = max(config.EPS_END, eps * config.EPS_DECAY)

        # 4. TRAINING LOOP (Perform multiple updates per game)
        # Often we train 'N' times per episode to speed up learning
        if replay_buffer.is_ready(config.BATCH_SIZE):
            policy_net.train()
            
            for _ in range(config.TRAIN_N_TIMES_PER_GAME): 
                # Sample and move to device
                states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(config.BATCH_SIZE)
                
                s_batch = torch.tensor(states, dtype=torch.float32).to(my_device)
                a_batch = torch.tensor(actions, dtype=torch.long).to(my_device)
                r_batch = torch.tensor(rewards, dtype=torch.float32).to(my_device)
                ns_batch = torch.tensor(next_states, dtype=torch.float32).to(my_device)
                d_batch = torch.tensor(dones, dtype=torch.float32).to(my_device)
                m_batch = torch.tensor(next_masks, dtype=torch.float32).to(my_device)

                # CALCULATE TARGETS
                with torch.no_grad():
                    next_q_values = target_net(ns_batch) 
                    masked_next_q = next_q_values.masked_fill(m_batch == 0, -1e9)
                    next_q_max = masked_next_q.max(dim=1)[0]
                    # Bellman: r + gamma * max(Q)
                    # (Note: In your logic, ensure r is relative to the player whose turn it is)
                    target_q = r_batch + (config.GAMMA * next_q_max * (1 - d_batch))

                # GRADIENT STEP
                optimizer.zero_grad()
                q_values = policy_net(s_batch)
                predicted_qs = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                loss = nn.functional.mse_loss(predicted_qs, target_q)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            # --- METRICS (Updated for Self-Play) ---
            loss_history.append(loss.item())
            q_magnitude_history.append(torch.mean(torch.abs(q_values)).item())
            unique_states_history.append(len(unique_states_seen))

            # 5. SYNC TARGET NETWORK (e.g., every 50 episodes)
            if episode % config.TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # 6. EVALUATION
            if episode % 20 == 0:
                win_rate = evaluate_vs_random(policy_net, num_games=20)
                win_rate_vs_rand_hist.append(win_rate)
                print(f"Ep {episode} | Eps: {eps:.2f} | Unique States: {len(unique_states_seen)} | Win Rate: {win_rate:.2f}")

    return policy_net
"""
                

# Synthetic training with a contrived replay buffer - prove that the network will learn
# win and loss

from notebooks.training_examples_last_2_moves_20251221 import generate_artificial_replay_buffer_for_training
import copy
def train_on_synthetic_replay_buffer(policy_net, optimizer):
    # 1. SETUP: Static buffer and target net
    replay_buffer = generate_artificial_replay_buffer_for_training()
    batch_size = 16 
    target_net = copy.deepcopy(policy_net) 
    target_net.eval()

    # Move batch to device once
    states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(batch_size)
    s_batch = torch.tensor(states, dtype=torch.float32).to(my_device)
    a_batch = torch.tensor(actions, dtype=torch.long).to(my_device)
    r_batch = torch.tensor(rewards, dtype=torch.float32).to(my_device)
    ns_batch = torch.tensor(next_states, dtype=torch.float32).to(my_device)
    d_batch = torch.tensor(dones, dtype=torch.float32).to(my_device)
    m_batch = torch.tensor(next_masks, dtype=torch.float32).to(my_device)

    policy_net.train()

    for episode in range(1, num_episodes + 1):
        # 3. CALCULATE TARGETS
        with torch.no_grad():
            next_q_values = target_net(ns_batch) 
            masked_next_q = next_q_values.masked_fill(m_batch == 0, -1e9)
            next_q_max = masked_next_q.max(dim=1)[0]
            target_q = r_batch - (1 - d_batch) * gamma * next_q_max

        # 4. GRADIENT STEP
        optimizer.zero_grad()
        q_values = policy_net(s_batch)
        predicted_qs = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
        
        loss = nn.functional.mse_loss(predicted_qs, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        # --- POPULATE METRICS ---
        # 1. Loss
        loss_history.append(loss.item())

        # 2. Avg Absolute Magnitude of ALL Q-values in the batch
        q_magnitude_history.append(torch.mean(torch.abs(q_values)).item())

        # 3. Avg Absolute Magnitude of the Q-values for the CHOSEN moves
        q_magnitude_history_win_loss.append(torch.mean(torch.abs(predicted_qs)).item())

        # 4. Unique States (Static at 16 for this test)
        unique_states_history.append(batch_size)

        # 5. Win Rate vs Random (Every 20 episodes)
        if episode % 20 == 0:
            # Note: evaluate_vs_random should call policy_net.eval() internally
            win_rate = evaluate_vs_random(num_games=20)
            win_rate_vs_rand_hist.append(win_rate)
            policy_net.train() # Switch back to training mode after eval

        # 6 & 7. Neuron Health (Weights Health)
        with torch.no_grad():
            all_weights = torch.cat([p.view(-1) for p in policy_net.parameters()])
            total_params = all_weights.numel()
            
            dead_neurons = (torch.abs(all_weights) < 0.01).sum().item()
            exploding_neurons = (torch.abs(all_weights) > 10.0).sum().item()
            
            dead_neuron_cnt_hist.append(dead_neurons / total_params) # Stored as %
            exploding_neuron_cnt_hist.append(exploding_neurons / total_params)

        # 8. Average Gradient Magnitude
        total_grad = 0.0
        grad_count = 0
        for p in policy_net.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().sum().item()
                grad_count += p.grad.numel()
        grad_history.append(total_grad / grad_count if grad_count > 0 else 0)

        #9. Batch terminal position percentage
        actual_pct = np.mean(dones)
        terminal_pct_history.append(actual_pct)

        # 5. SYNC TARGET NETWORK
        if episode % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Ep {episode} | Loss: {loss.item():.4f} | Q-Mag: {q_magnitude_history[-1]:.2f} | Grad: {grad_history[-1]:.6f}")

    return policy_net


# *****************************************************************
# Function to show result metrics 
# *****************************************************************
def plot_training_metrics(loss_hist, q_hist, q_terminal_hist, states_hist, 
                          win_rate_hist, dead_hist, exploding_hist, grad_hist, 
                          terminal_pct_hist, eval_freq=10):
    """
    Simplified plotting function for DQN training.
    6 subplots consolidating 8 metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Connect-4 DQN Training Dashboard', fontsize=16, fontweight='bold')
    
    # Generate x-axis indices
    episodes = np.arange(len(loss_hist))
    eval_episodes = np.arange(len(win_rate_hist)) * eval_freq

    # --- Plot 1: Loss ---
    ax = axes[0, 0]
    ax.plot(episodes, loss_hist, color='blue', alpha=0.3)
    if len(loss_hist) > 10:
        smoothed = np.convolve(loss_hist, np.ones(10)/10, mode='valid')
        ax.plot(episodes[9:], smoothed, color='blue', label='Smoothed Loss')
    ax.set_title('Training Loss (MSE)')
    ax.set_yscale('log') # Log scale is often better for seeing convergence
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Q-Value Magnitudes (Combined) ---
    ax = axes[0, 1]
    ax.plot(episodes, q_hist, label='Avg All States', alpha=0.8)
    ax.plot(episodes, q_terminal_hist, label='Avg Win/Loss States', alpha=0.8, linestyle='--')
    ax.set_title('Mean |Q| Predictions')
    ax.axhline(y=1.0, color='r', linestyle=':', label='Target (1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Exploration (Unique States) ---
    ax = axes[0, 2]
    ax.plot(episodes, states_hist, color='green', linewidth=2)
    ax.set_title('Unique States Explored')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Win Rate AND Terminal % ---
    ax = axes[1, 0]
    # Plot the batch terminal % as a light blue area/line
    ax.plot(episodes, terminal_pct_hist, color='cyan', alpha=0.3, label='Batch Terminal %')
    if len(terminal_pct_hist) > 10:
        smoothed_term = np.convolve(terminal_pct_hist, np.ones(20)/20, mode='valid')
        ax.plot(episodes[19:], smoothed_term, color='teal', alpha=0.6, label='Avg Terminal %')
    
    # Plot the actual Win Rate (Markers)
    ax.plot(eval_episodes, win_rate_hist, marker='o', color='gold', linewidth=2, label='Win Rate vs Random')
    
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)')
    ax.set_ylim(0, 1.1)
    ax.set_title('Win Rate & Terminal Signal')
    ax.legend(loc='lower right', fontsize='small')
    ax.grid(True, alpha=0.3)

    # --- Plot 5: NN Health (Combined Dead/Exploding) ---
    ax = axes[1, 1]
    ax.plot(episodes, dead_hist, label='Dead (<0.01)', color='black')
    ax.plot(episodes, exploding_hist, label='Exploding (>10)', color='red')
    ax.set_title('Neuron Health (Weight Counts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Gradients ---
    ax = axes[1, 2]
    ax.plot(episodes, grad_hist, color='purple')
    ax.set_title('Mean Absolute Gradient')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# *****************************************************************
# Train, show results
# *****************************************************************
train_dqn_agent(policy_net, optimizer, config)                                           # <--- REAL SELF PLAY
#train_on_synthetic_replay_buffer(policy_net, optimizer)    # <--- SYNTHETIC EXPERIENCE

plot_training_metrics(
    loss_history,
    q_magnitude_history,
    q_magnitude_history_win_loss,
    unique_states_history,
    win_rate_vs_rand_hist,
    dead_neuron_cnt_hist,
    exploding_neuron_cnt_hist,
    grad_history,
    terminal_pct_history,
    eval_freq=10)



# *****************************************************************
# Save the policy 
# *****************************************************************


# *****************************************************************
# Test the trained policy 
# *****************************************************************
# Test 1: Test on cases in the replay buffer
test_sample = [0, 1]         # Choose which item in the replay buffer to use
test_batch_size = len(test_sample)     # Choose a single item from the replay buffer to test

#examples = get_training_examples()
replay_buffer = generate_artificial_replay_buffer_for_training()
print( "Test Synthetic Replay Buffer Loaded!  Length: ", len( replay_buffer) )

replay_sample = replay_buffer.sample( test_batch_size, test_sample )
# print( initial_sample )

state = replay_sample[0]
print( "Sample state:")
print( state )
print( "Sample action: ", replay_sample[1] )
print( "Sample reward: ", replay_sample[2])
# Simplified test call
policy_net.eval() # CRITICAL for test consistency!
with torch.no_grad():
    q_values = policy_net(state) # The class handles the rest
#
#state_tensor = torch.FloatTensor(state).to('cpu')
#with torch.no_grad():
#    q_values = policy_net(state_tensor)
print( "Q Values on synthetic test state: ", q_values )

print( "Sample flipped state:")
state = state[:, [1, 0], :, :]
print( state )
policy_net.eval() # CRITICAL for test consistency!
with torch.no_grad():
    q_values = policy_net(state) # The class handles the rest
#state_tensor = torch.FloatTensor(state).to('cpu')
#with torch.no_grad():
#    q_values = policy_net(state_tensor)
print( "Q Values on REVERSED test state: ", q_values )

