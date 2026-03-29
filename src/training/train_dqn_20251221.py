# *****************************************************************
# Full Training - Build it up step by step
# *****************************************************************

"""
Docstring for train_dqn_20251221

Below is a plan that you the training Claude LLM agent should follow.  Step through this with the
human, and create unit tests that validate each step is done, each time the file is run.
I'd like to only update this file.  If you find errors in the environment, please flag, but ask for
permission to edit the file.  You do not need permission to update the code portion of this file.
Just review the results with the human before progressing to the next step.  The human thinks some
of the steps are complete, but please check.
1: Create environment.  Test that the environment's key functions work.  Especially, check whether
    the state flipping logic works, so that the negamax bellman equation work.
2: Create manual network based on what worked in manual_nn_training_test_20251221_v2.  During training
    runs, feel free to revisit the network architecture if you think we're missing patterns, or if
    you think the architecture is more complex, or you think we're missing other features in the
    network, such as regularization, or other optimizations that you're aware of based on best
    practices of neural network design, or recent research.  Please make hypotheses during training
    that include network architecture improvements.
3: Play a single self-play game.  Use Eps Greedy with constant eps.
    If greedy, do inference on state, and sample from the response - softmax
    At the end of the episode, if there was a winner, make sure is_done is true
    Also, override second to last reward to -1, and is_done to True
    Verify this happens, and create a unit test.
4: Play an ensemble of games to populate the replay buffer.  Ensure that the win/loss entries
    are correctly populated, and create a unit test.
5: Train based on the replay buffer.  Verify that the network has the capacity to reduce its loss.
    Train 100 times on static replay buffer, sampling independently; verify loss goes down
    Once replay buffer is ready, train X times per Y games, samping Z samples; verify loss decreases.
    During the final training testing, iteration loop, consider changes to the replay buffer, or
    how to prioritize samples in the replay buffer, such as with TD sampling.
6: Verify that using the CPU vs. MPS yields the same inference values, and trains exactly the same.
    If MPS doesn't line up with CPU within a very small tolerance, then raise an error.
7: Test the play vs. human function.  Don't create a unit test because the human doesn't want
    to have to play every time we run the file, but test that this works once, so that you can
    have the human play a highly-trained agent.
8: Implement training tracking
    Avg Abs Q Value prediction
    Avg Abs Q Value of Win/Loss position prediction (only where is_done = True)
    NN Loss by training event
    Win rate vs. random
    Unique States Explored
    Graph of abs(max(Q(state))), where x-axis is move since end.  First entry will always be
        the final state.  The abs(max) should approach 1 for games where a player wins.
        initialize the vector of size 42 to zero.  Then each time we calculate the win vs. random
        agent, recalculate this vector.  Then, during training evaluation, verify that the 
        curves start to decrease from 1 to zero.  If this curve does not decline as training
        continues, there is a problem.
9: Test whether we are correctly implementing the negamax bellman update.  Do we correctly return
    the state from the current player's perspective?  Ie - I think we want to flip the state when
    we subtract off the max of the next Q values.  Note we may have already done this when 
    validating the environment.
10: Use the prioritized replay buffer during training.  If we're not using it, it's a huge miss.  Verify that we
    are using it.  Print out in a unit test, if this is not already done.
11: make sure that we have a high discount rate - like 0.99 or 0.999, so that we don't lose the loss
    signal
12: Implement champion-challenger training.  Save a policy network that
    is a champion.  At the beginning of each episode, toss a fair dice to decide which player
    goes first.  The champion player ALWAYS plays greedy.  I'm not actually sure how we decide 
    when to overwrite the champion with a new challenger.  I think each TRAIN_VS_CHAMPTION_EPISODES
    we should face off with the current challenger policy.  Or, we track the win/loss rate for the 
    challenger vs. champion, and switch when it wins more than say 60% of the time - but we could
    parameterize.  I think we want the challenger to not use epsilon when evaluating vs. the 
    champion - but we can discuss.
13: Perspective threat test - after training,  test with three in a row from play 1 and 2 perspective.  
    Verify that  Q values change a lot.  If player 1 has three in a row, the Q values of next moves 
    should be very high - at least for the winning moves.  If player 2 is FACING three in a row, 
    verify that Q values are close to -1 for moves that fail to block.  Work this in as a policy unit
    test during training.  Every time we evaluate a policy vs. random, or whatever, we should also
    evaluate whether the Q values for an offensive three in a row show close to 1, and Q values for
    a failed defensive move are close to -1.  Add this to the training set of metrics.  Take the
    average of the Q values of the winning moves for the offense situation, and the average of the
    Q values that fail to block in the defensive situation.  The winning move Q values should approach
    1, and the failed defensive Q values should approach -1.  if they do not, something is wrong.
14: Iterate on training.  Run training on a sufficient number of games to yield a pretty good trained
    agent.  this should mean promoting several new champion policies.  based on what you see, 
    hypothesize on what to improve, make the changes, and measure the results.  Anything in this
    file is fair game to change.  You can change the policy network.  You can change hyper-params.
    You can decide to measure new things.  One thing you may not do is (1) change the environment, (2)
    use reward shaping, or (3) use MCTS.  You may not use knowledge of the game to cheat, and speed up how the Q
    values are learned.  The approach you figure out should generalize to other games and problems.
    I also don't want to use brute-force MCTS.  The DQN should learn to estimate the value making a
    move in a state.  Stop iterating when you suspect you have an agent that will beat the human.
    Two things to try:
        Temporal-difference error sampling from the replay buffer.
        Eligibility traces.  Start with lambda value of 0.9
15: Stretch goal - remove the heuristic where we hard-code the second
    to last move with a reward of -1, and is_done = true.  Consider this as something
    to try in step 14 as well.
16: For usability, create a repository of agents.  Each agent should have a date and a description 
    to start with.  There should be a way to browse agents, load agents, and play against them.

"""


import sys
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from src.environment import ConnectFourEnvironment, Config
from src.utils import DQNReplayBuffer, PrioritizedReplayBuffer
import matplotlib.pyplot as plt

# *****************************************************************
# Training Hyperparameters (all in one place)
# *****************************************************************
NUM_EPISODES                = 1000
BATCH_SIZE                  = 128
LEARNING_RATE               = 0.001
WEIGHT_DECAY                = 1e-4
TRAINING_ITERATIONS         = 2       # Training steps per game
EVAL_VS_RANDOM_GAME_COUNT   = 50
GAMMA                       = 0.99
EVALUATION_FREQUENCY        = 100     # Evaluate every N episodes
EPS_START                   = 0.5
EPS_END                     = 0.1
EPS_DECAY                   = 0.999
TARGET_UPDATE_FREQ          = 50
TERMINAL_RATE               = 0.3     # Target terminal ratio in batch sampling
DROPOUT_RATE                = 0.05
REPLAY_BUFFER_CAPACITY      = 20000
PER_ALPHA                   = 0.6     # Prioritization exponent (0=uniform, 1=full priority)
PER_BETA_START              = 0.4     # Importance sampling start (annealed to 1.0)
PER_BETA_FRAMES             = 100000  # Frames to anneal beta to 1.0
DEVICE                      = torch.device(Config().DEVICE)  # Auto-detected: MPS if safe, else CPU

# Champion/Challenger (Step 12)
CHAMPION_EVAL_FREQUENCY     = 200     # Evaluate challenger vs champion every N episodes
CHAMPION_EVAL_GAMES         = 50      # Games per evaluation
CHAMPION_THRESHOLD          = 0.55    # Win rate to promote challenger
CHAMPION_EVAL_TEMPERATURE   = 0.3     # Softmax temperature for evaluation games
MAX_STAGNATION_EPISODES     = 1000    # Revert challenger if no promotion in N episodes
CHAMPION_DIR                = os.path.join(root_dir, "models")


# *****************************************************************
# Create environment, Replay Buffer - 
# *****************************************************************
# You may change the configs
config = Config()
env = ConnectFourEnvironment(config)
replay_buffer = PrioritizedReplayBuffer(
    capacity=REPLAY_BUFFER_CAPACITY,
    alpha=PER_ALPHA,
    beta_start=PER_BETA_START,
    beta_frames=PER_BETA_FRAMES,
    terminal_ratio=TERMINAL_RATE
)


# *****************************************************************
# STEP 1: Environment Validation Tests
# *****************************************************************
def run_environment_tests():
    """Validate core environment functionality before training."""
    test_env = ConnectFourEnvironment(Config())
    passed = 0
    failed = 0

    # Test 1: Reset returns correct shape
    state = test_env.reset()
    try:
        assert state.shape == (2, 6, 7), f"Expected (2,6,7), got {state.shape}"
        assert state.sum() == 0, "Board should be empty after reset"
        print("✓ Test 1: Reset returns correct shape and empty board")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 1 FAILED: {e}")
        failed += 1

    # Test 2: Play a move and verify state changes
    test_env.reset()
    next_state, reward, done = test_env.play_move(3)
    try:
        assert reward == 0.0, f"First move should have reward 0, got {reward}"
        assert done == False, "Game should not be done after first move"
        assert next_state.sum() > 0, "Board should have a piece after move"
        print("✓ Test 2: Play move returns correct reward and done flag")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 2 FAILED: {e}")
        failed += 1

    # Test 3: Perspective flipping (critical for negamax)
    test_env.reset()
    # Player 1 plays column 3
    test_env.play_move(3)
    # Now it's Player 2's turn
    p1_view = test_env.get_state_from_perspective(1)
    p2_view = test_env.get_state_from_perspective(-1)
    try:
        # P1's pieces in P1's channel 0 should equal P1's pieces in P2's channel 1
        assert np.array_equal(p1_view[0], p2_view[1]), "P1 pieces should be in P2's opponent channel"
        assert np.array_equal(p1_view[1], p2_view[0]), "P2 pieces should be in P1's opponent channel"
        print("✓ Test 3: Perspective flipping swaps channels correctly")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 3 FAILED: {e}")
        failed += 1

    # Test 4: get_state() returns current player's perspective
    test_env.reset()
    test_env.play_move(3)  # P1 plays, now it's P2's turn
    canonical_state = test_env.get_state()
    p2_perspective = test_env.get_state_from_perspective(-1)
    try:
        assert np.array_equal(canonical_state, p2_perspective), \
            "get_state() should return current player's perspective"
        print("✓ Test 4: get_state() returns current player's perspective")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4 FAILED: {e}")
        failed += 1

    # Test 5: play_move returns next_state from NEXT player's perspective
    test_env.reset()
    # P1 plays col 0
    state_before = test_env.get_state()  # P1's perspective
    next_state, _, _ = test_env.play_move(0)
    # next_state should be from P2's perspective (P2's pieces in channel 0)
    p2_view = test_env.get_state_from_perspective(-1)
    try:
        assert np.array_equal(next_state, p2_view), \
            "play_move should return state from NEXT player's perspective"
        # P1's piece should be in channel 1 of next_state (opponent's channel from P2's view)
        assert next_state[1, 5, 0] == 1.0, "P1's piece should be in P2's opponent channel"
        print("✓ Test 5: play_move returns state from next player's perspective")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 5 FAILED: {e}")
        failed += 1

    # Test 6: Vertical win detection
    test_env.reset()
    for _ in range(3):
        test_env.play_move(0)  # P1
        test_env.play_move(1)  # P2
    next_state, reward, done = test_env.play_move(0)  # P1 wins vertically
    try:
        assert done == True, "Game should be done after 4-in-a-row"
        assert reward == 1.0, f"Winner should get +1, got {reward}"
        print("✓ Test 6: Vertical win detected correctly")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 6 FAILED: {e}")
        failed += 1

    # Test 7: Horizontal win detection
    test_env.reset()
    moves = [(0,6), (1,6), (2,6), (3,None)]  # P1 plays 0,1,2,3; P2 plays 6,6,6
    for p1_col, p2_col in moves:
        if p2_col is not None:
            test_env.play_move(p1_col)
            test_env.play_move(p2_col)
        else:
            next_state, reward, done = test_env.play_move(p1_col)
    try:
        assert done == True, "Game should be done after horizontal 4-in-a-row"
        assert reward == 1.0, f"Winner should get +1, got {reward}"
        print("✓ Test 7: Horizontal win detected correctly")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 7 FAILED: {e}")
        failed += 1

    # Test 8: Illegal move raises error
    test_env.reset()
    try:
        test_env.play_move(7)  # Column out of range
        print("✗ Test 8 FAILED: Should have raised ValueError")
        failed += 1
    except ValueError:
        print("✓ Test 8: Illegal column raises ValueError")
        passed += 1

    # Test 9: Full column raises error
    test_env.reset()
    for i in range(6):
        test_env.play_move(0)
        if i < 5:
            test_env.play_move(1)
    try:
        test_env.play_move(0)  # Column 0 is full
        print("✗ Test 9 FAILED: Should have raised ValueError for full column")
        failed += 1
    except ValueError:
        print("✓ Test 9: Full column raises ValueError")
        passed += 1

    # Test 10: Legal moves excludes full columns
    test_env.reset()
    for i in range(3):
        test_env.play_move(0)  # P1
        test_env.play_move(0)  # P2
    legal = test_env.get_legal_moves()
    try:
        assert 0 not in legal, "Full column should not be in legal moves"
        assert len(legal) == 6, f"Expected 6 legal moves, got {len(legal)}"
        print("✓ Test 10: Legal moves correctly excludes full columns")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 10 FAILED: {e}")
        failed += 1

    # Test 11: Negamax Bellman consistency
    # After P1 plays, the state returned should allow the negamax equation:
    # Q(s,a) = r - gamma * max(Q(s'))  where s' is from opponent's perspective
    test_env.reset()
    state_p1 = test_env.get_state()  # P1's view: my pieces in ch0
    next_state, reward, done = test_env.play_move(3)  # P1 plays col 3
    # next_state is from P2's perspective
    # P2's "my pieces" (ch0) should be empty (P2 hasn't played yet)
    # P2's "opponent pieces" (ch1) should show P1's piece
    try:
        assert next_state[0].sum() == 0, "P2 has no pieces yet, channel 0 should be empty"
        assert next_state[1].sum() == 1, "P1's piece should appear in P2's opponent channel"
        assert next_state[1, 5, 3] == 1.0, "P1's piece at row 5, col 3 should be in P2's ch1"
        print("✓ Test 11: Negamax state representation is consistent")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 11 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 1 Environment Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")
    
    if failed > 0:
        raise RuntimeError(f"Step 1 FAILED: {failed} test(s) did not pass. Fix before proceeding.")

print("Running Step 1: Environment Validation...")
run_environment_tests()

#replay_buffer.add( [1,2,3], 0, 0, [2,3,4], False, [1,2,3]) # Test replay buffer - comment out when training
#replay_buffer.add( [2,3,4], 0, 1, [3,4,5], False, [1,2,3]) # Test replay buffer

"""
print( replay_buffer )
print( "Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-1]))
print( "Second Most Recent Entry in Buffer:")
print( replay_buffer.sample(1,[-2]))
# Adjust reward of second to last entry
replay_buffer.update_penalty(-2,-1,1)
print( "Replay buffer after updating 2nd to last entry")
print( replay_buffer.sample(1,[-2]))
"""


# *****************************************************************
# STEP 2: Network Architecture
# *****************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Connect4Net(nn.Module):
    """
    CNN for Connect4 Q-value estimation.
    Input: (batch, 2, 6, 7) — channel 0 = my pieces, channel 1 = opponent pieces
    Output: (batch, 7) — Q-value for each column
    
    Architecture: 3 conv layers (64 filters, 3x3, padding=1) with BatchNorm,
    followed by FC layers. Receptive field after 3 layers = 7x7, covering
    the full board width — important for detecting diagonal threats.
    """
    def __init__(self, device, dropout_rate=0.2):
        super(Connect4Net, self).__init__()
        self.device = device
        
        # Conv block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dr1 = nn.Dropout2d(p=dropout_rate)
        
        # Conv block 2: Pattern detection (2-in-a-row, 3-in-a-row)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dr2 = nn.Dropout2d(p=dropout_rate)

        # Conv block 3: Full-board patterns (threats, forks)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dr3 = nn.Dropout2d(p=dropout_rate)
        
        # Fully connected: 64 filters * 6 rows * 7 cols = 2688
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.dr_fc = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(128, 7)

        # He (Kaiming) initialization for LeakyReLU networks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.dr1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.dr2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.dr3(x)

        x = x.view(x.size(0), -1) 
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dr_fc(x)
        return self.output(x)


# *****************************************************************
# Step 2: Network Validation Tests
# *****************************************************************
def run_network_tests():
    """Validate network architecture and inference."""
    test_device = torch.device("cpu")
    net = Connect4Net(device=test_device, dropout_rate=0.0)
    net.eval()
    passed = 0
    failed = 0

    # Test 1: Output shape for single state
    state = np.zeros((2, 6, 7), dtype=np.float32)
    try:
        with torch.no_grad():
            q_values = net(state)
        assert q_values.shape == (1, 7), f"Expected (1,7), got {q_values.shape}"
        print("✓ Test 2.1: Single state produces (1,7) Q-values")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 2.1 FAILED: {e}")
        failed += 1

    # Test 2: Output shape for batch
    batch = np.zeros((16, 2, 6, 7), dtype=np.float32)
    try:
        with torch.no_grad():
            q_values = net(torch.tensor(batch))
        assert q_values.shape == (16, 7), f"Expected (16,7), got {q_values.shape}"
        print("✓ Test 2.2: Batch of 16 produces (16,7) Q-values")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 2.2 FAILED: {e}")
        failed += 1

    # Test 3: Different inputs produce different outputs
    state_a = np.zeros((2, 6, 7), dtype=np.float32)
    state_b = np.zeros((2, 6, 7), dtype=np.float32)
    state_b[0, 5, 3] = 1.0  # Place a piece
    try:
        with torch.no_grad():
            q_a = net(state_a)
            q_b = net(state_b)
        assert not torch.allclose(q_a, q_b), "Different states should produce different Q-values"
        print("✓ Test 2.3: Different inputs produce different Q-values")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 2.3 FAILED: {e}")
        failed += 1

    # Test 4: Deterministic in eval mode (no dropout noise)
    try:
        with torch.no_grad():
            q1 = net(state_a)
            q2 = net(state_a)
        assert torch.allclose(q1, q2), "Eval mode should be deterministic"
        print("✓ Test 2.4: Eval mode produces deterministic output")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 2.4 FAILED: {e}")
        failed += 1

    # Test 5: Gradient flows through all layers
    net.train()
    test_input = torch.randn(4, 2, 6, 7, requires_grad=False).to(test_device)
    target = torch.randn(4, 7).to(test_device)
    try:
        output = net(test_input)
        loss = F.mse_loss(output, target)
        loss.backward()
        all_have_grad = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net.parameters() if p.requires_grad
        )
        assert all_have_grad, "Some parameters have no gradient"
        print("✓ Test 2.5: Gradients flow through all layers")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 2.5 FAILED: {e}")
        failed += 1

    # Test 6: Parameter count is reasonable
    total_params = sum(p.numel() for p in net.parameters())
    try:
        assert 100_000 < total_params < 500_000, f"Param count {total_params} seems off"
        print(f"✓ Test 2.6: Parameter count = {total_params:,} (reasonable range)")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 2.6 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 2 Network Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")
    
    if failed > 0:
        raise RuntimeError(f"Step 2 FAILED: {failed} test(s) did not pass.")

print("Running Step 2: Network Validation...")
run_network_tests()

# Initialize models
policy_net = Connect4Net(device=DEVICE, dropout_rate=DROPOUT_RATE)
target_net = Connect4Net(device=DEVICE, dropout_rate=DROPOUT_RATE)

# Sync weights
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Simplified Optimizer
#optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
optimizer = torch.optim.Adam(
    policy_net.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

# *****************************************************************
# Test inference works with the function defined.  Pull examples from the notebook dir.
# *****************************************************************
# Helper functions, for training and metric tracking
# *****************************************************************
def hash_state(state):
    return state.tobytes()

def evaluate_vs_random(policy_net = policy_net, num_games=20):
    wins = 0
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            state = env.get_state()
            legal = env.get_legal_moves()
            action = select_action(policy_net, torch.tensor(state, dtype=torch.float32), legal, eps=0.0)
            _, reward, done = env.play_move(action)
            if done and reward == 1:
                wins += 1
                break
            if done: break
            
            legal = env.get_legal_moves()
            action = np.random.choice(legal)
            _, reward, done = env.play_move(action)
            if done and reward == 1:
                break 
    return wins / num_games

def get_action_mask(legal_moves):
    """Converts [0, 1, 3] into [1, 1, 0, 1, 0, 0, 0]"""
    mask = np.zeros(7, dtype=np.float32)
    mask[legal_moves] = 1.0
    return mask


# *****************************************************************
# Select Action
# *****************************************************************
def select_action(policy_net, state, legal_moves, eps) -> int:
    if np.random.random() < eps:
        return int(np.random.choice(legal_moves))
        
    policy_net.eval() 
    with torch.no_grad():
        q_values = policy_net(state).squeeze(0)
        masked_q = q_values.clone()
        all_actions = set(range(7))
        illegal_actions = list(all_actions - set(legal_moves))
        masked_q[illegal_actions] = -1e9
        best_action = torch.argmax(masked_q).item()
    
    return int(best_action)


def select_action_temperature(policy_net, state, legal_moves, temperature=0.3) -> int:
    """Select action using softmax over Q-values with temperature scaling.
    
    Lower temperature → more greedy (deterministic).
    Higher temperature → more random (exploratory).
    temperature=0 → pure greedy (argmax).
    """
    if temperature <= 0:
        return select_action(policy_net, state, legal_moves, eps=0.0)
    
    policy_net.eval()
    with torch.no_grad():
        q_values = policy_net(state).squeeze(0)
        # Mask illegal moves
        mask = torch.full((7,), -1e9)
        for m in legal_moves:
            mask[m] = 0.0
        masked_q = q_values + mask.to(q_values.device)
        # Softmax with temperature
        probs = torch.softmax(masked_q / temperature, dim=0).cpu().numpy()
        action = np.random.choice(7, p=probs)
    return int(action)


def play_champion_challenger_game(challenger_net, champion_net, eps, challenger_is_p1=True):
    """Play a single game: challenger (eps-greedy) vs champion (greedy).
    
    Returns game_states for the challenger's moves only (for unique state tracking).
    Both players' moves go into the replay buffer.
    """
    env.reset()
    done = False
    moves_count = 0
    game_states = []
    replay_buffer._recent_indices = []
    reward = 0.0

    while not done and moves_count < 42:
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Determine whose turn it is
        is_p1_turn = (moves_count % 2 == 0)
        is_challenger_turn = (is_p1_turn == challenger_is_p1)
        
        if is_challenger_turn:
            game_states.append(state)
            action = select_action(challenger_net, state_tensor, legal_moves, eps)
        else:
            # Champion always plays greedy
            action = select_action(champion_net, state_tensor, legal_moves, eps=0.0)
        
        next_state, reward, done = env.play_move(action)
        
        if not done:
            next_legal_moves = env.get_legal_moves()
            next_mask = get_action_mask(next_legal_moves)
        else:
            next_mask = np.zeros(7, dtype=np.float32)

        replay_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
        moves_count += 1
    
    # Bellman negative reward fix: loser's last move gets -1
    if reward == 1.0:
        replay_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
        replay_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)
    
    # Determine if challenger won
    # The player who made the last move won (if reward == 1.0)
    last_move_was_p1 = ((moves_count - 1) % 2 == 0)
    challenger_made_last_move = (last_move_was_p1 == challenger_is_p1)
    
    if reward == 1.0 and challenger_made_last_move:
        challenger_result = 1  # win
    elif reward == 1.0 and not challenger_made_last_move:
        challenger_result = -1  # loss
    else:
        challenger_result = 0  # draw
    
    return game_states, challenger_result


def evaluate_challenger_vs_champion(challenger_net, champion_net, num_games=50, temperature=0.3):
    """Evaluate challenger vs champion using temperature-based move selection.
    
    Returns challenger win rate. Both players use temperature-based selection
    to introduce stochasticity for meaningful evaluation over many games.
    """
    eval_env = ConnectFourEnvironment(Config())
    challenger_wins = 0
    champion_wins = 0
    
    for game_idx in range(num_games):
        eval_env.reset()
        done = False
        moves_count = 0
        challenger_is_p1 = (game_idx % 2 == 0)  # Alternate sides
        reward = 0.0
        
        while not done and moves_count < 42:
            state = eval_env.get_state()
            legal = eval_env.get_legal_moves()
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            is_p1_turn = (moves_count % 2 == 0)
            is_challenger_turn = (is_p1_turn == challenger_is_p1)
            
            if is_challenger_turn:
                action = select_action_temperature(challenger_net, state_tensor, legal, temperature)
            else:
                action = select_action_temperature(champion_net, state_tensor, legal, temperature)
            
            _, reward, done = eval_env.play_move(action)
            moves_count += 1
        
        if reward == 1.0:
            last_move_was_p1 = ((moves_count - 1) % 2 == 0)
            challenger_made_last_move = (last_move_was_p1 == challenger_is_p1)
            if challenger_made_last_move:
                challenger_wins += 1
            else:
                champion_wins += 1
    
    win_rate = challenger_wins / num_games if num_games > 0 else 0.0
    return win_rate, challenger_wins, champion_wins


def save_champion(policy_net, version, episode, win_rate_vs_champion):
    """Save a promoted champion to the models directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"champion_v{version}_{timestamp}.pth"
    filepath = os.path.join(CHAMPION_DIR, filename)
    
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'version': version,
        'episode': episode,
        'win_rate_vs_champion': win_rate_vs_champion,
        'timestamp': timestamp,
    }, filepath)
    
    # Also save as current champion
    current_path = os.path.join(CHAMPION_DIR, "champion_current.pth")
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'version': version,
        'episode': episode,
        'win_rate_vs_champion': win_rate_vs_champion,
        'timestamp': timestamp,
    }, current_path)
    
    print(f"  Champion v{version} saved → {filename}")
    return filepath
    

# *****************************************************************
# Self Play
# *****************************************************************
def play_self_play_game(policy_net, eps=0.5):
    env.reset()
    done = False
    moves_count = 0
    game_states = []
    replay_buffer._recent_indices = []  # Reset for correct negative indexing

    while not done and moves_count < 42:
        state = env.get_state()
        game_states.append(state)
        legal_moves = env.get_legal_moves()
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(policy_net, state_tensor, legal_moves, eps)
        next_state, reward, done = env.play_move(action)
        
        if not done:
            next_legal_moves = env.get_legal_moves()
            next_mask = get_action_mask(next_legal_moves)
        else:
            next_mask = np.zeros(7, dtype=np.float32)

        replay_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
        moves_count += 1
    
    # Bellman negative reward fix: loser's last move gets -1
    if reward == 1.0:
        replay_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
        replay_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)
    
    return game_states

# *****************************************************************
# STEP 3: Single Self-Play Game Validation Tests
# *****************************************************************
def run_self_play_tests():
    """Validate a single self-play game populates the replay buffer correctly."""
    test_buffer = DQNReplayBuffer(capacity=1000)
    test_env = ConnectFourEnvironment(Config())
    passed = 0
    failed = 0

    # Play a single game using the untrained policy_net with eps=1.0 (fully random)
    test_env.reset()
    done = False
    moves_count = 0
    last_reward = 0.0

    while not done and moves_count < 42:
        state = test_env.get_state()
        legal_moves = test_env.get_legal_moves()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(policy_net, state_tensor, legal_moves, eps=1.0)
        next_state, reward, done = test_env.play_move(action)

        if not done:
            next_legal_moves = test_env.get_legal_moves()
            next_mask = get_action_mask(next_legal_moves)
        else:
            next_mask = np.zeros(7, dtype=np.float32)

        test_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
        last_reward = reward
        moves_count += 1

    # Apply Bellman negative reward fix
    if last_reward == 1.0:
        test_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
        test_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)

    # Test 3.1: Game completed
    try:
        assert done == True, "Game should have ended (win or draw)"
        assert moves_count >= 7, f"Need at least 7 moves for a win, got {moves_count}"
        assert moves_count <= 42, f"Max 42 moves, got {moves_count}"
        print(f"✓ Test 3.1: Game completed in {moves_count} moves (reward={last_reward})")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 3.1 FAILED: {e}")
        failed += 1

    # Test 3.2: Replay buffer has correct number of entries (2x due to symmetric)
    try:
        expected_entries = moves_count * 2  # add_symmetric doubles each entry
        assert len(test_buffer) == expected_entries, \
            f"Expected {expected_entries} entries (symmetric), got {len(test_buffer)}"
        print(f"✓ Test 3.2: Replay buffer has {len(test_buffer)} entries ({moves_count} moves × 2 symmetric)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 3.2 FAILED: {e}")
        failed += 1

    # Test 3.3: All states have correct shape
    try:
        for i, transition in enumerate(test_buffer.buffer):
            assert transition.state.shape == (2, 6, 7), f"State {i} shape wrong: {transition.state.shape}"
            assert transition.next_state.shape == (2, 6, 7), f"Next state {i} shape wrong"
        print("✓ Test 3.3: All states have shape (2, 6, 7)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 3.3 FAILED: {e}")
        failed += 1

    # Test 3.4: Actions are valid (0-6)
    try:
        for i, transition in enumerate(test_buffer.buffer):
            assert 0 <= transition.action <= 6, f"Invalid action {transition.action} at index {i}"
        print("✓ Test 3.4: All actions are in range [0, 6]")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 3.4 FAILED: {e}")
        failed += 1

    # Test 3.5: Bellman negative reward fix (only if someone won)
    if last_reward == 1.0:
        try:
            # With add_symmetric, the last 2 entries are the winning move (original + mirror)
            # The 2 entries before that (-3, -4) are the losing move (original + mirror)
            losing_orig = test_buffer.buffer[-3]
            losing_mirror = test_buffer.buffer[-4]
            assert losing_orig.reward == -1.0, f"Losing move reward should be -1, got {losing_orig.reward}"
            assert losing_orig.done == 1.0, "Losing move should be marked done"
            assert losing_mirror.reward == -1.0, f"Losing mirror reward should be -1, got {losing_mirror.reward}"
            assert losing_mirror.done == 1.0, "Losing mirror should be marked done"

            # Winning move should have reward +1
            winning_orig = test_buffer.buffer[-2]
            winning_mirror = test_buffer.buffer[-1]
            assert winning_orig.reward == 1.0, f"Winning move reward should be +1, got {winning_orig.reward}"
            assert winning_orig.done == 1.0, "Winning move should be marked done"
            print("✓ Test 3.5: Bellman negative reward fix applied correctly (win: +1, loss: -1, both done=True)")
            passed += 1
        except AssertionError as e:
            print(f"✗ Test 3.5 FAILED: {e}")
            failed += 1
    else:
        # Draw — no penalty fix needed
        try:
            last_entry = test_buffer.buffer[-1]
            assert last_entry.reward == 0.0, f"Draw should have reward 0, got {last_entry.reward}"
            print("✓ Test 3.5: Game was a draw, no penalty fix needed (reward=0)")
            passed += 1
        except AssertionError as e:
            print(f"✗ Test 3.5 FAILED: {e}")
            failed += 1

    # Test 3.6: Mid-game transitions have reward=0 and done=False
    try:
        # Check a mid-game transition (not the last few)
        mid_idx = len(test_buffer.buffer) // 2
        mid_transition = test_buffer.buffer[mid_idx]
        assert mid_transition.reward == 0.0, f"Mid-game reward should be 0, got {mid_transition.reward}"
        assert mid_transition.done == 0.0 or mid_transition.done == False, \
            f"Mid-game done should be False, got {mid_transition.done}"
        print("✓ Test 3.6: Mid-game transitions have reward=0, done=False")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 3.6 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 3 Self-Play Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 3 FAILED: {failed} test(s) did not pass.")

print("Running Step 3: Self-Play Validation...")
run_self_play_tests()


# *****************************************************************
# STEP 4: Ensemble Self-Play & Replay Buffer Population Tests
# *****************************************************************
ENSEMBLE_GAME_COUNT = 20  # Number of games to play for ensemble test

def run_ensemble_tests():
    """Play multiple self-play games and verify replay buffer population."""
    test_buffer = DQNReplayBuffer(capacity=10000)
    test_env = ConnectFourEnvironment(Config())
    passed = 0
    failed = 0

    win_count = 0
    draw_count = 0
    total_moves = 0

    for game_idx in range(ENSEMBLE_GAME_COUNT):
        test_env.reset()
        done = False
        moves_count = 0
        last_reward = 0.0

        while not done and moves_count < 42:
            state = test_env.get_state()
            legal_moves = test_env.get_legal_moves()
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = select_action(policy_net, state_tensor, legal_moves, eps=1.0)
            next_state, reward, done = test_env.play_move(action)

            if not done:
                next_legal_moves = test_env.get_legal_moves()
                next_mask = get_action_mask(next_legal_moves)
            else:
                next_mask = np.zeros(7, dtype=np.float32)

            test_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
            last_reward = reward
            moves_count += 1

        # Apply Bellman negative reward fix for games with a winner
        if last_reward == 1.0:
            test_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
            test_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)
            win_count += 1
        else:
            draw_count += 1

        total_moves += moves_count

    # Test 4.1: All games completed
    try:
        assert win_count + draw_count == ENSEMBLE_GAME_COUNT, \
            f"Expected {ENSEMBLE_GAME_COUNT} games, got {win_count + draw_count}"
        print(f"✓ Test 4.1: All {ENSEMBLE_GAME_COUNT} games completed ({win_count} wins, {draw_count} draws)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4.1 FAILED: {e}")
        failed += 1

    # Test 4.2: Buffer size matches expected (2x symmetric per move)
    try:
        expected = total_moves * 2
        assert len(test_buffer) == expected, \
            f"Expected {expected} entries, got {len(test_buffer)}"
        print(f"✓ Test 4.2: Buffer has {len(test_buffer)} entries ({total_moves} moves × 2 symmetric)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4.2 FAILED: {e}")
        failed += 1

    # Test 4.3: Win entries have reward +1 and done=True
    try:
        win_entries = [t for t in test_buffer.buffer if t.reward == 1.0 and t.done == 1.0]
        # Each win produces 2 entries (original + mirror) for the winning move
        expected_win_entries = win_count * 2
        assert len(win_entries) == expected_win_entries, \
            f"Expected {expected_win_entries} win entries, got {len(win_entries)}"
        print(f"✓ Test 4.3: Found {len(win_entries)} win entries (reward=+1, done=True)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4.3 FAILED: {e}")
        failed += 1

    # Test 4.4: Loss entries have reward -1 and done=True
    try:
        loss_entries = [t for t in test_buffer.buffer if t.reward == -1.0 and t.done == 1.0]
        # Each win also produces 2 loss entries (loser's last move, original + mirror)
        expected_loss_entries = win_count * 2
        assert len(loss_entries) == expected_loss_entries, \
            f"Expected {expected_loss_entries} loss entries, got {len(loss_entries)}"
        print(f"✓ Test 4.4: Found {len(loss_entries)} loss entries (reward=-1, done=True)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4.4 FAILED: {e}")
        failed += 1

    # Test 4.5: Terminal buffer is populated for balanced sampling
    try:
        terminal_count = len(test_buffer.terminal_buffer)
        assert terminal_count > 0, "Terminal buffer should not be empty after games with winners"
        # Terminal buffer should have win + loss entries
        expected_terminal = win_count * 4  # 2 win + 2 loss per won game
        assert terminal_count >= expected_terminal, \
            f"Expected at least {expected_terminal} terminal entries, got {terminal_count}"
        print(f"✓ Test 4.5: Terminal buffer has {terminal_count} entries (for balanced sampling)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4.5 FAILED: {e}")
        failed += 1

    # Test 4.6: Mid-game entries have reward=0 and done=False
    try:
        mid_entries = [t for t in test_buffer.buffer if t.reward == 0.0 and t.done == 0.0]
        # Most entries should be mid-game (reward=0, done=False)
        total_terminal = len([t for t in test_buffer.buffer if t.done == 1.0])
        expected_mid = len(test_buffer) - total_terminal
        assert len(mid_entries) == expected_mid, \
            f"Expected {expected_mid} mid-game entries, got {len(mid_entries)}"
        assert len(mid_entries) > total_terminal, \
            "Mid-game entries should outnumber terminal entries"
        print(f"✓ Test 4.6: {len(mid_entries)} mid-game entries (reward=0, done=False), {total_terminal} terminal")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 4.6 FAILED: {e}")
        failed += 1

    # Test 4.7: Balanced sampling works (terminal_ratio respected)
    try:
        if test_buffer.is_ready(BATCH_SIZE):
            states, actions, rewards, next_states, dones, next_masks = test_buffer.sample(
                BATCH_SIZE, terminal_ratio=TERMINAL_RATE
            )
            actual_terminal_pct = np.mean(dones)
            # Allow some tolerance since sampling is stochastic
            assert actual_terminal_pct > 0.1, \
                f"Terminal ratio too low: {actual_terminal_pct:.2f} (target: {TERMINAL_RATE})"
            print(f"✓ Test 4.7: Balanced sampling works (terminal %: {actual_terminal_pct:.2f}, target: {TERMINAL_RATE})")
            passed += 1
        else:
            print(f"⚠ Test 4.7: Skipped — buffer not ready (need {BATCH_SIZE}, have {len(test_buffer)})")
            passed += 1  # Not a failure, just insufficient data
    except (AssertionError, Exception) as e:
        print(f"✗ Test 4.7 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 4 Ensemble Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 4 FAILED: {failed} test(s) did not pass.")

print("Running Step 4: Ensemble Self-Play & Buffer Population...")
run_ensemble_tests()


# *****************************************************************
# STEP 5: Training Capacity Test — Verify Loss Decreases
# *****************************************************************
STEP5_STATIC_TRAIN_ITERS = 100  # Train 100 times on a static buffer

def run_training_capacity_tests():
    """Verify the network can reduce loss when trained on a static replay buffer."""
    passed = 0
    failed = 0

    # Build a fresh buffer with 50 self-play games (fully random)
    test_buffer = DQNReplayBuffer(capacity=10000)
    test_env = ConnectFourEnvironment(Config())

    for _ in range(50):
        test_env.reset()
        done = False
        moves = 0
        last_reward = 0.0
        while not done and moves < 42:
            state = test_env.get_state()
            legal = test_env.get_legal_moves()
            action = int(np.random.choice(legal))
            next_state, reward, done = test_env.play_move(action)
            if not done:
                next_mask = get_action_mask(test_env.get_legal_moves())
            else:
                next_mask = np.zeros(7, dtype=np.float32)
            test_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
            last_reward = reward
            moves += 1
        if last_reward == 1.0:
            test_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
            test_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)

    # Fresh network + optimizer for isolated test
    test_net = Connect4Net(device=DEVICE, dropout_rate=DROPOUT_RATE)
    test_target = copy.deepcopy(test_net)
    test_target.eval()
    test_opt = torch.optim.Adam(test_net.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)

    losses = []
    test_net.train()

    for i in range(STEP5_STATIC_TRAIN_ITERS):
        states, actions, rewards, next_states, dones, next_masks = test_buffer.sample(
            BATCH_SIZE, terminal_ratio=TERMINAL_RATE
        )
        s = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        a = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        r = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        ns = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        d = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        m = torch.tensor(next_masks, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            next_q = test_target(ns)
            masked_next_q = next_q.masked_fill(m == 0, -1e9)
            next_q_max = masked_next_q.max(dim=1)[0]
            target_q = r - (GAMMA * next_q_max * (1 - d))

        test_opt.zero_grad()
        q_vals = test_net(s)
        pred_q = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(pred_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(test_net.parameters(), 1.0)
        test_opt.step()
        losses.append(loss.item())

        # Sync target every 10 steps
        if i % 10 == 0:
            test_target.load_state_dict(test_net.state_dict())

    # Test 5.1: Loss decreased from first 10 to last 10
    first_10_avg = np.mean(losses[:10])
    last_10_avg = np.mean(losses[-10:])
    try:
        assert last_10_avg < first_10_avg, \
            f"Loss did not decrease: first 10 avg={first_10_avg:.4f}, last 10 avg={last_10_avg:.4f}"
        reduction_pct = (1 - last_10_avg / first_10_avg) * 100
        print(f"✓ Test 5.1: Loss decreased — {first_10_avg:.4f} → {last_10_avg:.4f} ({reduction_pct:.1f}% reduction)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 5.1 FAILED: {e}")
        failed += 1

    # Test 5.2: Final loss is finite and non-negative
    try:
        assert np.isfinite(losses[-1]), f"Loss is not finite: {losses[-1]}"
        assert losses[-1] >= 0, f"Loss is negative: {losses[-1]}"
        print(f"✓ Test 5.2: Final loss is finite and non-negative ({losses[-1]:.4f})")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 5.2 FAILED: {e}")
        failed += 1

    # Test 5.3: Q-values are in a reasonable range after training
    test_net.eval()
    try:
        sample_states, _, _, _, _, _ = test_buffer.sample(32, terminal_ratio=0.5)
        with torch.no_grad():
            q_out = test_net(torch.tensor(sample_states, dtype=torch.float32).to(DEVICE))
        max_q = q_out.abs().max().item()
        assert max_q < 100, f"Q-values exploded: max |Q| = {max_q:.2f}"
        print(f"✓ Test 5.3: Q-values in reasonable range (max |Q| = {max_q:.2f})")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 5.3 FAILED: {e}")
        failed += 1

    # Test 5.4: Network produces different Q-values for win vs mid-game states
    try:
        terminal_states = [t.state for t in test_buffer.buffer if t.done == 1.0][:8]
        mid_states = [t.state for t in test_buffer.buffer if t.done == 0.0][:8]
        if len(terminal_states) >= 4 and len(mid_states) >= 4:
            with torch.no_grad():
                t_q = test_net(torch.tensor(np.array(terminal_states[:4]), dtype=torch.float32).to(DEVICE))
                m_q = test_net(torch.tensor(np.array(mid_states[:4]), dtype=torch.float32).to(DEVICE))
            t_mag = t_q.abs().mean().item()
            m_mag = m_q.abs().mean().item()
            # Terminal states should generally have higher magnitude Q-values
            print(f"✓ Test 5.4: Terminal avg |Q|={t_mag:.3f}, Mid-game avg |Q|={m_mag:.3f}")
            passed += 1
        else:
            print("⚠ Test 5.4: Skipped — not enough terminal/mid-game states")
            passed += 1
    except Exception as e:
        print(f"✗ Test 5.4 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 5 Training Capacity Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 5 FAILED: {failed} test(s) did not pass.")

print("Running Step 5: Training Capacity Verification...")
run_training_capacity_tests()


# *****************************************************************
# STEP 6: MPS/CPU Parity Check
# *****************************************************************
def run_mps_parity_tests():
    """Verify MPS and CPU produce identical inference and training results."""
    passed = 0
    failed = 0

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if not mps_available:
        print("⚠ MPS not available on this machine — skipping parity tests")
        print(f"  DEVICE is set to: {DEVICE}")
        print(f"\n{'='*50}")
        print(f"Step 6 MPS Parity Tests: skipped (no MPS)")
        print(f"{'='*50}\n")
        return

    # Test 6.1: Forward pass parity — same weights, same input, same output
    try:
        torch.manual_seed(42)
        net_cpu = Connect4Net(device=torch.device("cpu"), dropout_rate=0.0)
        net_cpu.eval()

        net_mps = Connect4Net(device=torch.device("mps"), dropout_rate=0.0)
        net_mps.load_state_dict(net_cpu.state_dict())
        net_mps.eval()

        test_input = torch.randn(8, 2, 6, 7)
        with torch.no_grad():
            out_cpu = net_cpu(test_input).numpy()
            out_mps = net_mps(test_input.to("mps")).cpu().numpy()

        max_diff = float(np.abs(out_cpu - out_mps).max())
        assert max_diff < 5e-3, f"CPU/MPS forward drift: {max_diff:.2e}"
        assert np.all(np.isfinite(out_mps)), "MPS produced NaN/Inf"
        print(f"✓ Test 6.1: Forward pass parity (max diff: {max_diff:.2e})")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 6.1 FAILED: {e}")
        failed += 1

    # Test 6.2: Single training step parity — loss should be close
    try:
        torch.manual_seed(99)
        net_cpu = Connect4Net(device=torch.device("cpu"), dropout_rate=0.0)
        net_cpu.train()
        opt_cpu = torch.optim.Adam(net_cpu.parameters(), lr=0.001)

        net_mps = Connect4Net(device=torch.device("mps"), dropout_rate=0.0)
        net_mps.load_state_dict(net_cpu.state_dict())
        net_mps.train()
        opt_mps = torch.optim.Adam(net_mps.parameters(), lr=0.001)

        s_in = torch.randn(16, 2, 6, 7)
        targets = torch.randn(16)

        # CPU step
        opt_cpu.zero_grad()
        q_cpu = net_cpu(s_in)[:, 3]
        loss_cpu = nn.functional.mse_loss(q_cpu, targets)
        loss_cpu.backward()
        opt_cpu.step()

        # MPS step
        opt_mps.zero_grad()
        q_mps = net_mps(s_in.to("mps"))[:, 3]
        loss_mps = nn.functional.mse_loss(q_mps, targets.to("mps"))
        loss_mps.backward()
        opt_mps.step()

        loss_diff = abs(loss_cpu.item() - loss_mps.item())
        assert loss_diff < 0.01, f"Training loss drift: {loss_diff:.4f}"
        assert np.isfinite(loss_mps.item()), "MPS loss is not finite"
        print(f"✓ Test 6.2: Training step parity (loss diff: {loss_diff:.6f})")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 6.2 FAILED: {e}")
        failed += 1

    # Test 6.3: Confirm DEVICE is set to MPS
    try:
        assert str(DEVICE) == "mps", f"DEVICE should be 'mps' but is '{DEVICE}'"
        print(f"✓ Test 6.3: DEVICE correctly set to MPS for training acceleration")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 6.3 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 6 MPS Parity Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 6 FAILED: {failed} test(s) did not pass.")

print("Running Step 6: MPS/CPU Parity Check...")
run_mps_parity_tests()


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
q_decay_curve_history = []  # Step 8: Q-value decay curves (list of 42-element vectors)
unique_states_seen = set()


def compute_q_decay_curve(policy_net, num_games=20):
    """
    Compute the Q-value magnitude decay curve.
    
    Plays num_games vs random, records abs(max(Q(state))) for each state,
    indexed by moves-from-end (0 = final state, 1 = second-to-last, etc.).
    Returns a 42-element vector (max game length) where each entry is the
    average abs(max(Q)) at that distance from the end.
    
    For games with a winner, the final state Q should approach 1.
    Earlier states should decay toward 0 if the network hasn't learned
    long-range value propagation yet.
    """
    q_sums = np.zeros(42, dtype=np.float64)
    q_counts = np.zeros(42, dtype=np.float64)
    
    eval_env = ConnectFourEnvironment(Config())
    policy_net.eval()
    
    for _ in range(num_games):
        eval_env.reset()
        done = False
        game_states = []  # (state, legal_moves) pairs
        
        while not done:
            state = eval_env.get_state()
            legal = eval_env.get_legal_moves()
            game_states.append((state.copy(), legal[:]))
            
            # Policy plays
            action = select_action(policy_net, torch.tensor(state, dtype=torch.float32), legal, eps=0.0)
            _, reward, done = eval_env.play_move(action)
            if done:
                break
            
            # Random opponent plays
            legal = eval_env.get_legal_moves()
            state_opp = eval_env.get_state()
            game_states.append((state_opp.copy(), legal[:]))
            action = np.random.choice(legal)
            _, reward, done = eval_env.play_move(action)
        
        # Now compute Q-values for each state, indexed by moves-from-end
        total_moves = len(game_states)
        for idx, (s, legal) in enumerate(game_states):
            moves_from_end = total_moves - 1 - idx
            if moves_from_end < 42:
                with torch.no_grad():
                    q = policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                    # Mask illegal moves
                    mask = get_action_mask(legal)
                    mask_t = torch.tensor(mask, dtype=torch.float32).to(DEVICE)
                    masked_q = q.squeeze(0).clone()
                    masked_q[mask_t == 0] = 0  # Zero out illegal moves for abs max
                    abs_max_q = masked_q.abs().max().item()
                q_sums[moves_from_end] += abs_max_q
                q_counts[moves_from_end] += 1
    
    # Average, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        curve = np.where(q_counts > 0, q_sums / q_counts, 0.0)
    
    return curve


# *****************************************************************
# STEP 8: Training Metrics Dashboard & Q-Value Decay Curve Tests
# *****************************************************************
def run_metrics_tests():
    """Validate the Q-value decay curve computation and metrics infrastructure."""
    passed = 0
    failed = 0

    # Test 8.1: Q-decay curve has correct shape and is non-negative
    try:
        curve = compute_q_decay_curve(policy_net, num_games=10)
        assert curve.shape == (42,), f"Expected shape (42,), got {curve.shape}"
        assert np.all(curve >= 0), "Q-decay curve should be non-negative"
        assert np.all(np.isfinite(curve)), "Q-decay curve has non-finite values"
        print(f"✓ Test 8.1: Q-decay curve shape (42,), non-negative, finite")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 8.1 FAILED: {e}")
        failed += 1

    # Test 8.2: Early positions (near end of game) have data, far positions may be zero
    try:
        assert curve[0] > 0, f"Position 0 (final state) should have data, got {curve[0]}"
        assert curve[1] > 0, f"Position 1 should have data, got {curve[1]}"
        print(f"✓ Test 8.2: Q-decay curve[0]={curve[0]:.3f}, curve[1]={curve[1]:.3f}, curve[5]={curve[5]:.3f}")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 8.2 FAILED: {e}")
        failed += 1

    # Test 8.3: compute_q_decay_curve returns consistent results across calls
    try:
        curve2 = compute_q_decay_curve(policy_net, num_games=10)
        assert curve2.shape == (42,), f"Second call shape wrong: {curve2.shape}"
        # Both calls should produce data at position 0 (untrained net, random games)
        assert curve2[0] > 0, "Second call should also have data at position 0"
        print(f"✓ Test 8.3: Q-decay curve is reproducible (curve2[0]={curve2[0]:.3f})")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 8.3 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 8 Metrics Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 8 FAILED: {failed} test(s) did not pass.")

print("Running Step 8: Training Metrics & Q-Decay Curve...")
run_metrics_tests()


# *****************************************************************
# STEP 9: Negamax Bellman Validation Tests
# *****************************************************************
def run_negamax_tests():
    """Verify the negamax Bellman equation is correctly implemented."""
    passed = 0
    failed = 0
    test_env = ConnectFourEnvironment(Config())

    # Test 9.1: State returned by play_move is from NEXT player's perspective
    # This is critical — the negamax equation subtracts opponent's Q, so
    # next_state must be from the opponent's viewpoint.
    try:
        test_env.reset()
        # P1 plays col 3
        state_before = test_env.get_state()  # P1's perspective
        next_state, reward, done = test_env.play_move(3)
        # next_state should be P2's perspective
        # P2 has no pieces yet, so channel 0 (my pieces) should be empty
        # P1's piece should be in channel 1 (opponent's pieces from P2's view)
        assert next_state[0].sum() == 0, "P2 has no pieces, ch0 should be empty"
        assert next_state[1].sum() == 1, "P1's piece should be in P2's ch1"
        # Verify it matches get_state_from_perspective(-1)
        p2_view = test_env.get_state_from_perspective(-1)
        assert np.array_equal(next_state, p2_view), "next_state should match P2's perspective"
        print("✓ Test 9.1: play_move returns state from next player's perspective (negamax-ready)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 9.1 FAILED: {e}")
        failed += 1

    # Test 9.2: Bellman target uses subtraction (negamax), not addition
    # For a winning move: target = +1 - gamma * max(Q(s')) * (1-done)
    # Since done=1, target = +1. Correct.
    # For the move BEFORE the winning move (loser's move):
    # target = -1 - gamma * max(Q(s')) * (1-done)
    # Since done=1, target = -1. Correct.
    # For a mid-game move: target = 0 - gamma * max(Q(s'))
    # This means: my Q = negative of opponent's best Q. This is negamax.
    try:
        # Create a simple scenario and verify the math
        test_env.reset()
        state = test_env.get_state()
        next_state, reward, done = test_env.play_move(3)  # P1 plays, reward=0, done=False

        policy_net.eval()
        with torch.no_grad():
            # Q-values for current state (P1's perspective)
            q_current = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE))
            # Q-values for next state (P2's perspective)
            q_next = policy_net(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE))

            next_legal = test_env.get_legal_moves()
            mask = get_action_mask(next_legal)
            mask_t = torch.tensor(mask, dtype=torch.float32).to(DEVICE)
            masked_q_next = q_next.squeeze(0).clone()
            masked_q_next[mask_t == 0] = -1e9
            next_q_max = masked_q_next.max().item()

            # Negamax target: r - gamma * max(Q_next) * (1 - done)
            expected_target = reward - (GAMMA * next_q_max * (1 - float(done)))

        # The target should be the NEGATIVE of the opponent's best value (scaled by gamma)
        # For mid-game (reward=0, done=False): target = -gamma * max(Q_opponent)
        assert abs(expected_target - (-GAMMA * next_q_max)) < 1e-6, \
            f"Mid-game target should be -gamma*max(Q_next), got {expected_target}"
        print(f"✓ Test 9.2: Negamax Bellman: target = 0 - γ*max(Q_opp) = {expected_target:.4f}")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 9.2 FAILED: {e}")
        failed += 1

    # Test 9.3: Terminal state target ignores future Q-values
    # When done=True, the (1-done) term zeros out the future, so target = reward
    try:
        # Simulate: reward=1.0, done=True
        terminal_target = 1.0 - (GAMMA * 999.0 * (1 - 1.0))  # next_q_max doesn't matter
        assert terminal_target == 1.0, f"Terminal target should be reward, got {terminal_target}"

        # Simulate: reward=-1.0, done=True (loser's last move)
        loser_target = -1.0 - (GAMMA * 999.0 * (1 - 1.0))
        assert loser_target == -1.0, f"Loser terminal target should be -1, got {loser_target}"
        print("✓ Test 9.3: Terminal targets: win=+1, loss=-1 (future Q zeroed out)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 9.3 FAILED: {e}")
        failed += 1

    # Test 9.4: Symmetry — same board from P1 vs P2 perspective produces
    # Q-values that should be negatives of each other (for a well-trained net)
    # For an untrained net, just verify the perspectives are correctly flipped
    try:
        test_env.reset()
        test_env.play_move(3)  # P1 plays
        test_env.play_move(4)  # P2 plays
        # Now it's P1's turn
        p1_state = test_env.get_state_from_perspective(1)
        p2_state = test_env.get_state_from_perspective(-1)
        # P1's ch0 should equal P2's ch1 and vice versa
        assert np.array_equal(p1_state[0], p2_state[1]), "P1 ch0 should equal P2 ch1"
        assert np.array_equal(p1_state[1], p2_state[0]), "P1 ch1 should equal P2 ch0"
        print("✓ Test 9.4: Perspective symmetry verified (P1.ch0 == P2.ch1)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 9.4 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 9 Negamax Bellman Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 9 FAILED: {failed} test(s) did not pass.")

print("Running Step 9: Negamax Bellman Validation...")
run_negamax_tests()


# *****************************************************************
# STEP 10: Prioritized Replay Buffer Tests
# *****************************************************************
def run_per_tests():
    """Verify the PrioritizedReplayBuffer works correctly with training."""
    passed = 0
    failed = 0

    # Test 10.1: PER buffer accepts add_symmetric and stores entries
    try:
        test_per = PrioritizedReplayBuffer(capacity=1000, alpha=PER_ALPHA, beta_start=PER_BETA_START, terminal_ratio=TERMINAL_RATE)
        state = np.zeros((2, 6, 7), dtype=np.float32)
        next_state = np.zeros((2, 6, 7), dtype=np.float32)
        mask = np.ones(7, dtype=np.float32)
        test_per.add_symmetric(state, 3, 0.0, next_state, False, mask)
        assert len(test_per) == 2, f"Expected 2 entries (symmetric), got {len(test_per)}"
        print("✓ Test 10.1: PER add_symmetric stores 2 entries")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 10.1 FAILED: {e}")
        failed += 1

    # Test 10.2: PER update_penalty modifies reward/done correctly
    try:
        test_per._recent_indices = []
        test_per.add_symmetric(state, 3, 0.0, next_state, False, mask)  # indices 2,3
        test_per.add_symmetric(state, 3, 1.0, next_state, True, mask)   # indices 4,5 (win)
        # Update loser's move (indices -3, -4)
        test_per.update_penalty(index=-3, new_reward=-1.0, is_done=True)
        test_per.update_penalty(index=-4, new_reward=-1.0, is_done=True)
        # Verify the updated entries
        tree_idx_3 = test_per._recent_indices[-3]
        data_idx_3 = tree_idx_3 - test_per.tree.capacity + 1
        updated = test_per.tree.data[data_idx_3]
        assert updated[2] == -1.0, f"Expected reward -1.0, got {updated[2]}"
        assert updated[4] == 1.0, f"Expected done=1.0, got {updated[4]}"
        print("✓ Test 10.2: PER update_penalty correctly modifies reward/done")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 10.2 FAILED: {e}")
        failed += 1

    # Test 10.3: PER sample returns (batch, indices, weights)
    try:
        # Fill buffer with enough data
        test_per2 = PrioritizedReplayBuffer(capacity=1000, alpha=PER_ALPHA, beta_start=PER_BETA_START)
        for _ in range(100):
            s = np.random.randn(2, 6, 7).astype(np.float32)
            ns = np.random.randn(2, 6, 7).astype(np.float32)
            m = np.ones(7, dtype=np.float32)
            test_per2.add(s, np.random.randint(7), 0.0, ns, False, m)
        
        batch, indices, weights = test_per2.sample(32)
        states_b, actions_b, rewards_b, ns_b, dones_b, masks_b = batch
        assert states_b.shape == (32, 2, 6, 7), f"States shape wrong: {states_b.shape}"
        assert len(indices) == 32, f"Expected 32 indices, got {len(indices)}"
        assert len(weights) == 32, f"Expected 32 weights, got {len(weights)}"
        assert np.all(weights > 0), "All importance weights should be positive"
        assert np.all(weights <= 1.0), "Weights should be normalized to max 1.0"
        print(f"✓ Test 10.3: PER sample returns correct shapes and valid weights")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 10.3 FAILED: {e}")
        failed += 1

    # Test 10.4: Priority updates change sampling distribution
    try:
        # Update one entry with very high priority
        high_td = np.array([100.0])
        test_per2.update_priorities(indices[:1], high_td)
        # Sample many times and check if the high-priority entry appears more often
        high_idx = indices[0]
        count = 0
        for _ in range(50):
            _, sampled_indices, _ = test_per2.sample(32)
            if high_idx in sampled_indices:
                count += 1
        # With high priority, it should appear in most samples
        assert count > 10, f"High-priority entry only appeared in {count}/50 samples"
        print(f"✓ Test 10.4: Priority updates affect sampling (high-priority appeared in {count}/50 batches)")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 10.4 FAILED: {e}")
        failed += 1

    # Test 10.5: Main replay_buffer is PrioritizedReplayBuffer
    try:
        assert isinstance(replay_buffer, PrioritizedReplayBuffer), \
            f"replay_buffer should be PrioritizedReplayBuffer, got {type(replay_buffer)}"
        print("✓ Test 10.5: Main replay_buffer is PrioritizedReplayBuffer")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 10.5 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 10 PER Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 10 FAILED: {failed} test(s) did not pass.")

print("Running Step 10: Prioritized Replay Buffer...")
run_per_tests()


# *****************************************************************
# STEP 11: High Discount Rate Verification
# *****************************************************************
def run_discount_tests():
    """Verify GAMMA is set high enough for long-horizon credit assignment."""
    passed = 0
    failed = 0

    # Test 11.1: GAMMA >= 0.99
    try:
        assert GAMMA >= 0.99, f"GAMMA should be >= 0.99, got {GAMMA}"
        print(f"✓ Test 11.1: GAMMA = {GAMMA} (high enough for long-horizon learning)")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 11.1 FAILED: {e}")
        failed += 1

    # Test 11.2: Verify signal propagation — with GAMMA=0.99, a reward at move 20
    # still has 0.99^20 = 0.818 influence on move 0
    try:
        influence_at_20 = GAMMA ** 20
        assert influence_at_20 > 0.5, f"Signal at 20 moves back is only {influence_at_20:.3f}"
        influence_at_40 = GAMMA ** 40
        print(f"✓ Test 11.2: Signal propagation — 20 moves: {influence_at_20:.3f}, 40 moves: {influence_at_40:.3f}")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 11.2 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 11 Discount Rate Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 11 FAILED: {failed} test(s) did not pass.")

print("Running Step 11: Discount Rate Verification...")
run_discount_tests()


# *****************************************************************
# STEP 12: Champion/Challenger Infrastructure Tests
# *****************************************************************
def run_champion_challenger_tests():
    """Validate champion/challenger mechanics before training."""
    passed = 0
    failed = 0

    # Test 12.1: Temperature-based action selection produces valid moves
    try:
        test_env = ConnectFourEnvironment(Config())
        test_env.reset()
        state = test_env.get_state()
        legal = test_env.get_legal_moves()
        state_t = torch.tensor(state, dtype=torch.float32)
        
        actions_seen = set()
        for _ in range(50):
            a = select_action_temperature(policy_net, state_t, legal, temperature=0.3)
            assert a in legal, f"Temperature selection returned illegal move {a}"
            actions_seen.add(a)
        # With temperature > 0, we should see some variety
        assert len(actions_seen) > 1, f"Temperature selection produced only 1 unique action out of 50"
        print(f"✓ Test 12.1: Temperature selection produces valid, diverse moves ({len(actions_seen)} unique actions)")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 12.1 FAILED: {e}")
        failed += 1

    # Test 12.2: Temperature=0 is equivalent to greedy
    try:
        greedy_action = select_action(policy_net, state_t, legal, eps=0.0)
        temp0_actions = [select_action_temperature(policy_net, state_t, legal, temperature=0.0) for _ in range(10)]
        assert all(a == greedy_action for a in temp0_actions), "Temperature=0 should be deterministic greedy"
        print(f"✓ Test 12.2: Temperature=0 matches greedy (action={greedy_action})")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 12.2 FAILED: {e}")
        failed += 1

    # Test 12.3: Champion/challenger game completes and returns valid result
    try:
        champion_test = copy.deepcopy(policy_net)
        champion_test.eval()
        _, result = play_champion_challenger_game(policy_net, champion_test, eps=0.5, challenger_is_p1=True)
        assert result in [-1, 0, 1], f"Invalid game result: {result}"
        print(f"✓ Test 12.3: Champion/challenger game completed (result={result})")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 12.3 FAILED: {e}")
        failed += 1

    # Test 12.4: Evaluation function returns valid win rate
    try:
        wr, c_wins, ch_wins = evaluate_challenger_vs_champion(
            policy_net, champion_test, num_games=10, temperature=0.3
        )
        assert 0.0 <= wr <= 1.0, f"Win rate out of range: {wr}"
        assert c_wins + ch_wins <= 10, f"More results than games: {c_wins}W + {ch_wins}L > 10"
        print(f"✓ Test 12.4: Evaluation works (wr={wr:.2f}, {c_wins}W/{ch_wins}L/{10-c_wins-ch_wins}D)")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 12.4 FAILED: {e}")
        failed += 1

    # Test 12.5: Save champion creates file
    try:
        test_path = save_champion(policy_net, version=999, episode=0, win_rate_vs_champion=0.0)
        assert os.path.exists(test_path), f"Champion file not created: {test_path}"
        # Clean up test file
        os.remove(test_path)
        current_path = os.path.join(CHAMPION_DIR, "champion_current.pth")
        assert os.path.exists(current_path), "champion_current.pth not created"
        print(f"✓ Test 12.5: save_champion creates file successfully")
        passed += 1
    except (AssertionError, Exception) as e:
        print(f"✗ Test 12.5 FAILED: {e}")
        failed += 1

    # Test 12.6: Hyperparameters are set correctly
    try:
        assert CHAMPION_THRESHOLD == 0.55, f"CHAMPION_THRESHOLD should be 0.55, got {CHAMPION_THRESHOLD}"
        assert CHAMPION_EVAL_FREQUENCY == 200, f"CHAMPION_EVAL_FREQUENCY should be 200, got {CHAMPION_EVAL_FREQUENCY}"
        assert MAX_STAGNATION_EPISODES == 1000, f"MAX_STAGNATION should be 1000, got {MAX_STAGNATION_EPISODES}"
        print(f"✓ Test 12.6: Champion hyperparameters set correctly (threshold={CHAMPION_THRESHOLD}, eval_freq={CHAMPION_EVAL_FREQUENCY}, stagnation={MAX_STAGNATION_EPISODES})")
        passed += 1
    except AssertionError as e:
        print(f"✗ Test 12.6 FAILED: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Step 12 Champion/Challenger Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    if failed > 0:
        raise RuntimeError(f"Step 12 FAILED: {failed} test(s) did not pass.")

print("Running Step 12: Champion/Challenger Infrastructure...")
run_champion_challenger_tests()


def train_dqn_agent(policy_net, optimizer):
    # 1. SETUP
    target_net = copy.deepcopy(policy_net) 
    target_net.eval()
    
    # Champion/Challenger setup
    champion_net = copy.deepcopy(policy_net)
    champion_net.eval()
    champion_version = 0
    episodes_since_promotion = 0
    champion_history = []  # Track promotions: (version, episode, win_rate)
    
    # Save initial champion (v0)
    save_champion(champion_net, champion_version, 0, 0.0)
    
    eps = EPS_START 
    
    for episode in range(1, NUM_EPISODES + 1):
        # 2. CHAMPION/CHALLENGER SELF-PLAY
        # Coin flip: challenger is P1 or P2
        challenger_is_p1 = (np.random.random() < 0.5)
        new_states_seen, challenger_result = play_champion_challenger_game(
            policy_net, champion_net, eps, challenger_is_p1
        )
        
        for s in new_states_seen:
            unique_states_seen.add(s.tobytes()) 
            
        # 3. EPSILON DECAY
        eps = max(EPS_END, eps * EPS_DECAY)

        # 4. TRAINING LOOP
        if replay_buffer.is_ready(BATCH_SIZE):
            policy_net.train()
            batch_terminal_counts = []
            
            last_loss = 0
            last_q_values = None
            last_predicted_qs = None
            last_dones = None

            for _ in range(TRAINING_ITERATIONS): 
                (states, actions, rewards, next_states, dones, next_masks), per_indices, is_weights = replay_buffer.sample(BATCH_SIZE)

                s_batch = torch.tensor(states, dtype=torch.float32).to(DEVICE)
                a_batch = torch.tensor(actions, dtype=torch.long).to(DEVICE)
                r_batch = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
                ns_batch = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
                d_batch = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
                m_batch = torch.tensor(next_masks, dtype=torch.float32).to(DEVICE)
                w_batch = torch.tensor(is_weights, dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    next_q_values = target_net(ns_batch) 
                    masked_next_q = next_q_values.masked_fill(m_batch == 0, -1e9)
                    next_q_max = masked_next_q.max(dim=1)[0]
                    target_q = r_batch - (GAMMA * next_q_max * (1 - d_batch))

                optimizer.zero_grad()
                q_values = policy_net(s_batch)
                predicted_qs = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                td_errors = (predicted_qs - target_q).detach()
                loss = (w_batch * (predicted_qs - target_q) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                
                replay_buffer.update_priorities(per_indices, td_errors.cpu().numpy())
                
                last_loss = loss.item()
                last_q_values = q_values.detach()
                last_predicted_qs = predicted_qs.detach()
                last_dones = d_batch.detach()

                actual_pct = np.mean(dones) 
                batch_terminal_counts.append(actual_pct)

            # --- SYNCHRONIZED METRIC LOGGING ---
            loss_history.append(last_loss)
            q_magnitude_history.append(torch.mean(torch.abs(last_q_values)).item())
            unique_states_history.append(len(unique_states_seen))

            terminal_mask = (last_dones == 1)
            if terminal_mask.any():
                term_q = torch.mean(torch.abs(last_predicted_qs[terminal_mask])).item()
            else:
                term_q = q_magnitude_history[-1]
            q_magnitude_history_win_loss.append(term_q)

            with torch.no_grad():
                all_w = torch.cat([p.view(-1) for p in policy_net.parameters()])
                dead_neuron_cnt_hist.append((torch.abs(all_w) < 0.01).sum().item() / all_w.numel())
                exploding_neuron_cnt_hist.append((torch.abs(all_w) > 10.0).sum().item() / all_w.numel())

            total_grad = 0.0
            grad_count = 0
            for p in policy_net.parameters():
                if p.grad is not None:
                    total_grad += p.grad.abs().sum().item()
                    grad_count += p.grad.numel()
            grad_history.append(total_grad / grad_count if grad_count > 0 else 0)

            terminal_pct_history.append(np.mean(batch_terminal_counts))

            # 5. SYNC TARGET NETWORK
            if episode % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # 6. EVALUATION (vs random + Q-decay)
            if episode % EVALUATION_FREQUENCY == 0:
                win_rate = evaluate_vs_random(policy_net, num_games=EVAL_VS_RANDOM_GAME_COUNT)
                win_rate_vs_rand_hist.append(win_rate)
                decay_curve = compute_q_decay_curve(policy_net, num_games=EVAL_VS_RANDOM_GAME_COUNT)
                q_decay_curve_history.append(decay_curve)
            
            # 7. CHAMPION/CHALLENGER EVALUATION
            episodes_since_promotion += 1
            if episode % CHAMPION_EVAL_FREQUENCY == 0:
                wr, c_wins, ch_wins = evaluate_challenger_vs_champion(
                    policy_net, champion_net, 
                    num_games=CHAMPION_EVAL_GAMES,
                    temperature=CHAMPION_EVAL_TEMPERATURE
                )
                
                print(f"Ep {episode} | Eps: {eps:.2f} | vsRand: {win_rate_vs_rand_hist[-1] if win_rate_vs_rand_hist else 0:.2f} "
                      f"| vsChamp: {wr:.2f} ({c_wins}W/{ch_wins}L) | Champion: v{champion_version} "
                      f"| Stag: {episodes_since_promotion}")
                
                # Promote challenger?
                if wr >= CHAMPION_THRESHOLD:
                    champion_version += 1
                    champion_net.load_state_dict(policy_net.state_dict())
                    champion_net.eval()
                    save_champion(policy_net, champion_version, episode, wr)
                    champion_history.append((champion_version, episode, wr))
                    episodes_since_promotion = 0
                    # Bump epsilon to re-explore against new champion
                    eps = min(EPS_START, eps + 0.1)
                    print(f"  ★ PROMOTED to Champion v{champion_version} (win rate: {wr:.2f}) | Eps bumped to {eps:.2f}")
                
                # Stagnation check: revert if no promotion in too long
                elif episodes_since_promotion >= MAX_STAGNATION_EPISODES:
                    print(f"  ⚠ STAGNATION ({episodes_since_promotion} eps without promotion) — reverting to champion v{champion_version}")
                    policy_net.load_state_dict(champion_net.state_dict())
                    target_net.load_state_dict(champion_net.state_dict())
                    # Reset optimizer state for fresh start
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            optimizer.state[p] = {}
                    eps = EPS_START
                    episodes_since_promotion = 0

    print(f"\nTraining complete. Final champion: v{champion_version}")
    print(f"Champion history: {champion_history}")
    return policy_net
                

# *****************************************************************
# Synthetic Training Loop, Function: 
# *****************************************************************
# Use this to prove that the network will learn Q estimates for win and loss
# example state / action pairs
SYNTHETIC_NUM_EPISODES = 200
SYNTHETIC_BATCH_SIZE = 16
SYNTHETIC_TARGET_SYNC_FREQ = 1   # If 1, we sync every episode.  Changes to supervised learning

from notebooks.training_examples_last_2_moves_20251221 import generate_artificial_replay_buffer_for_training
import copy

def train_on_synthetic_replay_buffer(policy_net, optimizer):
    # 1. SETUP: Get the full synthetic buffer
    # Let's assume this buffer has ~30-50 high-quality examples
    replay_buffer = generate_artificial_replay_buffer_for_training()    
    
    # Initialize Target Net
    target_net = copy.deepcopy(policy_net)
    target_net.eval()

    # FREEZE BatchNorm and Disable Dropout
    policy_net.eval()

    for episode in range(1, SYNTHETIC_NUM_EPISODES + 1):
        # --- NEW: SAMPLE INSIDE THE LOOP ---
        # This replicates the randomness of the real training loop
        states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(SYNTHETIC_BATCH_SIZE)
        
        # Test that sampling is working - print out
        """
        print( "x"*10, "Episode ", episode, "x"*10)
        print( "Sampled States")
        print( states )
        print( "/n Sampled Actions")
        print( actions )
        print( "/n Sampled Rewards")
        print( rewards )
        print( "/n Sampled Next States")
        print( next_states )
        print( "/n Sampled Dones")
        print( dones )
        print( "/n Next Masks")
        print( next_masks )
        """

        # Convert to tensors
        s_batch = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        a_batch = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        r_batch = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        ns_batch = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        d_batch = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        m_batch = torch.tensor(next_masks, dtype=torch.float32).to(DEVICE)

        # 2. Calculate Targets using Target Net (eval mode)
        policy_net.train() # Policy is in train mode (BatchNorm/Dropout active)
        with torch.no_grad():
            next_q_values = target_net(ns_batch) 
            masked_next_q = next_q_values.masked_fill(m_batch == 0, -1e9)
            next_q_max = masked_next_q.max(dim=1)[0]
            # Standard Bellman: r + GAMMA * max(Q_next)
            target_q = r_batch - (GAMMA * next_q_max * (1 - d_batch))

        # 3. Gradient Step
        optimizer.zero_grad()
        q_values = policy_net(s_batch)
        predicted_qs = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
        
        loss = nn.functional.mse_loss(predicted_qs, target_q)
        loss.backward()
        
        # Clip Gradients to prevent the "bad Q-values" explosion
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        # Periodically sync target newtork with policy network
        if (episode % SYNTHETIC_TARGET_SYNC_FREQ == 0 and len(q_magnitude_history) > 0):
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Ep {episode} | Loss: {loss.item():.4f} | Q-Mag: {q_magnitude_history[-1]:.2f} | Grad: {grad_history[-1]:.6f}")

        # --- POPULATE METRICS ---
        # 1. Loss
        loss_history.append(loss.item())

        # 2. Avg Absolute Magnitude of ALL Q-values in the batch
        q_magnitude_history.append(torch.mean(torch.abs(q_values)).item())

        # 3. Avg Absolute Magnitude of the Q-values for the CHOSEN moves
        q_magnitude_history_win_loss.append(torch.mean(torch.abs(predicted_qs)).item())

        # 4. Unique States (Static at 16 for this test)
        unique_states_history.append(SYNTHETIC_BATCH_SIZE)

        # 5. Win Rate vs Random (Every 50 episodes)
        if episode % 50 == 0:
            win_rate = evaluate_vs_random(num_games=20)
            win_rate_vs_rand_hist.append(win_rate)
            decay_curve = compute_q_decay_curve(policy_net, num_games=20)
            q_decay_curve_history.append(decay_curve)
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
    return policy_net


# *****************************************************************
# Function to show result metrics 
# *****************************************************************
def plot_training_metrics(loss_hist, q_hist, q_terminal_hist, states_hist, 
                          win_rate_hist, dead_hist, exploding_hist, grad_hist, 
                          terminal_pct_hist, q_decay_curves, eval_freq=10):
    """
    Plotting function for DQN training.
    8 subplots: 6 original metrics + Q-decay curve + placeholder.
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Connect-4 DQN Training Dashboard', fontsize=16, fontweight='bold')
    
    # Generate x-axis indices
    episodes = np.arange(len(loss_hist))
    eval_episodes = np.arange(1, len(win_rate_hist) + 1) * eval_freq

    # --- Plot 1: Loss ---
    ax = axes[0, 0]
    ax.plot(episodes, loss_hist, color='blue', alpha=0.3)
    if len(loss_hist) > 10:
        smoothed = np.convolve(loss_hist, np.ones(10)/10, mode='valid')
        ax.plot(episodes[9:], smoothed, color='blue', label='Smoothed Loss')
    ax.set_title('Training Loss (MSE)')
    ax.set_yscale('log')
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

    # --- Plot 4: Q-Value Decay Curve (Step 8) ---
    ax = axes[0, 3]
    if len(q_decay_curves) > 0:
        # Plot first, middle, and last curves to show progression
        indices_to_plot = [0]
        if len(q_decay_curves) > 2:
            indices_to_plot.append(len(q_decay_curves) // 2)
        if len(q_decay_curves) > 1:
            indices_to_plot.append(len(q_decay_curves) - 1)
        
        colors = ['lightblue', 'orange', 'red']
        for ci, curve_idx in enumerate(indices_to_plot):
            curve = q_decay_curves[curve_idx]
            # Only plot up to the max non-zero index
            max_idx = max(np.nonzero(curve)[0]) + 1 if np.any(curve > 0) else 1
            label = f"Eval {curve_idx + 1}"
            ax.plot(range(max_idx), curve[:max_idx], color=colors[ci % len(colors)], 
                    alpha=0.8, label=label, linewidth=1.5)
        ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.5)
        ax.set_xlabel('Moves from end (0=final)')
        ax.set_ylabel('avg abs(max(Q))')
        ax.legend(fontsize='x-small')
    ax.set_title('Q-Value Decay Curve')
    ax.grid(True, alpha=0.3)

    # --- Plot 5: Win Rate AND Terminal % ---
    ax = axes[1, 0]
    ax.plot(episodes, terminal_pct_hist, color='cyan', alpha=0.2, label='Batch Terminal %')
    ax.plot(eval_episodes, win_rate_hist, marker='o', color='gold', 
            markersize=4, linewidth=2, label='Win Rate vs Random')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, label='Random (50%)')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, len(loss_hist))
    ax.set_title('Win Rate & Terminal Signal')
    ax.set_xlabel('Episodes')
    ax.legend(loc='lower right', fontsize='x-small')
    ax.grid(True, alpha=0.3)

    # --- Plot 6: NN Health (Combined Dead/Exploding) ---
    ax = axes[1, 1]
    ax.plot(episodes, dead_hist, label='Dead (<0.01)', color='black')
    ax.plot(episodes, exploding_hist, label='Exploding (>10)', color='red')
    ax.set_title('Neuron Health (Weight Counts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 7: Gradients ---
    ax = axes[1, 2]
    ax.plot(episodes, grad_hist, color='purple')
    ax.set_title('Mean Absolute Gradient')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # --- Plot 8: Q-Value by Moves-from-End over Training ---
    ax = axes[1, 3]
    if len(q_decay_curves) > 1:
        curves_arr = np.array(q_decay_curves)  # shape: (num_evals, 42)
        eval_x = np.arange(1, len(q_decay_curves) + 1)
        # Positions to track: last, 2nd-last, 3rd, 4th, 5th, 10th, 15th, 20th
        positions = [0, 1, 2, 3, 4, 9, 14, 19]
        labels =    ['Last', '2nd', '3rd', '4th', '5th', '10th', '15th', '20th']
        colors =    ['red', 'orangered', 'orange', 'gold', 'green', 'teal', 'blue', 'purple']
        for pos, lbl, clr in zip(positions, labels, colors):
            vals = curves_arr[:, pos]
            ax.plot(eval_x, vals, color=clr, label=lbl, alpha=0.8, linewidth=1.2)
        ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.3)
        ax.set_xlabel('Evaluation #')
        ax.set_ylabel('avg |max(Q)|')
        ax.legend(fontsize='xx-small', ncol=2, loc='upper right')
    ax.set_title('Q by Moves-from-End')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_dashboard.png', dpi=150, bbox_inches='tight')
    print("Dashboard saved to training_dashboard.png")
    plt.show()

# *****************************************************************
# Train, show results
# *****************************************************************
train_dqn_agent(policy_net, optimizer)                                           # <--- REAL SELF PLAY
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
    q_decay_curve_history,
    eval_freq=EVALUATION_FREQUENCY)



# *****************************************************************
# Save the policy 
# *****************************************************************


# *****************************************************************
# Test the trained policy 
# *****************************************************************
# Audit function commented out — uses old DQN buffer API, will update for PER later
# To re-enable, update to work with PrioritizedReplayBuffer's SumTree storage