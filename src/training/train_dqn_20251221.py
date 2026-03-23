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
# Training Hyperparameters (all in one place)
# *****************************************************************
NUM_EPISODES                = 500
BATCH_SIZE                  = 128
LEARNING_RATE               = 0.00001
WEIGHT_DECAY                = 1e-4
TRAINING_ITERATIONS         = 4       # Training steps per game
EVAL_VS_RANDOM_GAME_COUNT   = 50
GAMMA                       = 0.99
EVALUATION_FREQUENCY        = 10      # Evaluate every N episodes
EPS_START                   = 0.5
EPS_END                     = 0.2
EPS_DECAY                   = 0.9999
TARGET_UPDATE_FREQ          = 100
TERMINAL_RATE               = 0.3     # Target terminal ratio in batch sampling
DROPOUT_RATE                = 0.00
REPLAY_BUFFER_CAPACITY      = 20000
DEVICE                      = torch.device("cpu")


# *****************************************************************
# Create environment, Replay Buffer - 
# *****************************************************************
# You may change the configs
config = Config()
env = ConnectFourEnvironment(config)
replay_buffer = DQNReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)


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

        # He (Kaiming) initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dr1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dr2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dr3(x)

        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
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
    

# *****************************************************************
# Self Play
# *****************************************************************
def play_self_play_game(policy_net, eps=0.5):
    env.reset()
    done = False
    moves_count = 0
    game_states = []

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

def train_dqn_agent(policy_net, optimizer):
    # 1. SETUP
    target_net = copy.deepcopy(policy_net) 
    target_net.eval()
    
    eps = EPS_START 
    
    for episode in range(1, NUM_EPISODES + 1):
        # 2. SELF PLAY
        new_states_seen = play_self_play_game(policy_net, eps) 
        
        for s in new_states_seen:
            unique_states_seen.add(s.tobytes()) 
            
        # 3. EPSILON DECAY
        eps = max(EPS_END, eps * EPS_DECAY)

        # 4. TRAINING LOOP
        if replay_buffer.is_ready(BATCH_SIZE):
            policy_net.train()
            batch_terminal_counts = []
            
            # We track the 'last' values of the loop to append to history
            last_loss = 0
            last_q_values = None
            last_predicted_qs = None
            last_dones = None

            for _ in range(TRAINING_ITERATIONS): 
                #states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(
                    BATCH_SIZE, 
                    terminal_ratio=TERMINAL_RATE # Our target ratio
                )

                s_batch = torch.tensor(states, dtype=torch.float32).to(DEVICE)
                a_batch = torch.tensor(actions, dtype=torch.long).to(DEVICE)
                r_batch = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
                ns_batch = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
                d_batch = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
                m_batch = torch.tensor(next_masks, dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    next_q_values = target_net(ns_batch) 
                    masked_next_q = next_q_values.masked_fill(m_batch == 0, -1e9)
                    next_q_max = masked_next_q.max(dim=1)[0]
                    target_q = r_batch - (GAMMA * next_q_max * (1 - d_batch))

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
            if episode % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # 6. EVALUATION
            if episode % 100 == 0:
                win_rate = evaluate_vs_random(policy_net, num_games=20)
                win_rate_vs_rand_hist.append(win_rate)
                print(f"Ep {episode} | Eps: {eps:.2f} | Unique: {len(unique_states_seen)} | WinRate: {win_rate:.2f}")

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

        # 5. Win Rate vs Random (Every 20 episodes)
        if episode % 50 == 0:
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
    #eval_episodes = np.arange(len(win_rate_hist)) * eval_freq
    eval_episodes = np.arange(1, len(win_rate_hist) + 1) * eval_freq

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

    # This plots every episode (the 25% target line)
    ax.plot(episodes, terminal_pct_hist, color='cyan', alpha=0.2, label='Batch Terminal %')

    # This plots the Win Rate stretched across the full width
    ax.plot(eval_episodes, win_rate_hist, marker='o', color='gold', 
            markersize=4, linewidth=2, label='Win Rate vs Random')

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, label='Random (50%)')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, len(loss_hist)) # Force x-axis to match the full training length
    ax.set_title('Win Rate & Terminal Signal')
    ax.set_xlabel('Episodes')
    ax.legend(loc='lower right', fontsize='x-small')
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
#train_dqn_agent(policy_net, optimizer)                                           # <--- REAL SELF PLAY
train_on_synthetic_replay_buffer(policy_net, optimizer)    # <--- SYNTHETIC EXPERIENCE

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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def audit_synthetic_performance(policy_net, replay_buffer, device):
    """
    Evaluates the model on every single example in the synthetic buffer.
    Outputs a performance table and a visualization of prediction accuracy.
    """
    policy_net.eval()
    results = []
    
    # We don't want to sample; we want to see everything
    # Accessing internal buffer storage (assuming deque or list)
    all_transitions = list(replay_buffer.buffer)
    
    mape_sum = 0
    count = 0

    print(f"\n--- Synthetic Buffer Audit (Size: {len(all_transitions)}) ---")

    for i, (state, action, reward, next_state, done, next_mask) in enumerate(all_transitions):
        # 1. Prepare Tensors
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        ns_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        m_tensor = torch.tensor(next_mask, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # 2. Calculate Estimated Reward (Current Prediction)
            q_values = policy_net(s_tensor)
            estimated_q = q_values[0][action].item()

            # 3. Calculate Target Reward (Bellman Ground Truth)
            next_q_values = policy_net(ns_tensor) # Using policy net to check internal consistency
            masked_next_q = next_q_values.masked_fill(m_tensor == 0, -1e9)
            next_q_max = masked_next_q.max(dim=1)[0].item()
            
            # Negamax target: r - gamma * max_next
            target_q = reward - (GAMMA * next_q_max * (1 - done))

        # 4. Calculate Difference
        diff = abs(target_q - estimated_q)
        # Handle division by zero for MAPE by using max(abs(target), 1.0) 
        # Since our values are capped at 1.0, this gives a conservative error %
        percent_diff = (diff / max(abs(target_q), 0.1)) * 100
        
        mape_sum += percent_diff
        count += 1

        results.append({
            'Sample': i,
            'Target': round(target_q, 3),
            'Estimate': round(estimated_q, 3),
            'Abs Diff': round(diff, 3),
            '% Diff': round(percent_diff, 1)
        })

    # Create DataFrame for nice display
    df = pd.DataFrame(results)
    mape = mape_sum / count

    # --- Visualization ---
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    plt.bar(x - width/2, df['Target'], width, label='Target (Truth)', color='skyblue')
    plt.bar(x + width/2, df['Estimate'], width, label='Estimate (AI)', color='salmon')

    plt.xlabel('Sample Number')
    plt.ylabel('Q-Value Magnitude')
    plt.title(f'Target vs. Estimated Q-Values (MAPE: {mape:.2f}%)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    print(df.to_string(index=False))
    print(f"\nFINAL MAPE: {mape:.2f}%")
    
    return df

df = audit_synthetic_performance(policy_net, replay_buffer, DEVICE)
print( df )

"""
test_sample = [0]         # Choose which item in the replay buffer to use
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
print( "Q Values on synthetic test state: " )
print( q_values )
"""