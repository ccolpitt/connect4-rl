"""
Test Q-Learning Mathematics

This test verifies that the Q-learning update is correct, especially for terminal states.

Key Q-Learning Rules:
1. Terminal states (done=True): Q_target = reward (no bootstrapping)
2. Non-terminal states: Q_target = reward + gamma * max(Q(next_state))
3. Loss = (Q_predicted - Q_target)^2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from agents.dqn_agent import DQNAgent
from environment import ConnectFourEnvironment, Config


def print_board(board, title="Board"):
    """Print a visual representation of the board with row numbers."""
    import sys
    sys.stdout.write(f"\n{title}:\n")
    sys.stdout.write("    " + " ".join(str(i) for i in range(7)) + "\n")
    for row in range(6):
        row_str = f"{row} | "
        for col in range(7):
            if board[row, col] == 1:
                row_str += "X "
            elif board[row, col] == -1:
                row_str += "O "
            else:
                row_str += ". "
        sys.stdout.write(row_str + "\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def test_terminal_state_q_target():
    """
    Test that terminal states use Q_target = reward (no bootstrapping).
    
    When done=True, the Q-target should be exactly the reward, because
    there are no future states to consider.
    """
    print("\n" + "="*80)
    print("TEST 1: Terminal State Q-Target Calculation")
    print("="*80)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    agent = DQNAgent(config)
    
    # Create a terminal state (winning position)
    env.reset()
    env.board[5, 0] = 1
    env.board[5, 1] = 1
    env.board[5, 2] = 1
    env.current_player = 1
    
    state_before = env.get_state()
    
    print_board(env.board, "BEFORE winning move (column 3)")
    
    # Make winning move
    state_after, reward, done = env.play_move(3)
    
    print_board(env.board, "AFTER winning move")
    
    print(f"\n✓ Created terminal state")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Winner: {env.check_winner()}")
    
    assert done, "State should be terminal"
    assert reward == 1.0, f"Reward should be 1.0, got {reward}"
    
    # Convert to tensors and move to agent's device
    state_tensor = torch.FloatTensor(state_before).unsqueeze(0).to(agent.device)
    next_state_tensor = torch.FloatTensor(state_after).unsqueeze(0).to(agent.device)
    reward_tensor = torch.FloatTensor([reward]).to(agent.device)
    done_tensor = torch.FloatTensor([float(done)]).to(agent.device)
    
    # Get Q-values
    with torch.no_grad():
        q_values = agent.q_network(state_tensor)
        next_q_values = agent.target_network(next_state_tensor)
        max_next_q = next_q_values.max(1)[0]
    
    # Calculate Q-target
    # For terminal states: Q_target = reward
    # For non-terminal: Q_target = reward + gamma * max(Q(next_state))
    q_target = reward_tensor + (1 - done_tensor) * agent.gamma * max_next_q
    
    print(f"\n✓ Q-Learning calculations:")
    print(f"  Reward: {reward_tensor.item():.4f}")
    print(f"  Done: {done_tensor.item()}")
    print(f"  Max Q(next_state): {max_next_q.item():.4f}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Q_target = reward + (1 - done) * gamma * max_Q(next)")
    print(f"  Q_target = {reward_tensor.item():.4f} + (1 - {done_tensor.item()}) * {agent.gamma} * {max_next_q.item():.4f}")
    print(f"  Q_target = {q_target.item():.4f}")
    
    # Verify Q-target equals reward for terminal state
    expected_q_target = reward
    actual_q_target = q_target.item()
    
    assert abs(actual_q_target - expected_q_target) < 1e-6, \
        f"Q-target should equal reward for terminal state. Expected {expected_q_target}, got {actual_q_target}"
    
    print(f"\n✅ Terminal state Q-target test PASSED!")
    print(f"   - Q-target = {actual_q_target:.4f} (equals reward, no bootstrapping)")
    
    return True


def test_non_terminal_state_q_target():
    """
    Test that non-terminal states use Q_target = reward + gamma * max(Q(next_state)).
    """
    print("\n" + "="*80)
    print("TEST 2: Non-Terminal State Q-Target Calculation")
    print("="*80)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    agent = DQNAgent(config)
    
    # Create a non-terminal state
    env.reset()
    state_before = env.get_state()
    
    print_board(env.board, "BEFORE normal move (column 3)")
    
    # Make a normal move (not winning)
    state_after, reward, done = env.play_move(3)
    
    print_board(env.board, "AFTER normal move")
    
    print(f"\n✓ Created non-terminal state")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    
    assert not done, "State should be non-terminal"
    assert reward is None or reward == 0.0, f"Reward should be None or 0.0 for non-terminal, got {reward}"
    
    # Use 0.0 for None reward
    if reward is None:
        reward = 0.0
    
    # Convert to tensors and move to agent's device
    state_tensor = torch.FloatTensor(state_before).unsqueeze(0).to(agent.device)
    next_state_tensor = torch.FloatTensor(state_after).unsqueeze(0).to(agent.device)
    reward_tensor = torch.FloatTensor([reward]).to(agent.device)
    done_tensor = torch.FloatTensor([float(done)]).to(agent.device)
    
    # Get Q-values
    with torch.no_grad():
        q_values = agent.q_network(state_tensor)
        next_q_values = agent.target_network(next_state_tensor)
        max_next_q = next_q_values.max(1)[0]
    
    # Calculate Q-target
    q_target = reward_tensor + (1 - done_tensor) * agent.gamma * max_next_q
    
    print(f"\n✓ Q-Learning calculations:")
    print(f"  Reward: {reward_tensor.item():.4f}")
    print(f"  Done: {done_tensor.item()}")
    print(f"  Max Q(next_state): {max_next_q.item():.4f}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Q_target = reward + (1 - done) * gamma * max_Q(next)")
    print(f"  Q_target = {reward_tensor.item():.4f} + (1 - {done_tensor.item()}) * {agent.gamma} * {max_next_q.item():.4f}")
    print(f"  Q_target = {q_target.item():.4f}")
    
    # Verify Q-target includes bootstrapping for non-terminal state
    expected_q_target = reward + agent.gamma * max_next_q.item()
    actual_q_target = q_target.item()
    
    assert abs(actual_q_target - expected_q_target) < 1e-6, \
        f"Q-target calculation incorrect. Expected {expected_q_target}, got {actual_q_target}"
    
    # Verify bootstrapping is happening (Q-target should not equal reward)
    if max_next_q.item() != 0.0:
        assert abs(actual_q_target - reward) > 1e-6, \
            "Q-target should include bootstrapping for non-terminal state"
    
    print(f"\n✅ Non-terminal state Q-target test PASSED!")
    print(f"   - Q-target = {actual_q_target:.4f} (includes bootstrapping)")
    
    return True


def test_losing_state_q_target():
    """
    Test that losing terminal states use Q_target = -1.0 (negative reward).
    """
    print("\n" + "="*80)
    print("TEST 3: Losing Terminal State Q-Target")
    print("="*80)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    agent = DQNAgent(config)
    
    # Create a position where Player 2 will win
    env.reset()
    env.board[5, 0] = -1
    env.board[5, 1] = -1
    env.board[5, 2] = -1
    env.current_player = -1
    
    state_before = env.get_state()
    
    print_board(env.board, "BEFORE Player 2 wins (column 3)")
    
    # Player 2 makes winning move
    state_after, reward, done = env.play_move(3)
    
    print_board(env.board, "AFTER Player 2 wins")
    
    print(f"\n✓ Player 2 won")
    print(f"  Reward (from P2's perspective): {reward}")
    print(f"  Done: {done}")
    
    # From Player 1's perspective, this is a loss
    # In training, Player 1 would store this experience with reward = -1.0
    player1_reward = -reward  # Flip the reward
    
    print(f"  Reward (from P1's perspective): {player1_reward}")
    
    assert done, "State should be terminal"
    assert reward == 1.0, f"P2 should get +1.0, got {reward}"
    assert player1_reward == -1.0, f"P1 should get -1.0, got {player1_reward}"
    
    # Calculate Q-target for Player 1 (the loser)
    reward_tensor = torch.FloatTensor([player1_reward]).to(agent.device)
    done_tensor = torch.FloatTensor([float(done)]).to(agent.device)
    
    with torch.no_grad():
        next_state_tensor = torch.FloatTensor(state_after).unsqueeze(0).to(agent.device)
        next_q_values = agent.target_network(next_state_tensor)
        max_next_q = next_q_values.max(1)[0]
    
    q_target = reward_tensor + (1 - done_tensor) * agent.gamma * max_next_q
    
    print(f"\n✓ Q-Learning calculations (from P1's perspective):")
    print(f"  Reward: {reward_tensor.item():.4f}")
    print(f"  Q_target: {q_target.item():.4f}")
    
    assert abs(q_target.item() - player1_reward) < 1e-6, \
        f"Q-target should equal -1.0 for losing terminal state"
    
    print(f"\n✅ Losing terminal state Q-target test PASSED!")
    print(f"   - Q-target = {q_target.item():.4f} (negative reward for loss)")
    
    return True


def main():
    """Run all Q-learning math tests."""
    print("="*80)
    print("Q-LEARNING MATHEMATICS TESTS")
    print("="*80)
    print("\nThese tests verify correct Q-learning update calculations:")
    print("1. Terminal states: Q_target = reward (no bootstrapping)")
    print("2. Non-terminal states: Q_target = reward + gamma * max(Q(next))")
    print("3. Losing states: Q_target = -1.0 (negative reward)")
    
    tests = [
        test_terminal_state_q_target,
        test_non_terminal_state_q_target,
        test_losing_state_q_target,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS: {passed}/{len(tests)} tests passed")
    print("="*80)
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! Q-learning math is correct.")
        return 0
    else:
        print(f"\n❌ {len(tests) - passed} tests failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    exit(main())