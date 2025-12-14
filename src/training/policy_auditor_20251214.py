"""
Policy Evaluator - Tactical Test Suite for Connect 4 Agents

This module provides a comprehensive test suite to evaluate whether agents can
handle basic tactical scenarios in Connect 4. These tests check if agents can:
- Block immediate threats (defensive)
- Take winning moves (offensive)
- Prevent opponent from creating unstoppable threats
- Create their own unstoppable threats

The tests are designed to be independent of the agent type (DQN, A2C, PPO, etc.)
and provide clear pass/fail results for each tactical scenario.

Usage:
    from src.training.policy_evaluator_20251214 import evaluate_policy
    
    results = evaluate_policy(agent, verbose=True)
    print(f"Passed: {results['passed']}/{results['total']}")
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from src.environment import ConnectFourEnvironment, Config


@dataclass
class TestScenario:
    """
    Represents a single tactical test scenario.
    
    Attributes:
        name: Human-readable name of the test
        description: What the test checks
        board: 2D numpy array representing the board state (6x7)
               1 = Player 1 (agent), -1 = Player 2 (opponent), 0 = empty
        current_player: Which player's turn it is (1 or -1)
        correct_actions: List of column indices that are correct moves
        category: Type of test (defensive, offensive, strategic)
    """
    name: str
    description: str
    board: np.ndarray
    current_player: int
    correct_actions: List[int]
    category: str


def create_test_scenarios() -> List[TestScenario]:
    """
    Create all tactical test scenarios.
    
    Returns:
        List of TestScenario objects
    """
    scenarios = []
    
    # ========================================================================
    # DEFENSIVE TESTS - Can the agent block threats?
    # ========================================================================
    
    # Test 1: Block horizontal threat (bottom row)
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [-1, -1, -1, 0, 0, 0, 0]  # Opponent has 3 in a row, must block at col 3
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Horizontal Threat (Bottom)",
        description="Opponent has 3 in a row horizontally at bottom, agent must block",
        board=board.copy(),
        current_player=1,
        correct_actions=[3],
        category="defensive"
    ))
    
    # Test 2: Block horizontal threat (middle row)
    board = np.array([
        [0,  0,  0,  0,  0, 0, 0],
        [0,  0,  0,  0,  0, 0, 0],
        [0,  0,  0,  0,  0, 0, 0],
        [0, -1, -1, -1,  0, 0, 0],  # Opponent has 3 in a row, must block at col 4
        [0,  1,  1,  1, -1, 0, 0],
        [0, -1,  1, -1,  1, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Horizontal Threat (Middle)",
        description="Opponent has 3 in a row horizontally in middle, agent must block",
        board=board.copy(),
        current_player=1,
        correct_actions=[4],
        category="defensive"
    ))
    
    # Test 3: Block vertical threat
    board = np.array([
        [0,  0, 0, 0, 0, 0, 0],
        [0,  0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],  # Opponent has 3 in a column, must block at col 1
        [0, -1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0,  1, 1, 1, 0, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Vertical Threat",
        description="Opponent has 3 in a column, agent must block on top",
        board=board.copy(),
        current_player=1,
        correct_actions=[1],
        category="defensive"
    ))
    
    # Test 4: Block diagonal threat (/)
    board = np.array([
        [0,  0,  0,  0, 0, 0, 0],
        [0,  0,  0,  0, 0, 0, 0],
        [0,  0,  0, -1, 0, 0, 0],  # Opponent has 3 diagonal, must block at col 4
        [0,  0, -1,  1, 0, 0, 0],
        [0, -1,  1,  1, 0, 0, 0],
        [0,  1, -1, -1, 1, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Diagonal Threat (/)",
        description="Opponent has 3 in diagonal (/) pattern, agent must block",
        board=board.copy(),
        current_player=1,
        correct_actions=[0],
        category="defensive"
    ))
    
    # Test 5: Block diagonal threat (\)
    board = np.array([
        [0, 0, 0,  0,  0,  0, 0],
        [0, 0, 0,  0,  0,  0, 0],
        [0, 0, 0, -1,  0,  0, 0],  # Opponent has 3 diagonal, must block at col 6
        [0, 0, 0,  1, -1,  0, 0],
        [0, 0, 0,  1,  1, -1, 0],
        [0, 0, 1, -1, -1,  1, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Diagonal Threat (\\)",
        description="Opponent has 3 in diagonal (\\) pattern, agent must block",
        board=board.copy(),
        current_player=1,
        correct_actions=[6],
        category="defensive"
    ))
    
    # Test 6: Block before opponent gets 3 in a row (horizontal)
    board = np.array([
        [ 0,  0,  0, 0, 0, 0, 0],
        [ 0,  0,  0, 0, 0, 0, 0],
        [ 0,  0,  0, 0, 0, 0, 0],
        [ 0,  0,  0, 0, 0, 0, 0],
        [ 0,  1,  1, 0, 0, 0, 0],
        [ 0, -1, -1, 0, 0, 0, 0]  # Opponent has 2 in a row, should block at col 0 or 3
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Chain Before 3 (Horizontal)",
        description="Opponent has 2 in a row, agent should block before it becomes 3",
        board=board.copy(),
        current_player=1,
        correct_actions=[0,3],
        category="defensive"
    ))
    
    # Test 7: Block before opponent gets 3 in a column (horizontal)
    board = np.array([
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  1, 0, 0, 0],  # Opponent has 2 in a column, should block at col 1 or 4
        [0, 0, -1, -1, 0, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Block Chain Before 3 (Vertical)",
        description="Opponent has 2 in a horiz row, agent should block on top",
        board=board.copy(),
        current_player=1,
        correct_actions=[1, 4],
        category="defensive"
    ))
    
    # ========================================================================
    # OFFENSIVE TESTS - Can the agent take winning moves?
    # ========================================================================
    
    # Test 8: Complete 4 in a row (horizontal)
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0]  # Agent has 3 in a row, can win at col 3
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Win Horizontal",
        description="Agent has 3 in a row horizontally, should complete the 4th",
        board=board.copy(),
        current_player=1,
        correct_actions=[3],
        category="offensive"
    ))
    
    # Test 9: Complete 4 in a column (vertical)
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],  # Agent has 3 in a column, can win at col 1
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Win Vertical",
        description="Agent has 3 in a column, should complete the 4th",
        board=board.copy(),
        current_player=1,
        correct_actions=[1],
        category="offensive"
    ))
    
    # Test 10: Complete 4 in diagonal (/)
    board = np.array([
        [0,  0,  0,  0,  0, 0, 0],
        [0,  0,  0,  0,  0, 0, 0],
        [0,  0,  0,  1,  0, 0, 0],  # Agent has 3 diagonal, can win at col 4
        [0,  0,  1, -1,  0, 0, 0],
        [0,  1, -1, -1,  0, 0, 0],
        [0, -1,  1,  1, -1, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Win Diagonal (/)",
        description="Agent has 3 in diagonal (/) pattern, should complete the 4th",
        board=board.copy(),
        current_player=1,
        correct_actions=[0],
        category="offensive"
    ))
    
    # Test 11: Complete 4 with gap (horizontal)
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, -1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0]  # Agent has 3 with gap, can win at col 2
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Win Horizontal with Gap",
        description="Agent has 3 in a row with a gap, should fill the gap to win",
        board=board.copy(),
        current_player=1,
        correct_actions=[2],
        category="offensive"
    ))
    
    # Test 12: Create unstoppable threat (double threat)
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],  # Playing at col 2 creates two threats
        [0, 1, -1, 1, 0, 0, 0]
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Create Unstoppable Threat",
        description="Agent can create a double threat that opponent cannot block both",
        board=board.copy(),
        current_player=1,
        correct_actions=[2],
        category="offensive"
    ))

    # Test 13: Create unstoppable threat (double threat)
    board = np.array([
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0,  0,  0, 0, 0, 0],
        [0, 0, -1, -1, 0, 0, 0],
        [0, 0,  1,  1, 0, 0, 0] # Create a double threat
    ], dtype=int)
    scenarios.append(TestScenario(
        name="Create Unstoppable Threat",
        description="Agent can create a double threat that opponent cannot block both",
        board=board.copy(),
        current_player=1,
        correct_actions=[1,4],
        category="offensive"
    ))


    return scenarios


def evaluate_policy(
    agent: Any,
    scenarios: List[TestScenario] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate an agent's policy on tactical test scenarios.
    
    Args:
        agent: Agent with select_action(state, legal_moves) method
        scenarios: List of test scenarios (uses default if None)
        verbose: If True, print detailed results for each test
    
    Returns:
        Dictionary with evaluation results:
        {
            'passed': int,
            'failed': int,
            'total': int,
            'pass_rate': float,
            'results': List[Dict],  # Detailed results for each test
            'by_category': Dict[str, Dict]  # Results grouped by category
        }
    """
    if scenarios is None:
        scenarios = create_test_scenarios()
    
    config = Config()
    env = ConnectFourEnvironment(config)
    
    results = []
    passed = 0
    failed = 0
    
    # Track results by category
    by_category = {}
    
    if verbose:
        print("=" * 80)
        print("TACTICAL POLICY EVALUATION")
        print("=" * 80)
        print()
    
    for i, scenario in enumerate(scenarios, 1):
        # Set up the board state
        env.board = scenario.board.copy()
        env.current_player = scenario.current_player
        
        # Get state from current player's perspective
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        
        # Get agent's action
        action = agent.select_action(state, legal_moves)
        
        # Check if action is correct
        is_correct = action in scenario.correct_actions
        
        if is_correct:
            passed += 1
        else:
            failed += 1
        
        # Store result
        result = {
            'test_number': i,
            'name': scenario.name,
            'description': scenario.description,
            'category': scenario.category,
            'agent_action': action,
            'correct_actions': scenario.correct_actions,
            'passed': is_correct
        }
        results.append(result)
        
        # Update category stats
        if scenario.category not in by_category:
            by_category[scenario.category] = {'passed': 0, 'failed': 0, 'total': 0}
        by_category[scenario.category]['total'] += 1
        if is_correct:
            by_category[scenario.category]['passed'] += 1
        else:
            by_category[scenario.category]['failed'] += 1
        
        # Print if verbose
        if verbose:
            status = "✓ PASS" if is_correct else "✗ FAIL"
            print(f"Test {i}: {scenario.name}")
            print(f"  Category: {scenario.category}")
            print(f"  Description: {scenario.description}")
            print(f"  Agent chose: Column {action}")
            print(f"  Correct: {scenario.correct_actions}")
            print(f"  Result: {status}")
            print()
    
    # Calculate pass rate
    total = len(scenarios)
    pass_rate = passed / total if total > 0 else 0.0
    
    # Calculate pass rate by category
    for category in by_category:
        cat_total = by_category[category]['total']
        cat_passed = by_category[category]['passed']
        by_category[category]['pass_rate'] = cat_passed / cat_total if cat_total > 0 else 0.0
    
    if verbose:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({pass_rate:.1%})")
        print(f"Failed: {failed}")
        print()
        print("By Category:")
        for category, stats in by_category.items():
            print(f"  {category.capitalize()}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")
        print("=" * 80)
    
    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'pass_rate': pass_rate,
        'results': results,
        'by_category': by_category
    }


def main():
    """
    Example usage: Test a trained agent.
    
    Usage:
        python src/training/policy_evaluator_20251214.py --model models/a2c_agent.pth --agent-type a2c
    """
    import argparse
    from src.agents import A2CAgent, DQNAgent
    
    parser = argparse.ArgumentParser(description='Evaluate agent on tactical scenarios')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--agent-type', type=str, required=True, choices=['a2c', 'dqn'],
                        help='Type of agent')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    
    args = parser.parse_args()
    
    # Load agent
    print(f"Loading {args.agent_type.upper()} agent from {args.model}...")
    
    if args.agent_type == 'a2c':
        agent = A2CAgent(
            name="A2C-Test",
            player_id=1,
            conv_channels=[32, 64],
            fc_dims=[128],
            learning_rate=3e-4,
            gamma=0.99,
            entropy_coef=0.05,
            value_coef=0.5,
            max_grad_norm=1.0,
            max_log_prob=10.0,
            n_steps=5
        )
    else:  # dqn
        agent = DQNAgent(
            name="DQN-Test",
            player_id=1,
            conv_channels=[32, 64],
            fc_dims=[128],
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=10000,
            batch_size=32,
            target_update_freq=100
        )
    
    agent.load(args.model)
    print("Agent loaded successfully!\n")
    
    # Evaluate
    results = evaluate_policy(agent, verbose=args.verbose)
    
    if not args.verbose:
        print(f"\nResults: {results['passed']}/{results['total']} passed ({results['pass_rate']:.1%})")
        print("\nBy Category:")
        for category, stats in results['by_category'].items():
            print(f"  {category.capitalize()}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")


if __name__ == "__main__":
    main()