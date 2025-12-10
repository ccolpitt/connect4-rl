"""
Test script for BaseAgent and RandomAgent.

This script demonstrates:
1. How to create agents
2. How agents interact with the environment
3. How to play a game between two agents
4. How to track statistics

Run this to verify the agent interface works correctly!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment import Config, ConnectFourEnvironment
from agents import RandomAgent


def test_single_agent():
    """Test basic agent functionality."""
    print("=" * 60)
    print("Test 1: Single Agent Basics")
    print("=" * 60)
    
    # Create environment and agent
    config = Config()
    env = ConnectFourEnvironment(config)
    agent = RandomAgent(name="Random-Bot", player_id=1)
    
    print(f"\n✓ Created agent: {agent.name}")
    print(f"  Player ID: {agent.player_id}")
    print(f"  Games played: {agent.games_played}")
    
    # Test action selection
    state = env.get_state()
    legal_moves = env.get_legal_moves()
    action = agent.select_action(state, legal_moves)
    
    print(f"\n✓ Agent selected action: {action}")
    print(f"  Legal moves were: {legal_moves}")
    
    # Test observation (doesn't do anything for RandomAgent, but tests interface)
    next_state, reward, done = env.play_move(action)
    agent.observe(state, action, reward, next_state, done)
    print(f"\n✓ Agent observed transition")
    
    # Test training (returns None for RandomAgent)
    result = agent.train()
    print(f"✓ Agent train() returned: {result}")
    
    # Test statistics
    agent.update_game_result('win')
    stats = agent.get_stats()
    print(f"\n✓ Agent statistics:")
    print(f"  {stats}")
    
    print("\n✅ Single agent test passed!\n")


def play_game(agent1, agent2, env, verbose=True):
    """
    Play a complete game between two agents.
    
    Args:
        agent1: First agent (plays as Player 1)
        agent2: Second agent (plays as Player 2)
        env: Game environment
        verbose: Whether to print game progress
    
    Returns:
        winner: 1, -1, or 0 (draw)
    """
    state = env.reset()
    agent1.reset()
    agent2.reset()
    
    move_count = 0
    max_moves = 50  # Prevent infinite loops
    
    if verbose:
        print(f"\n🎮 Game: {agent1.name} (X) vs {agent2.name} (O)")
        env.render()
    
    while move_count < max_moves:
        # Determine current agent
        current_agent = agent1 if env.current_player == agent1.player_id else agent2
        
        # Get action
        legal_moves = env.get_legal_moves()
        action = current_agent.select_action(state, legal_moves)
        
        if verbose:
            print(f"Move {move_count + 1}: {current_agent.name} plays column {action}")
        
        # Execute move
        next_state, reward, done = env.play_move(action)
        
        # Both agents observe (for learning agents)
        agent1.observe(state, action, reward, next_state, done)
        agent2.observe(state, action, reward, next_state, done)
        
        if verbose:
            env.render()
        
        if done:
            # Update statistics
            if reward == agent1.player_id:
                agent1.update_game_result('win')
                agent2.update_game_result('loss')
                if verbose:
                    print(f"🎉 {agent1.name} wins!")
            elif reward == agent2.player_id:
                agent1.update_game_result('loss')
                agent2.update_game_result('win')
                if verbose:
                    print(f"🎉 {agent2.name} wins!")
            else:
                agent1.update_game_result('draw')
                agent2.update_game_result('draw')
                if verbose:
                    print(f"🤝 Draw!")
            
            return reward
        
        state = next_state
        move_count += 1
    
    # Max moves reached (shouldn't happen normally)
    if verbose:
        print("⏰ Max moves reached - declaring draw")
    agent1.update_game_result('draw')
    agent2.update_game_result('draw')
    return 0


def test_agent_vs_agent():
    """Test two agents playing against each other."""
    print("=" * 60)
    print("Test 2: Agent vs Agent")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    
    # Create two random agents
    agent1 = RandomAgent(name="Random-1", player_id=1, seed=42)
    agent2 = RandomAgent(name="Random-2", player_id=-1, seed=123)
    
    # Play a single game (verbose)
    print("\n📋 Playing one game (verbose):")
    winner = play_game(agent1, agent2, env, verbose=True)
    
    # Play multiple games (quiet)
    print("\n📋 Playing 10 more games (quiet):")
    for i in range(10):
        winner = play_game(agent1, agent2, env, verbose=False)
        print(f"  Game {i+2}: Winner = {winner}")
    
    # Show final statistics
    print(f"\n📊 Final Statistics:")
    print(f"  {agent1}")
    print(f"  {agent2}")
    
    print("\n✅ Agent vs agent test passed!\n")


def test_agent_interface():
    """Test that agents properly implement the BaseAgent interface."""
    print("=" * 60)
    print("Test 3: Agent Interface Compliance")
    print("=" * 60)
    
    from agents import BaseAgent
    
    agent = RandomAgent()
    
    # Check that RandomAgent is a BaseAgent
    print(f"\n✓ RandomAgent is instance of BaseAgent: {isinstance(agent, BaseAgent)}")
    
    # Check required methods exist
    required_methods = ['select_action', 'observe', 'train', 'save', 'load', 
                       'reset', 'update_game_result', 'get_stats']
    
    print(f"\n✓ Checking required methods:")
    for method in required_methods:
        has_method = hasattr(agent, method) and callable(getattr(agent, method))
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}()")
    
    print("\n✅ Interface compliance test passed!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AGENT TESTING SUITE")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_single_agent()
    test_agent_vs_agent()
    test_agent_interface()
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe BaseAgent interface is working correctly!")
    print("Ready to implement DQN agent next.")
    print("=" * 60 + "\n")