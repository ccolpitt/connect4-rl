"""
Interactive gameplay against trained agents.

This script allows you to play Connect 4 against a trained agent (DQN or A2C)
through a command-line interface.

Usage:
    python src/gameplay/play_against_agent.py --model models/a2c_agent_20251211_2228.pth --agent-type a2c
    python src/gameplay/play_against_agent.py --model models/dqn_agent.pth --agent-type dqn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
from typing import Optional

from src.environment import ConnectFourEnvironment, Config
from src.agents import A2CAgent, DQNAgent


class HumanPlayer:
    """Human player for interactive gameplay."""
    
    def __init__(self, name: str = "Human", player_id: int = 1):
        self.name = name
        self.player_id = player_id
    
    def select_action(self, state: np.ndarray, legal_moves: list) -> int:
        """Prompt human for action."""
        while True:
            try:
                print(f"\nLegal moves: {legal_moves}")
                action = int(input(f"{self.name}, enter column (0-6): "))
                if action in legal_moves:
                    return action
                else:
                    print(f"❌ Invalid move! Column {action} is not legal.")
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Game cancelled.")
                sys.exit(0)


def render_board_pretty(env: ConnectFourEnvironment):
    """Render board with colors and better formatting."""
    display = {1: '🔴', -1: '🟡', 0: '⚪'}
    
    print("\n" + "=" * 29)
    for row in env.board:
        print("│ " + " ".join(display[val] for val in row) + " │")
    print("=" * 29)
    print("  0  1  2  3  4  5  6")
    print()


def play_game(
    human_player: HumanPlayer,
    agent,
    env: ConnectFourEnvironment,
    human_first: bool = True
) -> Optional[int]:
    """
    Play one game between human and agent.
    
    Args:
        human_player: Human player
        agent: Trained agent (A2C or DQN)
        env: Game environment
        human_first: If True, human plays first (Player 1)
    
    Returns:
        Winner: 1 if human wins, -1 if agent wins, 0 if draw, None if cancelled
    """
    env.reset()
    done = False
    moves = 0
    
    # Determine who plays which side
    if human_first:
        human_id = 1
        agent_id = -1
        print("\n🎮 You are Player 1 (🔴)")
        print("🤖 Agent is Player 2 (🟡)")
    else:
        human_id = -1
        agent_id = 1
        print("\n🤖 Agent is Player 1 (🔴)")
        print("🎮 You are Player 2 (🟡)")
    
    render_board_pretty(env)
    
    while not done and moves < 42:
        current_player = env.get_current_player()
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        
        # Select action based on current player
        if current_player == human_id:
            action = human_player.select_action(state, legal_moves)
            print(f"\n🎮 You played column {action}")
        else:
            action = agent.select_action(state, legal_moves)
            print(f"\n🤖 Agent played column {action}")
        
        # Execute move
        next_state, reward, done = env.play_move(action)
        moves += 1
        
        # Render board
        render_board_pretty(env)
        
        # Check for winner
        if done:
            winner = env.check_winner()
            if winner == human_id:
                print("🎉 YOU WIN! Congratulations!")
                return 1
            elif winner == agent_id:
                print("😔 Agent wins. Better luck next time!")
                return -1
            else:
                print("🤝 It's a draw!")
                return 0
    
    return 0  # Draw if board full


def main():
    """Main gameplay loop."""
    parser = argparse.ArgumentParser(description='Play Connect 4 against a trained agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (e.g., models/a2c_agent.pth)')
    parser.add_argument('--agent-type', type=str, required=True, choices=['a2c', 'dqn'],
                        help='Type of agent (a2c or dqn)')
    parser.add_argument('--human-first', action='store_true',
                        help='Human plays first (default: agent plays first)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CONNECT 4 - HUMAN VS AGENT")
    print("=" * 60)
    
    # Load agent
    print(f"\n🤖 Loading {args.agent_type.upper()} agent from {args.model}...")
    
    config = Config()
    env = ConnectFourEnvironment(config)
    
    if args.agent_type == 'a2c':
        agent = A2CAgent(
            name="A2C-Agent",
            player_id=-1 if args.human_first else 1,
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
            name="DQN-Agent",
            player_id=-1 if args.human_first else 1,
            conv_channels=[32, 64],
            fc_dims=[128],
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=0.0,  # No exploration for evaluation
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=10000,
            batch_size=32,
            target_update_freq=100
        )
    
    try:
        agent.load(args.model)
        print(f"✅ Agent loaded successfully!")
        print(f"   Training steps: {agent.train_steps:,}")
        if hasattr(agent, 'episodes_completed'):
            print(f"   Episodes completed: {agent.episodes_completed:,}")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return
    
    # Create human player
    human = HumanPlayer(name="Human", player_id=1 if args.human_first else -1)
    
    # Game statistics
    human_wins = 0
    agent_wins = 0
    draws = 0
    games_played = 0
    
    print("\n" + "=" * 60)
    print("GAME RULES")
    print("=" * 60)
    print("• Connect 4 pieces in a row (horizontal, vertical, or diagonal) to win")
    print("• Enter column number (0-6) to drop your piece")
    print("• Press Ctrl+C to quit at any time")
    print("=" * 60)
    
    # Main game loop
    while True:
        games_played += 1
        print(f"\n{'=' * 60}")
        print(f"GAME {games_played}")
        print(f"{'=' * 60}")
        print(f"Score: You {human_wins} - {agent_wins} Agent (Draws: {draws})")
        
        result = play_game(human, agent, env, human_first=args.human_first)
        
        if result == 1:
            human_wins += 1
        elif result == -1:
            agent_wins += 1
        else:
            draws += 1
        
        # Ask to play again
        print(f"\n{'=' * 60}")
        print(f"FINAL SCORE: You {human_wins} - {agent_wins} Agent (Draws: {draws})")
        print(f"{'=' * 60}")
        
        try:
            play_again = input("\nPlay again? (y/n): ").lower().strip()
            if play_again != 'y':
                break
        except KeyboardInterrupt:
            print("\n")
            break
    
    # Final statistics
    print(f"\n{'=' * 60}")
    print("FINAL STATISTICS")
    print(f"{'=' * 60}")
    print(f"Games played: {games_played}")
    print(f"Your wins: {human_wins} ({100*human_wins/games_played:.1f}%)")
    print(f"Agent wins: {agent_wins} ({100*agent_wins/games_played:.1f}%)")
    print(f"Draws: {draws} ({100*draws/games_played:.1f}%)")
    print(f"{'=' * 60}")
    print("\n👋 Thanks for playing!")


if __name__ == "__main__":
    main()