"""
Evaluate Trained DQN Agent

This script loads a trained DQN agent and evaluates its performance
against a random agent to measure final win rate.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import DQNAgent, RandomAgent
from environment import ConnectFourEnvironment, Config


def evaluate_agent(model_path: str, num_games: int = 1000):
    """Evaluate trained agent against random opponent."""
    
    print("=" * 80)
    print("EVALUATING TRAINED DQN AGENT")
    print("=" * 80)
    
    # Load trained agent
    print(f"\n📂 Loading model: {model_path}")
    agent = DQNAgent(name="DQN-Trained", player_id=1)
    agent.load(model_path)
    agent.set_exploration(0.0)  # Greedy evaluation
    
    print(f"✓ Model loaded")
    print(f"  Training steps: {agent.train_steps:,}")
    print(f"  Episodes completed: {agent.episodes_completed:,}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    
    # Create opponent
    random_agent = RandomAgent(name="Random", player_id=-1)
    
    # Evaluate
    print(f"\n🎮 Playing {num_games} games vs random agent...")
    config = Config()
    env = ConnectFourEnvironment(config)
    
    dqn_wins = 0
    random_wins = 0
    draws = 0
    total_moves = 0
    
    for game in range(num_games):
        env.reset()
        done = False
        moves = 0
        current_player = 1
        
        while not done and moves < 42:
            state = env.get_state()
            legal_moves = env.get_legal_moves()
            
            if current_player == 1:
                action = agent.select_action(state, legal_moves)
            else:
                action = random_agent.select_action(state, legal_moves)
            
            next_state, reward, done = env.play_move(action)
            moves += 1
            current_player *= -1
        
        winner = env.check_winner()
        total_moves += moves
        
        if winner == 1:
            dqn_wins += 1
        elif winner == -1:
            random_wins += 1
        else:
            draws += 1
        
        if (game + 1) % 100 == 0:
            print(f"  Progress: {game + 1}/{num_games} games...")
    
    # Results
    win_rate = dqn_wins / num_games
    avg_moves = total_moves / num_games
    
    print(f"\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nGames played: {num_games}")
    print(f"DQN wins: {dqn_wins} ({win_rate:.1%})")
    print(f"Random wins: {random_wins} ({random_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    print(f"Average game length: {avg_moves:.1f} moves")
    
    print(f"\n📊 Performance Assessment:")
    if win_rate >= 0.95:
        print(f"  ✅ EXCELLENT - Agent dominates random play ({win_rate:.1%})")
    elif win_rate >= 0.85:
        print(f"  ✓ GOOD - Agent significantly outperforms random ({win_rate:.1%})")
    elif win_rate >= 0.70:
        print(f"  ⚠️  MODERATE - Agent is better but not dominant ({win_rate:.1%})")
    else:
        print(f"  ❌ POOR - Agent needs more training ({win_rate:.1%})")
    
    print("=" * 80)
    
    return win_rate, avg_moves


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of evaluation games (default: 1000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    evaluate_agent(args.model, args.games)