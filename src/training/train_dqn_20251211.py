"""
Train DQN Agent - Full Training Loop with Metrics and Visualization

This script trains a DQN agent to play Connect 4 through self-play, with comprehensive
monitoring of training progress and network health.

This script trains a DQN agent to play Connect 4 through self-play, with comprehensive
monitoring of training progress and network health.

Training Strategy:
------------------
1. Self-play: Agent plays against itself, learning from both perspectives
2. Experience replay: Store transitions and sample randomly for training
3. Epsilon decay: Gradually reduce exploration as agent improves
4. Periodic evaluation: Test against random agent to measure progress

Metrics Tracked:
----------------
- Training loss over time
- Win rate vs random agent
- Epsilon decay
- Q-value statistics
- Network health (dead/exploding neurons)
- Gradient statistics

Usage:
------
    python src/training/train_dqn_20251211.py --episodes 50000 --eval-freq 1000
"""

import sys
import os
# Add project root to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

from src.environment import ConnectFourEnvironment, Config
from src.agents import DQNAgent, RandomAgent


class NetworkHealthMonitor:
    """
    Monitor neural network health during training.
    
    Tracks:
    - Dead neurons (weights near zero)
    - Exploding weights (very large magnitudes)
    - Gradient statistics
    - Weight distribution
    """
    
    def __init__(self, dead_threshold: float = 1e-3, exploding_threshold: float = 100.0):
        """
        Initialize health monitor.
        
        Args:
            dead_threshold: Threshold below which weights are considered "dead"
            exploding_threshold: Threshold above which weights are considered "exploding"
        """
        self.dead_threshold = dead_threshold
        self.exploding_threshold = exploding_threshold
    
    def analyze_network(self, network: torch.nn.Module) -> Dict[str, float]:
        """
        Analyze network parameters for health issues.
        
        Args:
            network: Neural network to analyze
        
        Returns:
            Dictionary with health metrics
        """
        all_weights = []
        dead_count = 0
        exploding_count = 0
        total_params = 0
        
        for name, param in network.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                
                # Count dead neurons
                dead_count += np.sum(np.abs(weights) < self.dead_threshold)
                
                # Count exploding weights
                exploding_count += np.sum(np.abs(weights) > self.exploding_threshold)
                
                total_params += weights.size
        
        all_weights = np.array(all_weights)
        
        return {
            'total_params': total_params,
            'dead_neurons': dead_count,
            'dead_percentage': 100.0 * dead_count / total_params if total_params > 0 else 0.0,
            'exploding_weights': exploding_count,
            'exploding_percentage': 100.0 * exploding_count / total_params if total_params > 0 else 0.0,
            'weight_mean': float(np.mean(all_weights)),
            'weight_std': float(np.std(all_weights)),
            'weight_min': float(np.min(all_weights)),
            'weight_max': float(np.max(all_weights)),
        }
    
    def analyze_gradients(self, network: torch.nn.Module) -> Dict[str, float]:
        """
        Analyze gradient statistics.
        
        Args:
            network: Neural network to analyze
        
        Returns:
            Dictionary with gradient metrics
        """
        all_grads = []
        
        for name, param in network.named_parameters():
            if param.grad is not None:
                grads = param.grad.data.cpu().numpy().flatten()
                all_grads.extend(grads)
        
        if len(all_grads) == 0:
            return {
                'grad_mean': 0.0,
                'grad_std': 0.0,
                'grad_min': 0.0,
                'grad_max': 0.0,
                'grad_norm': 0.0
            }
        
        all_grads = np.array(all_grads)
        
        return {
            'grad_mean': float(np.mean(all_grads)),
            'grad_std': float(np.std(all_grads)),
            'grad_min': float(np.min(all_grads)),
            'grad_max': float(np.max(all_grads)),
            'grad_norm': float(np.linalg.norm(all_grads))
        }


class TrainingMetrics:
    """Track and store training metrics over time."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        
        # Training metrics
        self.episodes = []
        self.losses = []
        self.mean_q_values = []
        self.epsilons = []
        
        # Evaluation metrics
        self.eval_episodes = []
        self.win_rates = []
        self.avg_game_lengths = []
        
        # Network health metrics
        self.health_episodes = []
        self.dead_neuron_percentages = []
        self.exploding_weight_percentages = []
        self.gradient_norms = []
        
        # State exploration metrics
        self.unique_states_episodes = []
        self.unique_states_counts = []
        self.unique_states_set = set()  # Track all unique states seen
        
        # Moving averages
        self.recent_losses = deque(maxlen=window_size)
        self.recent_q_values = deque(maxlen=window_size)
    
    def add_training_step(self, episode: int, loss: float, mean_q: float, epsilon: float):
        """Add training step metrics."""
        self.episodes.append(episode)
        self.losses.append(loss)
        self.mean_q_values.append(mean_q)
        self.epsilons.append(epsilon)
        self.recent_losses.append(loss)
        self.recent_q_values.append(mean_q)
    
    def add_evaluation(self, episode: int, win_rate: float, avg_length: float):
        """Add evaluation metrics."""
        self.eval_episodes.append(episode)
        self.win_rates.append(win_rate)
        self.avg_game_lengths.append(avg_length)
    
    def add_health_metrics(self, episode: int, dead_pct: float, exploding_pct: float, grad_norm: float):
        """Add network health metrics."""
        self.health_episodes.append(episode)
        self.dead_neuron_percentages.append(dead_pct)
        self.exploding_weight_percentages.append(exploding_pct)
        self.gradient_norms.append(grad_norm)
    
    def add_unique_state(self, state: np.ndarray):
        """Add a state to the unique states set using hash for memory efficiency."""
        # Use hash instead of tuple to save memory
        # Hash is deterministic for same state
        state_hash = hash(state.tobytes())
        self.unique_states_set.add(state_hash)
    
    def record_unique_states(self, episode: int):
        """Record the current count of unique states."""
        self.unique_states_episodes.append(episode)
        self.unique_states_counts.append(len(self.unique_states_set))
    
    def get_moving_average_loss(self) -> float:
        """Get moving average of recent losses."""
        return np.mean(self.recent_losses) if len(self.recent_losses) > 0 else 0.0
    
    def get_moving_average_q(self) -> float:
        """Get moving average of recent Q-values."""
        return np.mean(self.recent_q_values) if len(self.recent_q_values) > 0 else 0.0


def play_self_play_game(agent: DQNAgent, env: ConnectFourEnvironment, metrics: TrainingMetrics = None, track_states: bool = False) -> Tuple[int, int]:
    """
    Play one self-play game where agent plays both sides.
    
    Args:
        agent: DQN agent to train
        env: Connect 4 environment
        metrics: Optional metrics tracker to record unique states
        track_states: Whether to track unique states this episode (to avoid memory issues)
    
    Returns:
        Tuple of (winner, num_moves)
    """
    env.reset()
    done = False
    moves = 0
    
    while not done and moves < 42:
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        
        # Track unique state (only when requested to avoid memory issues)
        if track_states and metrics is not None:
            metrics.add_unique_state(state)
        
        # Agent selects action (use softmax sampling during training for better exploration)
        action = agent.select_action(state, legal_moves, use_softmax=True, temperature=1.0)
        
        # Execute action
        next_state, reward, done = env.play_move(action)
        moves += 1
        
        # Store experience
        agent.observe(state, action, reward, next_state, done)
        
        # Train if enough data
        if agent.replay_buffer.is_ready(agent.batch_size):
            agent.train()
    
    winner = env.check_winner()
    return winner, moves


def evaluate_against_random(
    agent: DQNAgent,
    num_games: int = 100,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Evaluate DQN agent against random agent.
    
    Args:
        agent: DQN agent to evaluate
        num_games: Number of evaluation games
        verbose: Print progress
    
    Returns:
        Tuple of (win_rate, avg_game_length)
    """
    config = Config()
    env = ConnectFourEnvironment(config)
    random_agent = RandomAgent(name="Random-Eval", player_id=-1)
    
    # Save current epsilon and set to greedy for evaluation
    original_epsilon = agent.epsilon
    agent.set_exploration(0.0)
    
    wins = 0
    losses = 0
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
                # Use greedy (argmax) during evaluation
                action = agent.select_action(state, legal_moves, use_softmax=False)
            else:
                action = random_agent.select_action(state, legal_moves)
            
            next_state, reward, done = env.play_move(action)
            moves += 1
            current_player *= -1
        
        winner = env.check_winner()
        total_moves += moves
        
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
        
        if verbose and (game + 1) % 20 == 0:
            print(f"  Evaluated {game + 1}/{num_games} games...")
    
    # Restore original epsilon
    agent.set_exploration(original_epsilon)
    
    win_rate = wins / num_games
    avg_length = total_moves / num_games
    
    return win_rate, avg_length


def plot_training_metrics(metrics: TrainingMetrics, save_path: str = None):
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


def train_dqn_agent(
    num_episodes: int = 50000,  # Number of games
    eval_freq: int = 1000,
    eval_games: int = 100,
    health_check_freq: int = 500,
    save_freq: int = 5000,
    model_save_path: str = None,
    plot_save_path: str = None
):
    """
    Train DQN agent with comprehensive monitoring.
    
    Args:
        num_episodes: Total training episodes
        eval_freq: Evaluate every N episodes
        eval_games: Number of games per evaluation
        health_check_freq: Check network health every N episodes
        save_freq: Save model every N episodes
        model_save_path: Path to save trained model
        plot_save_path: Path to save training plots
    """
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Set default paths with timestamp if not provided
    if model_save_path is None:
        model_save_path = f"models/dqn_agent_{timestamp}.pth"
    if plot_save_path is None:
        # Save plot in same directory as model
        model_dir = os.path.dirname(model_save_path)
        plot_save_path = os.path.join(model_dir, f"training_metrics_{timestamp}.png")
    
    print("=" * 80, flush=True)
    print("DQN AGENT TRAINING", flush=True)
    print("=" * 80, flush=True)
    print(f"\n📅 Training run: {timestamp}", flush=True)
    
    # Create agent and environment
    print("\n🤖 Initializing DQN agent...", flush=True)
    agent = DQNAgent(
        name="DQN-SelfPlay",
        player_id=1,
        conv_channels=[32, 64],
        fc_dims=[128],
        dropout_rate=0.1,
        learning_rate=1e-4,
        gamma=0.99,
        weight_decay=1e-5,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,  # Slower decay: ~1000 episodes to reach 0.37, ~2000 to reach 0.14
        buffer_size=100000,
        batch_size=64,
        min_buffer_size=500,  # Lower threshold so training starts sooner (~30 episodes)
        # Prioritized replay
        use_prioritized_replay=True,  # Use prioritized experience replay
        priority_alpha=0.6,  # Prioritization exponent (0.6-0.7 typical)
        priority_beta_start=0.4,  # Importance sampling (annealed to 1.0)
        priority_beta_frames=100000,  # Frames to anneal beta
        # Target network
        target_update_freq=1000,  # Only used if use_polyak=False
        use_polyak=True,  # Use Polyak averaging for smoother learning
        polyak_tau=0.005,  # Soft update coefficient (0.5% of Q-network per step)
        use_double_dqn=True
    )
    
    config = Config()
    env = ConnectFourEnvironment(config)
    
    print(f"  Agent: {agent}", flush=True)
    print(f"  Device: {agent.device}", flush=True)
    print(f"  Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}", flush=True)
    
    # Initialize monitoring
    metrics = TrainingMetrics(window_size=100)
    health_monitor = NetworkHealthMonitor()
    
    print(f"\n📋 Training configuration:", flush=True)
    print(f"  Episodes: {num_episodes:,}", flush=True)
    print(f"  Evaluation frequency: every {eval_freq} episodes", flush=True)
    print(f"  Health check frequency: every {health_check_freq} episodes", flush=True)
    print(f"  Model save frequency: every {save_freq} episodes", flush=True)
    print(f"  Action selection: Softmax sampling (training), Argmax (evaluation)", flush=True)
    if agent.use_prioritized_replay:
        print(f"  Replay buffer: Prioritized (α={agent.replay_buffer.alpha}, β={agent.replay_buffer.beta:.2f}→1.0)", flush=True)
    else:
        print(f"  Replay buffer: Uniform sampling", flush=True)
    print(f"  Target network: {'Polyak averaging (τ=0.005)' if agent.use_polyak else f'Hard update every {agent.target_update_freq} steps'}", flush=True)
    
    # Training loop
    print(f"\n🎮 Starting training...", flush=True)
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        # Play self-play game (track states every episode for first 100, then every 10)
        # This balances accuracy with memory usage
        track_states = (episode <= 100) or (episode % 10 == 0)
        winner, moves = play_self_play_game(agent, env, metrics, track_states)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record training metrics (if training occurred)
        if agent.train_steps > 0 and episode % 10 == 0:
            # Get recent training metrics
            if hasattr(agent, 'recent_loss') and hasattr(agent, 'recent_q'):
                metrics.add_training_step(
                    episode,
                    agent.recent_loss,
                    agent.recent_q,
                    agent.epsilon
                )
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            loss_str = f"{agent.recent_loss:.4f}" if hasattr(agent, 'recent_loss') and agent.recent_loss > 0 else "N/A"
            print(f"Episode {episode:4d} | Loss: {loss_str} | ε: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.replay_buffer):5d} | States: {len(metrics.unique_states_set):6d} | "
                  f"Steps: {agent.train_steps:6d}")
        
        # Periodic evaluation
        if episode % eval_freq == 0:
            print(f"\n📊 Episode {episode}/{num_episodes} - Evaluating...", flush=True)
            win_rate, avg_length = evaluate_against_random(agent, eval_games, verbose=True)
            metrics.add_evaluation(episode, win_rate, avg_length)
            
            # Record unique states count
            metrics.record_unique_states(episode)
            
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            eta = (num_episodes - episode) / eps_per_sec
            
            print(f"  Win rate vs random: {win_rate:.1%}", flush=True)
            print(f"  Avg game length: {avg_length:.1f} moves", flush=True)
            print(f"  Unique states explored: {len(metrics.unique_states_set):,}", flush=True)
            print(f"  Buffer size: {len(agent.replay_buffer):,}", flush=True)
            print(f"  Training steps: {agent.train_steps:,}", flush=True)
            print(f"  Speed: {eps_per_sec:.1f} eps/sec", flush=True)
            print(f"  ETA: {eta/60:.1f} minutes", flush=True)
        
        # Network health check
        if episode % health_check_freq == 0:
            health = health_monitor.analyze_network(agent.q_network)
            grads = health_monitor.analyze_gradients(agent.q_network)
            
            metrics.add_health_metrics(
                episode,
                health['dead_percentage'],
                health['exploding_percentage'],
                grads['grad_norm']
            )
            
            # Print health with evaluation or standalone
            if episode % eval_freq == 0 or episode % (health_check_freq * 2) == 0:
                print(f"  Network health (ep {episode}):")
                print(f"    Dead neurons: {health['dead_percentage']:.2f}%")
                print(f"    Exploding weights: {health['exploding_percentage']:.2f}%")
                print(f"    Gradient norm: {grads['grad_norm']:.2e}")
        
        # Save model periodically
        if episode % save_freq == 0:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            agent.save(model_save_path)
            print(f"  💾 Model saved to {model_save_path}")
    
    # Final evaluation
    print(f"\n🏁 Training complete!")
    print(f"\n📊 Final evaluation...")
    final_win_rate, final_avg_length = evaluate_against_random(agent, num_games=500, verbose=True)
    
    print(f"\n" + "=" * 80)
    print(f"FINAL RESULTS")
    print(f"=" * 80)
    print(f"Win rate vs random: {final_win_rate:.1%}")
    print(f"Average game length: {final_avg_length:.1f} moves")
    print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Total training steps: {agent.train_steps:,}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save(model_save_path)
    print(f"\n💾 Final model saved to {model_save_path}")
    
    # Plot metrics
    print(f"\n📈 Generating training plots...")
    plot_training_metrics(metrics, save_path=plot_save_path)
    
    return agent, metrics


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train DQN agent for Connect 4')
    parser.add_argument('--episodes', type=int, default=50000,
                        help='Number of training episodes (default: 50000)')
    parser.add_argument('--eval-freq', type=int, default=1000,
                        help='Evaluation frequency (default: 1000)')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Games per evaluation (default: 100)')
    parser.add_argument('--health-freq', type=int, default=500,
                        help='Health check frequency (default: 500)')
    parser.add_argument('--save-freq', type=int, default=5000,
                        help='Model save frequency (default: 5000)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Model save path (default: models/dqn_agent_YYYYMMDD_HHMM.pth)')
    parser.add_argument('--plot-path', type=str, default=None,
                        help='Plot save path (default: models/training_metrics_YYYYMMDD_HHMM.png)')
    
    args = parser.parse_args()
    
    # Train agent
    agent, metrics = train_dqn_agent(
        num_episodes=args.episodes,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games,
        health_check_freq=args.health_freq,
        save_freq=args.save_freq,
        model_save_path=args.model_path,
        plot_save_path=args.plot_path
    )
    
    print("\n✅ Training complete! Agent is ready to play.")


if __name__ == "__main__":
    main()