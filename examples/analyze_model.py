"""
Analyze DQN Model - Detailed Network Statistics

This script loads a trained DQN model and provides detailed analysis of:
- Network architecture
- Parameter counts and sizes
- Weight statistics
- Dead/exploding neuron analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from agents import DQNAgent


def analyze_model(model_path: str):
    """Analyze a trained DQN model."""
    
    print("=" * 80)
    print("DQN MODEL ANALYSIS")
    print("=" * 80)
    
    # Load model
    print(f"\n📂 Loading model from: {model_path}")
    agent = DQNAgent(name="DQN-Analysis")
    agent.load(model_path)
    
    print(f"✓ Model loaded successfully")
    print(f"  Training steps: {agent.train_steps:,}")
    print(f"  Episodes completed: {agent.episodes_completed:,}")
    print(f"  Current epsilon: {agent.epsilon:.4f}")
    
    # Network architecture
    print(f"\n🏗️  Network Architecture:")
    print(f"  Device: {agent.device}")
    
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory (weights only): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Layer-by-layer analysis
    print(f"\n📊 Layer-by-Layer Analysis:")
    print(f"  {'Layer':<30} {'Shape':<20} {'Parameters':<15} {'Dead %':<10} {'Exploding %'}")
    print(f"  {'-'*30} {'-'*20} {'-'*15} {'-'*10} {'-'*12}")
    
    dead_threshold = 1e-3
    exploding_threshold = 100.0
    
    total_dead = 0
    total_exploding = 0
    
    for name, param in agent.q_network.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            num_params = weights.size
            
            # Count dead and exploding
            dead_count = np.sum(np.abs(weights) < dead_threshold)
            exploding_count = np.sum(np.abs(weights) > exploding_threshold)
            
            total_dead += dead_count
            total_exploding += exploding_count
            
            dead_pct = 100.0 * dead_count / num_params
            exploding_pct = 100.0 * exploding_count / num_params
            
            shape_str = str(list(param.shape))
            
            print(f"  {name:<30} {shape_str:<20} {num_params:<15,} {dead_pct:<10.2f} {exploding_pct:.2f}")
    
    # Overall statistics
    print(f"\n📈 Overall Weight Statistics:")
    
    all_weights = []
    for param in agent.q_network.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.cpu().numpy().flatten())
    
    all_weights = np.array(all_weights)
    
    print(f"  Total weights: {len(all_weights):,}")
    print(f"  Mean: {np.mean(all_weights):.6f}")
    print(f"  Std: {np.std(all_weights):.6f}")
    print(f"  Min: {np.min(all_weights):.6f}")
    print(f"  Max: {np.max(all_weights):.6f}")
    print(f"  Median: {np.median(all_weights):.6f}")
    
    # Dead/exploding summary
    dead_pct_total = 100.0 * total_dead / len(all_weights)
    exploding_pct_total = 100.0 * total_exploding / len(all_weights)
    
    print(f"\n🔍 Network Health:")
    print(f"  Dead neurons (|w| < {dead_threshold}): {total_dead:,} ({dead_pct_total:.2f}%)")
    print(f"  Exploding weights (|w| > {exploding_threshold}): {total_exploding:,} ({exploding_pct_total:.2f}%)")
    print(f"  Active neurons: {len(all_weights) - total_dead:,} ({100 - dead_pct_total:.2f}%)")
    
    # Health assessment
    print(f"\n✅ Health Assessment:")
    if dead_pct_total < 5:
        print(f"  ✓ Dead neurons: HEALTHY ({dead_pct_total:.2f}% < 5%)")
    elif dead_pct_total < 20:
        print(f"  ⚠️  Dead neurons: MODERATE ({dead_pct_total:.2f}% between 5-20%)")
    else:
        print(f"  ❌ Dead neurons: CONCERNING ({dead_pct_total:.2f}% > 20%)")
    
    if exploding_pct_total < 1:
        print(f"  ✓ Exploding weights: HEALTHY ({exploding_pct_total:.2f}% < 1%)")
    else:
        print(f"  ⚠️  Exploding weights: NEEDS ATTENTION ({exploding_pct_total:.2f}% > 1%)")
    
    # Weight distribution
    print(f"\n📊 Weight Distribution:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"  Percentile: Value")
    for p in percentiles:
        val = np.percentile(all_weights, p)
        print(f"  {p:>3}%: {val:>10.6f}")
    
    # File size analysis
    file_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"\n💾 File Size Analysis:")
    print(f"  Total file size: {file_size:.2f} MB")
    print(f"  Weights only: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  Optimizer state: {file_size - (total_params * 4 / 1024 / 1024):.2f} MB")
    print(f"  (Adam stores 2x params for momentum + variance)")
    
    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze DQN model')
    parser.add_argument('--model', type=str, default='models/dqn_agent.pth',
                        help='Path to model file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    analyze_model(args.model)