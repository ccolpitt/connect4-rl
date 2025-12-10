"""
Test script for DQN Replay Buffer.

This script demonstrates the replay buffer functionality:
1. Creating a buffer
2. Adding experiences
3. Sampling batches
4. Buffer statistics

Run this to verify the replay buffer works correctly!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import random
import numpy as np
from environment import Config, ConnectFourEnvironment
from utils import DQNReplayBuffer


def test_basic_operations():
    """Test basic buffer operations."""
    print("=" * 60)
    print("Test 1: Basic Buffer Operations")
    print("=" * 60)
    
    # Create buffer
    buffer = DQNReplayBuffer(capacity=100)
    print(f"\n✓ Created buffer: {buffer}")
    print(f"  Initial stats: {buffer.get_stats()}")
    
    # Create environment
    config = Config()
    env = ConnectFourEnvironment(config)
    
    # Add one experience manually
    state = env.reset()
    action = 3
    next_state, reward, done = env.play_move(action)
    buffer.add(state, action, reward, next_state, done)
    
    print(f"\n✓ Added one experience")
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Action: {action}, Reward: {reward}, Done: {done}")
    
    print("\n✅ Basic operations test passed!\n")


def test_sampling():
    """Test batch sampling."""
    print("=" * 60)
    print("Test 2: Batch Sampling")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    buffer = DQNReplayBuffer(capacity=1000)
    
    # Add 20 experiences
    print("\n✓ Adding 20 experiences...")
    for i in range(20):
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        
        if not legal_moves:
            env.reset()
            state = env.get_state()
            legal_moves = env.get_legal_moves()
        
        action = random.choice(legal_moves)
        next_state, reward, done = env.play_move(action)
        buffer.add(state, action, reward, next_state, done)
        
        if done:
            env.reset()
    
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample a batch
    batch_size = 8
    print(f"\n✓ Sampling batch of {batch_size}...")
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")
    print(f"\n  Sample actions: {actions}")
    print(f"  Sample rewards: {rewards}")
    
    print("\n✅ Sampling test passed!\n")


def test_capacity():
    """Test buffer capacity and overflow."""
    print("=" * 60)
    print("Test 3: Buffer Capacity")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    buffer = DQNReplayBuffer(capacity=50)
    
    print(f"\n✓ Buffer capacity: {buffer.capacity}")
    
    # Fill buffer beyond capacity
    print(f"✓ Adding 60 experiences (more than capacity)...")
    for i in range(60):
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        
        if not legal_moves:
            env.reset()
            state = env.get_state()
            legal_moves = env.get_legal_moves()
        
        action = random.choice(legal_moves)
        next_state, reward, done = env.play_move(action)
        buffer.add(state, action, reward, next_state, done)
        
        if done:
            env.reset()
    
    stats = buffer.get_stats()
    print(f"\n  Final buffer size: {stats['size']}")
    print(f"  Capacity: {stats['capacity']}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Is full: {stats['is_full']}")
    
    assert len(buffer) == buffer.capacity, "Buffer should be at capacity!"
    print("\n✅ Capacity test passed! Buffer correctly limited to capacity.\n")


def test_is_ready():
    """Test is_ready functionality."""
    print("=" * 60)
    print("Test 4: is_ready() Method")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    buffer = DQNReplayBuffer(capacity=100)
    
    batch_size = 32
    print(f"\n✓ Checking if buffer ready for batch_size={batch_size}")
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Is ready: {buffer.is_ready(batch_size)}")
    
    # Add experiences until ready
    print(f"\n✓ Adding experiences until ready...")
    while not buffer.is_ready(batch_size):
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        
        if not legal_moves:
            env.reset()
            state = env.get_state()
            legal_moves = env.get_legal_moves()
        
        action = random.choice(legal_moves)
        next_state, reward, done = env.play_move(action)
        buffer.add(state, action, reward, next_state, done)
        
        if done:
            env.reset()
    
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Is ready: {buffer.is_ready(batch_size)}")
    
    # Now we can sample
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    print(f"\n✓ Successfully sampled batch of {batch_size}")
    
    print("\n✅ is_ready() test passed!\n")


def test_clear():
    """Test buffer clearing."""
    print("=" * 60)
    print("Test 5: Clear Buffer")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    buffer = DQNReplayBuffer(capacity=100)
    
    # Add some experiences
    for i in range(10):
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            env.reset()
            state = env.get_state()
            legal_moves = env.get_legal_moves()
        
        action = random.choice(legal_moves)
        next_state, reward, done = env.play_move(action)
        buffer.add(state, action, reward, next_state, done)
        
        if done:
            env.reset()
    
    print(f"\n✓ Buffer size before clear: {len(buffer)}")
    
    buffer.clear()
    
    print(f"✓ Buffer size after clear: {len(buffer)}")
    assert len(buffer) == 0, "Buffer should be empty after clear!"
    
    print("\n✅ Clear test passed!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DQN REPLAY BUFFER TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_basic_operations()
    test_sampling()
    test_capacity()
    test_is_ready()
    test_clear()
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe DQN Replay Buffer is working correctly!")
    print("Ready to implement DQN Value Network next.")
    print("=" * 60 + "\n")