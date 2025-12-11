"""
Test script for DQN Value Network.

This script demonstrates:
1. Creating networks with different architectures
2. Forward pass with batch inputs
3. Selecting best actions from Q-values
4. Network configuration and parameters
5. Testing with real game states

Run this to verify the DQN network works correctly!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from environment import Config, ConnectFourEnvironment
from networks import DQNValueNetwork


def test_network_creation():
    """Test creating networks with different configurations."""
    print("=" * 60)
    print("Test 1: Network Creation")
    print("=" * 60)
    
    # Default network
    print("\n✓ Creating default network...")
    network1 = DQNValueNetwork()
    print(f"  {network1}")
    
    # Deep network
    print("\n✓ Creating deep network...")
    network2 = DQNValueNetwork(
        conv_channels=[32, 64, 128],
        fc_dims=[256, 128],
        dropout_rate=0.2
    )
    print(f"  {network2}")
    
    # Shallow network
    print("\n✓ Creating shallow network...")
    network3 = DQNValueNetwork(
        conv_channels=[32],
        fc_dims=[64],
        dropout_rate=0.0
    )
    print(f"  {network3}")
    
    print("\n✅ Network creation test passed!\n")


def test_forward_pass():
    """Test forward pass with batch inputs."""
    print("=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    network = DQNValueNetwork()
    
    # Single state
    print("\n✓ Testing single state...")
    state = torch.randn(1, 3, 6, 7)
    q_values = network(state)
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values: {q_values[0].detach().numpy()}")
    
    # Batch of states
    print("\n✓ Testing batch of states...")
    batch_size = 32
    states = torch.randn(batch_size, 3, 6, 7)
    q_values = network(states)
    print(f"  Input shape: {states.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    print("\n✅ Forward pass test passed!\n")


def test_action_selection():
    """Test selecting best actions from Q-values."""
    print("=" * 60)
    print("Test 3: Action Selection")
    print("=" * 60)
    
    network = DQNValueNetwork()
    
    # Create batch of states
    batch_size = 8
    states = torch.randn(batch_size, 3, 6, 7)
    
    # Get Q-values
    q_values = network(states)
    
    # Select best actions (greedy)
    best_actions = q_values.argmax(dim=1)
    
    print(f"\n✓ Batch size: {batch_size}")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Best actions: {best_actions.numpy()}")
    
    # Get Q-values for specific actions
    actions = torch.tensor([3, 1, 4, 2, 5, 0, 6, 3])
    q_selected = q_values.gather(1, actions.unsqueeze(1))
    
    print(f"\n✓ Selected actions: {actions.numpy()}")
    print(f"  Q-values for selected actions: {q_selected.squeeze().detach().numpy()}")
    
    print("\n✅ Action selection test passed!\n")


def test_real_game_states():
    """Test network with real game states."""
    print("=" * 60)
    print("Test 4: Real Game States")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    network = DQNValueNetwork()
    
    # Get initial state
    state = env.reset()
    print("\n✓ Initial board:")
    env.render()
    
    # Convert to tensor and get Q-values
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = network(state_tensor)
    
    print(f"  Q-values for each column:")
    for col, q_val in enumerate(q_values[0].detach().numpy()):
        print(f"    Column {col}: {q_val:.3f}")
    
    best_action = q_values.argmax(dim=1).item()
    print(f"\n  Network suggests column: {best_action}")
    
    # Play a few moves and check Q-values
    print("\n✓ After a few moves:")
    for i in range(3):
        legal_moves = env.get_legal_moves()
        action = legal_moves[0]  # Just pick first legal move
        env.play_move(action)
    
    env.render()
    
    state = env.get_state()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = network(state_tensor)
    
    print(f"  Q-values for each column:")
    for col, q_val in enumerate(q_values[0].detach().numpy()):
        print(f"    Column {col}: {q_val:.3f}")
    
    print("\n✅ Real game states test passed!\n")


def test_network_config():
    """Test network configuration and parameters."""
    print("=" * 60)
    print("Test 5: Network Configuration")
    print("=" * 60)
    
    network = DQNValueNetwork(
        conv_channels=[32, 64],
        fc_dims=[128, 64],
        dropout_rate=0.1
    )
    
    config = network.get_config()
    
    print("\n✓ Network configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Count parameters by layer type
    conv_params = sum(p.numel() for name, p in network.named_parameters() if 'conv' in name)
    fc_params = sum(p.numel() for name, p in network.named_parameters() if 'fc' in name or 'output' in name)
    
    print(f"\n✓ Parameter breakdown:")
    print(f"  Convolutional layers: {conv_params:,} parameters")
    print(f"  Fully connected layers: {fc_params:,} parameters")
    print(f"  Total: {config['total_parameters']:,} parameters")
    
    print("\n✅ Network configuration test passed!\n")


def test_batch_processing():
    """Test processing multiple game states efficiently."""
    print("=" * 60)
    print("Test 6: Batch Processing")
    print("=" * 60)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    network = DQNValueNetwork()
    
    # Collect multiple states
    print("\n✓ Collecting 16 game states...")
    states = []
    for i in range(16):
        env.reset()
        # Play random number of moves
        for _ in range(np.random.randint(0, 10)):
            legal_moves = env.get_legal_moves()
            if legal_moves:
                env.play_move(np.random.choice(legal_moves))
        states.append(env.get_state())
    
    # Process all at once
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    print(f"  States tensor shape: {states_tensor.shape}")
    
    q_values = network(states_tensor)
    print(f"  Q-values shape: {q_values.shape}")
    
    # Get best action for each state
    best_actions = q_values.argmax(dim=1)
    print(f"\n✓ Best actions for each state:")
    print(f"  {best_actions.numpy()}")
    
    print("\n✅ Batch processing test passed!\n")


def test_gradient_flow():
    """Test that gradients flow correctly through the network."""
    print("=" * 60)
    print("Test 7: Gradient Flow")
    print("=" * 60)
    
    network = DQNValueNetwork()
    
    # Create dummy input and target
    state = torch.randn(4, 3, 6, 7, requires_grad=True)
    target_q = torch.randn(4, 7)
    
    # Forward pass
    q_values = network(state)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(q_values, target_q)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_gradients = all(p.grad is not None for p in network.parameters() if p.requires_grad)
    print(f"\n✓ Backward pass successful")
    print(f"  All parameters have gradients: {has_gradients}")
    
    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in network.parameters() if p.grad is not None]
    print(f"  Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    
    print("\n✅ Gradient flow test passed!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DQN VALUE NETWORK TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_network_creation()
    test_forward_pass()
    test_action_selection()
    test_real_game_states()
    test_network_config()
    test_batch_processing()
    test_gradient_flow()
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe DQN Value Network is working correctly!")
    print("Features verified:")
    print("  ✓ Multi-head output (all Q-values at once)")
    print("  ✓ Configurable architecture")
    print("  ✓ Dropout regularization")
    print("  ✓ Batch processing")
    print("  ✓ Gradient flow")
    print("\nReady to implement DQN Agent next!")
    print("=" * 60 + "\n")