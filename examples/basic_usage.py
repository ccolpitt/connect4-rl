"""
Basic usage example for Connect 4 environment.

This script demonstrates:
1. Creating and initializing the environment
2. Playing moves
3. Checking game state
4. Detecting winners
5. Rendering the board
"""
# %%

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# %%
from environment import Config, ConnectFourEnvironment

# %%


def example_game():
    """Play a simple example game."""
    print("=" * 50)
    print("Connect 4 Environment - Basic Usage Example")
    print("=" * 50)
    
    # Initialize environment
    config = Config()
    env = ConnectFourEnvironment(config)
    
    print(f"\n1. Environment initialized:")
    print(f"   Board size: {env.rows} x {env.cols}")
    print(f"   Current player: {env.current_player}")
    print(f"   Device: {config.DEVICE}")
    
    # Show initial state
    print("\n2. Initial board:")
    env.render()
    
    # Get state representation
    state = env.get_state()
    print(f"3. State shape: {state.shape}")
    print(f"   - Channel 0 (Player 1): {state[0].sum()} pieces")
    print(f"   - Channel 1 (Player 2): {state[1].sum()} pieces")
    print(f"   - Channel 2 (Current player): {'Player 1' if state[2, 0, 0] == 1 else 'Player 2'}")
    
    # Play some moves
    print("\n4. Playing moves:")
    moves = [3, 3, 4, 4, 5, 5, 6]  # This will create a horizontal win for Player 1
    
    for i, col in enumerate(moves):
        legal_moves = env.get_legal_moves()
        print(f"\n   Move {i+1}: Player {env.current_player} plays column {col}")
        print(f"   Legal moves: {legal_moves}")
        
        state, reward, done = env.play_move(col)
        env.render()
        
        if done:
            if reward == config.PLAYER_1:
                print(f"   🎉 Player 1 (X) wins!")
            elif reward == config.PLAYER_2:
                print(f"   🎉 Player 2 (O) wins!")
            else:
                print(f"   🤝 It's a draw!")
            break
    
    print("\n5. Final state:")
    print(f"   Winner: {env.check_winner()}")
    print(f"   Game over: {env.is_terminal()}")
    print(f"   Legal moves remaining: {env.get_legal_moves()}")


# %%

def example_state_manipulation():
    """Demonstrate state manipulation features."""
    print("\n" + "=" * 50)
    print("State Manipulation Example")
    print("=" * 50)
    
    config = Config()
    env = ConnectFourEnvironment(config)
    
    # Set up a specific board position
    import numpy as np
    board = np.array([
        [0,  0,  0,  0,  0,  0,  1],
        [0,  0,  0,  0,  0,  0,  1],
        [0, -1,  0,  0,  0,  0,  1],
        [0,  1,  0,  0,  0,  0, -1],
        [0, -1,  1, -1,  1, -1, -1],
        [0,  1, -1,  1, -1, -1,  1]
    ])
    
    print("\n1. Setting custom board position:")
    env.set_state(board, config.PLAYER_1)
    env.render()
    
    print("2. Checking for winner:")
    winner = env.check_winner_from_state(board)
    print(f"   Winner: {winner if winner else 'None'}")
    
    print("\n3. Getting legal moves:")
    legal_moves = env.get_legal_moves()
    print(f"   Legal moves: {legal_moves}")
    
    print("\n4. Simulating move without modifying environment:")
    state = env.get_state()
    new_state = env.apply_move_to_state(state, 5, config.PLAYER_1)
    print("   Original board (unchanged):")
    env.render()
    print("   Simulated board (with move in column 5):")
    env.render_ext_state(new_state)


def example_win_detection():
    """Demonstrate win detection in different directions."""
    print("\n" + "=" * 50)
    print("Win Detection Examples")
    print("=" * 50)
    
    config = Config()
    
    # Horizontal win
    print("\n1. Horizontal Win:")
    env = ConnectFourEnvironment(config)
    for col in [0, 0, 1, 1, 2, 2, 3]:  # P1 wins horizontally
        env.play_move(col)
    env.render()
    print(f"   Winner: Player {env.check_winner()}")
    
    # Vertical win
    print("\n2. Vertical Win:")
    env = ConnectFourEnvironment(config)
    for col in [3, 4, 3, 4, 3, 4, 3]:  # P1 wins vertically
        env.play_move(col)
    env.render()
    print(f"   Winner: Player {env.check_winner()}")
    
    # Diagonal win
    print("\n3. Diagonal Win (/):")
    env = ConnectFourEnvironment(config)
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 4, 3]  # P1 wins diagonally
    for col in moves:
        env.play_move(col)
    env.render()
    print(f"   Winner: Player {env.check_winner()}")


if __name__ == '__main__':
    # Run all examples
    example_game()
    example_state_manipulation()
    example_win_detection()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)
# %%
