"""
Interactive playground for testing the Connect 4 environment.

Run this file in the Interactive Window (add # %% cells) or
run it directly to see examples.
"""

# %%
import sys
sys.path.insert(0, 'src')
from environment import Config, ConnectFourEnvironment

# %%
# Create environment
config = Config()
env = ConnectFourEnvironment(config)
print("Environment created!")
env.render()

# %%
# Check initial state
state = env.get_state()
print(f"State shape: {state.shape}")
print(f"Legal moves: {env.get_legal_moves()}")

# %%
# Play a move
env.play_move(3)
print("Played column 3:")
env.render()

# %%
# Play another move
env.play_move(3)
print("Played column 3 again:")
env.render()

# %%
# Check current player
print(f"Current player: {env.current_player}")
print(f"Legal moves: {env.get_legal_moves()}")

# %%
# Try your own code here!
# For example:
# env.play_move(4)
