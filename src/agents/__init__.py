"""
Agents module - RL agents for Connect 4.

This module contains all reinforcement learning agents that can play Connect 4.
All agents inherit from BaseAgent and implement the same interface.

The __all__ variable controls what gets exported when someone does:
    from agents import *

It's a Python convention that defines the "public API" of a module.

Example:
    # Without __all__, you'd need to import like this:
    from agents.base_agent import BaseAgent
    from agents.random_agent import RandomAgent
    
    # With __all__, you can import like this:
    from agents import BaseAgent, RandomAgent
    
    # Or even:
    from agents import *  # imports everything in __all__

Why use __all__?
1. **Clean imports**: Users don't need to know the internal file structure
2. **Public API**: Clearly defines what's meant to be used externally
3. **IDE support**: Better autocomplete and documentation
4. **Namespace control**: Prevents accidental imports of internal helpers
"""

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .dqn_agent import DQNAgent

# This list defines what gets exported when someone imports from this module
# Only classes/functions listed here will be available via "from agents import *"
__all__ = ['BaseAgent', 'RandomAgent', 'DQNAgent']