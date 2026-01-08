"""
DQN Replay Buffer - Experience Replay Memory for Deep Q-Learning

This module implements the replay buffer used in DQN (Deep Q-Network) algorithm.
The replay buffer stores past experiences and allows random sampling for training.

Key Concepts:
-------------
1. **Experience Replay**: Store transitions and sample randomly for training
2. **Breaking Correlation**: Random sampling breaks temporal correlation in data
3. **Sample Efficiency**: Reuse experiences multiple times for learning

Why Replay Buffer?
------------------
Without replay buffer:
- Agent learns from consecutive experiences (highly correlated)
- Forgets old experiences quickly
- Unstable learning

With replay buffer:
- Agent learns from random mix of old and new experiences
- Better sample efficiency (reuse data)
- More stable learning

Example:
--------
    buffer = DQNReplayBuffer(capacity=10000)
    
    # Store experience
    buffer.add(state, action, reward, next_state, done)
    
    # Sample batch for training
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, List

Transition = namedtuple( 'Transition', ('state', 'action', 'reward', 'next_state', 'done', 'next_mask'))

class DQNReplayBuffer:  
    """
    Replay buffer for storing and sampling experiences in DQN.
    
    The buffer stores transitions (s, a, r, s', done) and provides random
    sampling for training. It uses a deque with maximum length for automatic
    old experience removal.
    
    Attributes:
        capacity (int): Maximum number of experiences to store
        buffer (deque): Storage for experiences
        
    Experience Format:
        Each experience is a tuple: (state, action, reward, next_state, done)
        - state: np.ndarray of shape (3, 6, 7) - current state
        - action: int - action taken (column 0-6)
        - reward: float - reward received (1.0, -1.0, or 0.0)
        - next_state: np.ndarray of shape (3, 6, 7) - resulting state
        - done: bool - whether episode ended
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store.
                     When full, oldest experiences are automatically removed.
                     Typical values: 10,000 to 1,000,000
        Example:
            buffer = DQNReplayBuffer(capacity=50000)
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # Separate storage for terminal states to ensure balanced sampling
        self.terminal_buffer = deque(maxlen=capacity // 2)

        
    def add(self, 
            state: np.ndarray, 
            action: int, 
            reward: float,
            next_state: np.ndarray, 
            done: bool,
            next_mask: np.ndarray) -> None:
        """
        Add a new experience to the buffer.
        
        This method stores a single transition (s, a, r, s', done).
        If buffer is full, the oldest experience is automatically removed.
        """
        experience = Transition(state, action, reward, next_state, done, next_mask)
        self.buffer.append( experience )
        if done:
            self.terminal_buffer.append(experience)

    def add_symmetric(self, state, action, reward, next_state, done, next_mask):
        # 1. Add original
        self.add(state, action, reward, next_state, done, next_mask)
        
        # 2. Add Mirrored version
        # state is (2, 6, 7) or (3, 6, 7). We flip on the last dimension (width)
        state_mirrored = np.flip(state, axis=-1).copy()
        next_state_mirrored = np.flip(next_state, axis=-1).copy()
        
        # Flip action: 0 becomes 6, 1 becomes 5...
        action_mirrored = 6 - action
        
        # Flip mask: [1, 1, 0, 0, 0, 0, 0] becomes [0, 0, 0, 0, 0, 1, 1]
        next_mask_mirrored = np.flip(next_mask).copy()
        
        self.add(state_mirrored, action_mirrored, reward, next_state_mirrored, done, next_mask_mirrored)
    
    def update_penalty( self, index, new_reward, is_done ):
        #self.buffer[index] = self.buffer[index]._replace(reward = new_reward, done=is_done)
        if abs(index) > len(self.buffer):
            return

        # 1. Update the main buffer (inplace replacement)
        old_transition = self.buffer[index]
        new_transition = old_transition._replace(reward=float(new_reward), done=float(is_done))
        self.buffer[index] = new_transition
        
        # 2. Add to terminal buffer for balanced sampling
        if is_done:
            self.terminal_buffer.append(new_transition)

    def sample(self, batch_size: int, indices: list = None, terminal_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Augmented sample function.
        If indices are provided: returns those specific transitions.
        If indices are None: returns a balanced batch with terminal_ratio.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Not enough samples in buffer ({len(self.buffer)})")

        # Path A: Sample specific indices (Auditing/Debugging)
        if indices is not None:
            if len(indices) != batch_size:
                raise ValueError("Indices count must match batch_size")
            
            # Support negative indexing (like -1 for most recent)
            buffer_len = len(self.buffer)
            batch = [self.buffer[i if i >= 0 else buffer_len + i] for i in indices]
            
        # Path B: Balanced Sampling (Training)
        else:
            n_terminals = int(batch_size * terminal_ratio)
            
            # If we are early in training and don't have enough terminals, fall back to random
            if len(self.terminal_buffer) < n_terminals:
                batch = random.sample(self.buffer, batch_size)
            else:
                terminals = random.sample(self.terminal_buffer, n_terminals)
                non_terminals = random.sample(self.buffer, batch_size - n_terminals)
                batch = terminals + non_terminals
                random.shuffle(batch)

        # Path C: Consistent Unzipping and Conversion
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=np.int16)
        )
    
    def __len__(self) -> int:
        """
        Get current number of experiences in buffer.
        
        Returns:
            int: Number of experiences currently stored
        
        Example:
            buffer = DQNReplayBuffer(capacity=10000)
            print(f"Buffer size: {len(buffer)}")  # 0
            
            buffer.add(state, action, reward, next_state, done)
            print(f"Buffer size: {len(buffer)}")  # 1
        """
        return len(self.buffer)
    
    def clear(self) -> None:
        """
        Clear all experiences from the buffer.
        
        Useful for:
        - Starting fresh training
        - Resetting between experiments
        - Memory management
        
        Example:
            buffer.clear()
            print(f"Buffer size after clear: {len(buffer)}")  # 0
        """
        self.buffer.clear()
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences for training.
        
        Args:
            min_size: Minimum number of experiences needed (typically batch_size)
        
        Returns:
            bool: True if buffer has at least min_size experiences
        
        Example:
            # Wait until buffer has enough data
            while not buffer.is_ready(batch_size=32):
                # Collect more experiences
                play_one_step()
            
            # Now we can train
            batch = buffer.sample(32)
        """
        return len(self.buffer) >= min_size
    
    def get_stats(self) -> dict:
        """
        Get statistics about the buffer contents.
        
        Returns:
            dict: Statistics including size, capacity, and utilization
        
        Example:
            stats = buffer.get_stats()
            print(f"Buffer: {stats['size']}/{stats['capacity']} ({stats['utilization']:.1%})")
        """
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity if self.capacity > 0 else 0.0,
            'is_full': len(self.buffer) == self.capacity
        }
    
    def __repr__(self) -> str:
        """String representation of buffer."""
        return f"DQNReplayBuffer(size={len(self)}, capacity={self.capacity})"