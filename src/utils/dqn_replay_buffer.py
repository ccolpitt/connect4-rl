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

Transition = namedtuple( 'Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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
    
    def add(self, 
            state: np.ndarray, 
            action: int, 
            reward: float,
            next_state: np.ndarray, 
            done: bool) -> None:
        """
        Add a new experience to the buffer.
        
        This method stores a single transition (s, a, r, s', done).
        If buffer is full, the oldest experience is automatically removed.
        
        Args:
            state: Current state before action, shape (3, 6, 7)
            action: Action taken (column index 0-6)
            reward: Reward received (1.0 for win, -1.0 for loss, 0.0 for draw, None for continuing)
            next_state: Resulting state after action, shape (3, 6, 7)
            done: Whether the episode ended (True if game over)
        
        Example:
            # During gameplay
            state = env.get_state()
            action = agent.select_action(state, legal_moves)
            next_state, reward, done = env.play_move(action)
            buffer.add(state, action, reward, next_state, done)
        
        Note:
            - States are stored as-is (not copied), so make sure to pass copies if needed
            - Buffer automatically removes oldest when capacity is reached
        """
        # Named tuple - recommended default
        #experience = (state, action, reward, next_state, done)
        # Version with a dictionary -- not memory efficient, and harder to assemble into batches
        #experience = {"state":state,"action":action,"reward":reward,"next_state":next_state,"done":done}
        #self.buffer.append(experience)
        self.buffer.append( Transition( state, action, reward, next_state, done ) )
    
    def update_penalty( self, index, new_reward, is_done ):
        self.buffer[index] = self.buffer[index]._replace(reward = new_reward, done=is_done)

    def sample(self, batch_size: int, indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences for training.
        
        This method samples experiences from the buffer, either randomly or from
        specific indices (useful for auditing).
        
        Args:
            batch_size: Number of experiences to sample (typically 32, 64, or 128)
            indices: Optional list of specific indices to sample. If None, samples randomly.
                    Use negative indices to count from end (e.g., [-1] for most recent)
        
        Returns:
            Tuple of numpy arrays:
            - states: (batch_size, 3, 6, 7) - batch of states
            - actions: (batch_size,) - batch of actions
            - rewards: (batch_size,) - batch of rewards
            - next_states: (batch_size, 3, 6, 7) - batch of next states
            - dones: (batch_size,) - batch of done flags
        
        Raises:
            ValueError: If batch_size > len(buffer) or invalid indices
        
        Example:
            # Random sampling (normal training)
            states, actions, rewards, next_states, dones = buffer.sample(32)
            
            # Sample most recent experience (auditing)
            states, actions, rewards, next_states, dones = buffer.sample(1, indices=[-1])
            
            # Sample specific experiences
            states, actions, rewards, next_states, dones = buffer.sample(3, indices=[0, 5, 10])
        
        Note:
            - Random sampling is with replacement
            - Specific indices sampling is without replacement
            - Returns numpy arrays ready for PyTorch/TensorFlow
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer with only {len(self.buffer)} experiences. "
                f"Wait until buffer has at least {batch_size} experiences."
            )
        
        if indices is not None:
            # Sample specific indices
            if len(indices) != batch_size:
                raise ValueError(f"Number of indices ({len(indices)}) must match batch_size ({batch_size})")
            
            # Convert negative indices to positive
            buffer_len = len(self.buffer)
            positive_indices = [(i if i >= 0 else buffer_len + i) for i in indices]
            
            # Validate indices
            for i in positive_indices:
                if i < 0 or i >= buffer_len:
                    raise ValueError(f"Index {i} out of range for buffer of size {buffer_len}")
            
            # Get experiences at specific indices
            batch = [self.buffer[i] for i in positive_indices]
        else:
            # Randomly sample batch_size experiences
            batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate arrays
        #print( "We calculated the batch")
        states, actions, rewards, next_states, dones = zip(*batch)
        #print( "We broke out parts of the batch")
        #print( "States in replay buffer: ")
        #print( states )
        #print( "Full Batch")
        #print( batch )


        # Convert to numpy arrays for efficient computation
        states = np.array(states, dtype=np.float32)           # (batch_size, 3, 6, 7)
        actions = np.array(actions, dtype=np.int64)           # (batch_size,)
        rewards = np.array(rewards, dtype=np.float32)         # (batch_size,)
        next_states = np.array(next_states, dtype=np.float32) # (batch_size, 3, 6, 7)
        dones = np.array(dones, dtype=np.float32)             # (batch_size,) - 1.0 if done, 0.0 if not
        
        return states, actions, rewards, next_states, dones
    
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