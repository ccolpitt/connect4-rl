"""
Prioritized Experience Replay Buffer for DQN

This module implements Prioritized Experience Replay (PER) as described in:
Schaul et al. (2016) "Prioritized Experience Replay"

Key Concepts:
-------------
1. **TD-Error Priority**: Sample experiences based on how much we can learn from them
2. **Importance Sampling**: Correct bias introduced by non-uniform sampling
3. **Sum Tree**: Efficient data structure for proportional sampling

Why Prioritized Replay?
-----------------------
Standard replay buffer samples uniformly (all experiences equally likely).
Prioritized replay samples based on TD-error: |Q(s,a) - target|

Benefits:
- Learn more from important/surprising transitions
- Better sample efficiency (learn faster)
- Focus on mistakes (high TD-error = we were wrong)

Example:
--------
    buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta_start=0.4)
    
    # Store experience with initial priority
    buffer.add(state, action, reward, next_state, done)
    
    # Sample batch (returns indices for priority updates)
    batch, indices, weights = buffer.sample(batch_size=32)
    
    # After computing TD-errors, update priorities
    buffer.update_priorities(indices, td_errors)
"""

import numpy as np
from typing import Tuple, List
from collections import deque


class SumTree:
    """
    Sum Tree data structure for efficient proportional sampling.
    
    A binary tree where:
    - Leaf nodes store priorities
    - Parent nodes store sum of children
    - Root stores total sum of all priorities
    
    This allows O(log n) sampling and O(log n) priority updates.
    
    Structure:
              [sum of all]
             /            \\
        [sum left]    [sum right]
         /      \\        /      \\
      [p1]    [p2]    [p3]    [p4]  <- leaf priorities
    
    Attributes:
        capacity: Maximum number of leaf nodes
        tree: Array storing tree nodes (size 2*capacity - 1)
        data: Array storing actual experiences (size capacity)
        write_idx: Current write position in circular buffer
        n_entries: Current number of stored experiences
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum tree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        # Tree array: [parent nodes | leaf nodes]
        # Size: 2*capacity - 1 (capacity-1 parents + capacity leaves)
        self.tree = np.zeros(2 * capacity - 1)
        # Data array: stores actual experiences
        self.data = np.zeros(capacity, dtype=object)
        # Write position (circular buffer)
        self.write_idx = 0
        # Number of stored experiences
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate priority change up the tree.
        
        When a leaf priority changes, update all parent nodes.
        
        Args:
            idx: Tree index that changed
            change: Amount of change in priority
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve leaf index for a given cumulative sum.
        
        Traverse tree to find leaf where cumulative sum falls.
        
        Args:
            idx: Current node index
            s: Target cumulative sum
        
        Returns:
            Leaf index containing the target sum
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If leaf node, return it
        if left >= len(self.tree):
            return idx
        
        # If target sum is in left subtree
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # Otherwise it's in right subtree
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total sum of all priorities (root node)."""
        return self.tree[0]
    
    def add(self, priority: float, data: object) -> None:
        """
        Add new experience with given priority.
        
        Args:
            priority: Priority value (typically TD-error)
            data: Experience tuple (state, action, reward, next_state, done)
        """
        # Calculate tree index for this leaf
        idx = self.write_idx + self.capacity - 1
        
        # Store data
        self.data[self.write_idx] = data
        
        # Update tree with new priority
        self.update(idx, priority)
        
        # Move write pointer (circular)
        self.write_idx = (self.write_idx + 1) % self.capacity
        
        # Track number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float) -> None:
        """
        Update priority of a leaf node.
        
        Args:
            idx: Tree index to update
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Get experience for a given cumulative sum.
        
        Args:
            s: Target cumulative sum (between 0 and total())
        
        Returns:
            Tuple of (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for DQN.
    
    Samples experiences based on their TD-error (priority), allowing the agent
    to learn more from important/surprising transitions.
    
    Key Parameters:
        alpha: How much prioritization to use (0 = uniform, 1 = full priority)
        beta: Importance sampling correction (0 = no correction, 1 = full correction)
        
    Attributes:
        capacity: Maximum number of experiences
        alpha: Prioritization exponent
        beta: Importance sampling exponent
        beta_increment: How much to increase beta per sample
        epsilon: Small constant to ensure non-zero priorities
        tree: Sum tree for efficient sampling
        max_priority: Track maximum priority for new experiences
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
                   Typical: 0.6-0.7
            beta_start: Initial importance sampling exponent (0 = no correction)
                       Typical: 0.4, annealed to 1.0
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant added to priorities to ensure non-zero
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = epsilon
        
        # Sum tree for efficient sampling
        self.tree = SumTree(capacity)
        
        # Track maximum priority for new experiences
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add experience with maximum priority.
        
        New experiences get max priority to ensure they're sampled at least once.
        
        Args:
            state: Current state, shape (3, 6, 7)
            action: Action taken (0-6)
            reward: Reward received
            next_state: Next state, shape (3, 6, 7)
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        
        # New experiences get max priority
        priority = self.max_priority
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[Tuple, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences based on priorities.
        
        Returns experiences, their indices (for priority updates), and
        importance sampling weights (to correct bias).
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of:
            - batch: (states, actions, rewards, next_states, dones)
            - indices: Tree indices for priority updates
            - weights: Importance sampling weights
        """
        if self.tree.n_entries < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer with only "
                f"{self.tree.n_entries} experiences."
            )
        
        # Arrays to store batch
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Divide priority range into segments
        segment = self.tree.total() / batch_size
        
        # Sample one experience from each segment
        for i in range(batch_size):
            # Random value in this segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            # Get experience for this sum
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices[i] = idx
            priorities[i] = priority
        
        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta) / max_w
        sampling_probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()  # Normalize by max weight
        
        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD-errors.
        
        Priority = (|TD-error| + epsilon)^alpha
        
        Args:
            indices: Tree indices from sample()
            td_errors: TD-errors for each sampled experience
        """
        for idx, td_error in zip(indices, td_errors):
            # Priority = (|TD-error| + epsilon)^alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        """Get current number of experiences in buffer."""
        return self.tree.n_entries
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return self.tree.n_entries >= min_size
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': self.tree.n_entries,
            'capacity': self.capacity,
            'utilization': self.tree.n_entries / self.capacity,
            'is_full': self.tree.n_entries == self.capacity,
            'max_priority': self.max_priority,
            'beta': self.beta
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"PrioritizedReplayBuffer(size={len(self)}, capacity={self.capacity}, "
                f"alpha={self.alpha:.2f}, beta={self.beta:.2f})")