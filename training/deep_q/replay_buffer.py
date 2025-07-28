"""
Experience replay buffer for DQN training.

This module implements a simple replay buffer to store and sample training experiences.
"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Set, Dict
from ScotlandYard.core.game import TransportType


# Define experience tuple
Experience = namedtuple('Experience', [
    'state',           # Feature vector of current state
    'action',          # (destination, transport) tuple
    'reward',          # Immediate reward
    'next_state',      # Feature vector of next state
    'done',            # Whether episode ended
    'valid_moves',     # Valid moves from current state
    'next_valid_moves' # Valid moves from next state
])


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores game experiences and provides random sampling for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, 
             state: np.ndarray,
             action: Tuple[int, TransportType], 
             reward: float,
             next_state: np.ndarray,
             done: bool,
             valid_moves: Set[Tuple[int, TransportType]],
             next_valid_moves: Set[Tuple[int, TransportType]]):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state feature vector
            action: Action taken (destination, transport)
            reward: Reward received
            next_state: Next state feature vector
            done: Whether episode ended
            valid_moves: Valid moves from current state
            next_valid_moves: Valid moves from next state
        """
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            valid_moves=set(valid_moves),  # Make a copy
            next_valid_moves=set(next_valid_moves)
        )
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences randomly.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if we can sample a batch of given size."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.
    
    Samples experiences based on their TD error magnitude.
    This is a simplified implementation for the basic version.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, 
             state: np.ndarray,
             action: Tuple[int, TransportType], 
             reward: float,
             next_state: np.ndarray,
             done: bool,
             valid_moves: Set[Tuple[int, TransportType]],
             next_valid_moves: Set[Tuple[int, TransportType]],
             priority: Optional[float] = None):
        """
        Add experience with priority.
        
        Args:
            priority: Priority for this experience (uses max if None)
        """
        super().push(state, action, reward, next_state, done, valid_moves, next_valid_moves)
        
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritization.
        
        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for given indices.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)


def create_replay_buffer(config: Dict) -> ReplayBuffer:
    """
    Factory function to create replay buffer from configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Initialized replay buffer
    """
    training_params = config.get('training_parameters', {})
    buffer_size = training_params.get('replay_buffer_size', 10000)
    
    # For now, use simple replay buffer
    # Can extend to prioritized replay later
    return ReplayBuffer(capacity=buffer_size)
