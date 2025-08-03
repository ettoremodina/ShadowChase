"""
Deep Q-Learning (DQN) training module.

This module contains DQN-specific training algorithms and agents for Shadow Chase.
"""

# Core DQN components
try:
    from .dqn_model import DQNModel, create_dqn_model
    from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, create_replay_buffer
    from .dqn_trainer import DQNTrainer
    
    __all__ = [
        'DQNModel', 
        'create_dqn_model',
        'ReplayBuffer',
        'PrioritizedReplayBuffer',
        'create_replay_buffer',
        'DQNTrainer'
    ]
    
except ImportError as e:
    print(f"Warning: DQN components not available due to missing dependencies: {e}")
    __all__ = []
