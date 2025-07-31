"""
Training package for Scotland Yard advanced AI agents.

This package contains infrastructure for training sophisticated agents using
Monte Carlo Tree Search (MCTS) and Deep Q-Learning approaches.

Package structure:
- base_trainer: Abstract base class for all training algorithms
- feature_extractor: Game state to feature vector conversion
- utils/: Training utilities, environment, and evaluation tools
- mcts/: Monte Carlo Tree Search implementation (future)
- deep_q/: Deep Q-Learning implementation (future)
- configs/: Configuration files for different algorithms
"""

# Main training infrastructure
from .base_trainer import BaseTrainer, TrainingResult, EvaluationResult
from .feature_extractor_simple import GameFeatureExtractor 

# Training utilities (accessed via training.utils.*)
from . import utils

__all__ = [
    # Core training infrastructure
    'BaseTrainer',
    'TrainingResult', 
    'EvaluationResult',
    'GameFeatureExtractor',
    'FeatureConfig',
    
    # Utilities subpackage
    'utils'
]
