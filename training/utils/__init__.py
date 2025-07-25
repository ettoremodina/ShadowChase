"""
Training utilities package.

This package contains utility classes and functions for training Scotland Yard AI agents.

Module responsibilities:
- training_utils: Core data structures and basic calculations
- training_environment: Game episode execution and experience collection  
- evaluation: Agent evaluation, comparison, and performance reporting
"""

from .training_utils import (
    GameResult,
    calculate_win_rate,
    calculate_average_game_length
)
from .training_environment import TrainingEnvironment
from .evaluation import (
    AgentEvaluator, 
    EvaluationConfig,
)

__all__ = [
    # Core data structures
    'GameResult',
    
    # Training environment
    'TrainingEnvironment',
    
    # Evaluation and reporting
    'AgentEvaluator',
    'EvaluationConfig',
    
    # Basic calculations
    'calculate_win_rate',
    'calculate_average_game_length'
]
