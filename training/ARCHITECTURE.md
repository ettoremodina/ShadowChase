# Training Infrastructure Architecture

## Overview

The training infrastructure has been refactored to eliminate redundancy and provide clear separation of concerns. Here's the clean architecture:

## Package Structure

```
training/
├── __init__.py                    # Main package exports
├── base_trainer.py               # Abstract base class for all trainers
├── feature_extractor.py          # Game state to feature vector conversion
├── utils/                        # Training utilities subpackage
│   ├── __init__.py               # Utilities package exports
│   ├── training_utils.py         # Core data structures and basic calculations
│   ├── training_environment.py   # Game episode execution
│   └── evaluation.py             # Agent evaluation and performance reporting
├── mcts/                         # Future MCTS implementation
├── deep_q/                       # Future DQN implementation
└── configs/                      # Configuration files
```

## Responsibilities by Module

### `base_trainer.py`
- **Single Responsibility**: Abstract interface for all training algorithms
- **Key Methods**:
  - `train()`: Abstract method for training implementation
  - `get_trained_agent()`: Abstract method to retrieve trained agents
  - `evaluate()`: Delegates to evaluation utilities
  - `save_model()` / `load_model()`: Model persistence (to be overridden)

### `feature_extractor.py`
- **Single Responsibility**: Convert game states to ML-ready feature vectors
- **Key Classes**:
  - `FeatureConfig`: Configuration for feature extraction
  - `GameFeatureExtractor`: Main feature extraction logic

### `utils/training_utils.py`
- **Single Responsibility**: Core data structures and basic calculations
- **Contents**:
  - `GameResult`: Dataclass for game outcomes
  - `calculate_win_rate()`: Basic win rate calculation
  - `calculate_average_game_length()`: Basic game length calculation

### `utils/training_environment.py`
- **Single Responsibility**: Game episode execution and experience collection
- **Key Class**:
  - `TrainingEnvironment`: Runs single episodes and batches of games
  - Methods for collecting training experience
  - Integration with existing game infrastructure

### `utils/evaluation.py`
- **Single Responsibility**: Agent evaluation, comparison, and performance reporting
- **Key Classes**:
  - `AgentEvaluator`: Comprehensive agent evaluation
  - `EvaluationConfig`: Configuration for evaluation
- **Key Functions**:
  - `print_training_statistics()`: Moved here from training_utils

## Clean API Usage

### Basic Usage
```python
# Import main training components
from training import BaseTrainer, GameFeatureExtractor, FeatureConfig

# Import utilities as needed
from training.utils import TrainingEnvironment, AgentEvaluator

# Create feature extractor
config = FeatureConfig()
extractor = GameFeatureExtractor(config)

# Create training environment
env = TrainingEnvironment(map_size="test", num_detectives=2)

# Create evaluator
evaluator = AgentEvaluator()
```

### Implementing a Trainer
```python
from training import BaseTrainer, TrainingResult

class MyTrainer(BaseTrainer):
    def train(self, num_episodes, **kwargs):
        # Use TrainingEnvironment for episode execution
        # Use GameFeatureExtractor for state representation
        # Return TrainingResult
        pass
    
    def get_trained_agent(self, player):
        # Return trained agent instance
        pass
```

## Eliminated Redundancies

1. **Game Execution**: Now centralized in `TrainingEnvironment`
2. **Statistical Calculations**: Basic functions in `training_utils`, complex reporting in `evaluation`
3. **Agent Management**: Handled by `AgentEvaluator` for evaluation scenarios
4. **Performance Metrics**: Comprehensive reporting moved to `evaluation` module

## Benefits

1. **Single Responsibility**: Each module has one clear purpose
2. **No Duplication**: Functionality exists in exactly one place
3. **Clean Dependencies**: Clear dependency flow without circular imports
4. **Easy Testing**: Each component can be tested independently
5. **Future Extensibility**: Easy to add new training algorithms following the same pattern

## Import Pattern

- **Core infrastructure**: `from training import ...`
- **Utilities**: `from training.utils import ...`
- **Specific algorithms**: `from training.mcts import ...` (future)

This architecture follows the principle of "composition over inheritance" and ensures that each piece of functionality has a single, well-defined home.
