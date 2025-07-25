# Scotland Yard Training Infrastructure

This package provides the foundation for training advanced AI agents to play Scotland Yard using sophisticated algorithms like Monte Carlo Tree Search (MCTS) and Deep Q-Learning (DQN).

## Overview

The training infrastructure is designed to be modular and extensible, allowing for easy implementation of different training algorithms while sharing common components like feature extraction and evaluation.

## Directory Structure

```
training/
├── __init__.py                 # Package initialization
├── base_trainer.py            # Abstract base trainer class
├── feature_extractor.py       # Game state to feature vector conversion
├── mcts/                      # Monte Carlo Tree Search implementation
│   └── __init__.py
├── deep_q/                    # Deep Q-Learning implementation
│   └── __init__.py
├── utils/                     # Training utilities and helpers
│   └── __init__.py
└── configs/                   # Configuration files
    ├── mcts_config.json       # MCTS hyperparameters
    └── dqn_config.json        # DQN hyperparameters
```

## Core Components

### BaseTrainer

The `BaseTrainer` class provides a common interface for all training algorithms:

```python
from training.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def train(self, num_episodes, **kwargs):
        # Implement training logic
        pass
    
    def get_trained_agent(self, player):
        # Return trained agent
        pass
```

### GameFeatureExtractor

Converts complex game states into numerical feature vectors for machine learning:

```python
from training.feature_extractor import GameFeatureExtractor, FeatureConfig

# Create feature extractor
config = FeatureConfig(include_distances=True, include_tickets=True)
extractor = GameFeatureExtractor(config)

# Extract features
features = extractor.extract_features(game, Player.ROBBER)
```

### TrainingEnvironment

Provides a standardized environment for running training episodes:

```python
from training.utils import TrainingEnvironment

env = TrainingEnvironment(map_size="test", num_detectives=2)
result, experience = env.run_episode(mr_x_agent, detective_agent)
```

## Feature Vector Components

The feature extractor generates vectors containing:

1. **Game Phase Features** (5 dimensions)
   - Current turn progress
   - Mr. X visibility status
   - Reveal turn indicators
   - Game phase (early/mid/late)

2. **Board State Features** (up to 200 dimensions)
   - One-hot encoding of piece positions
   - Different values for Mr. X vs detectives

3. **Distance Features** (18 dimensions)
   - Distances between Mr. X and each detective
   - Reachability and threat level indicators
   - Statistical distance measures

4. **Ticket Features** (20 dimensions)
   - Remaining tickets for all players
   - Transport type availability
   - Special move capabilities

5. **Transport Connectivity** (12 dimensions)
   - Available moves by transport type
   - Connectivity and efficiency metrics

6. **Possible Positions** (3 dimensions)
   - Analysis of Mr. X's possible locations when hidden
   - Minimum distances to possible positions

## Usage Example

```python
# Test the infrastructure
python test_training_infrastructure.py

# The test script demonstrates:
# - Feature extraction from game states
# - Training environment usage
# - Mock trainer implementation
# - Configuration loading
```

## Configuration

Training parameters are stored in JSON configuration files:

- `configs/mcts_config.json` - MCTS-specific parameters
- `configs/dqn_config.json` - Deep Q-Learning parameters

## Next Steps

1. **Implement MCTS Trainer**: Create `mcts/mcts_trainer.py` with tree search algorithms
2. **Implement DQN Trainer**: Create `deep_q/dqn_trainer.py` with neural network training
3. **Create Trained Agents**: Develop agent classes that use the trained models
4. **Add Self-Play**: Implement agents training against themselves
5. **Curriculum Learning**: Progressive difficulty training regimens

## Integration with Existing Code

The training infrastructure integrates seamlessly with the existing Scotland Yard codebase:

- Uses existing game logic and state management
- Leverages the `GameHeuristics` class for distance calculations
- Utilizes the agent registry system for baseline comparisons
- Integrates with the game analysis and saving infrastructure

## Performance Considerations

- Feature vectors are designed for efficiency with caching
- Distance calculations are memoized in `GameHeuristics`
- Training environments support batch operations
- Configurable verbosity levels for performance tuning

This infrastructure provides a solid foundation for implementing sophisticated AI training algorithms while maintaining compatibility with the existing Scotland Yard game framework.
