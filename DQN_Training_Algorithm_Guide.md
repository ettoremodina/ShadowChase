# Deep Q-Network (DQN) Training Algorithm for Scotland Yard

## Table of Contents
1. [Overview](#overview)
2. [Deep Q-Learning Fundamentals](#deep-q-learning-fundamentals)
3. [Scotland Yard Game Context](#scotland-yard-game-context)
4. [DQN Architecture](#dqn-architecture)
5. [Training Process](#training-process)
6. [Feature Extraction](#feature-extraction)
7. [Experience Replay](#experience-replay)
8. [Implementation Details](#implementation-details)
9. [Training Configuration](#training-configuration)
10. [Model Loading and Inference](#model-loading-and-inference)

---

## Overview

This document provides a comprehensive explanation of how Deep Q-Networks (DQN) are used to train intelligent agents for the Scotland Yard board game. The implementation combines classical reinforcement learning techniques with deep neural networks to create agents that can learn optimal strategies through self-play.

### What is DQN?

Deep Q-Network (DQN) is a reinforcement learning algorithm that uses deep neural networks to approximate the Q-value function. In the context of Scotland Yard:

- **Q-values** represent the expected future reward for taking a specific action in a given game state
- **Deep Neural Networks** replace traditional Q-tables to handle the large state space
- **Experience Replay** improves sample efficiency by learning from past experiences
- **Target Networks** stabilize training by providing consistent targets

---

## Deep Q-Learning Fundamentals

### Q-Learning Equation

The core of Q-learning is the Bellman equation:

```
Q(s, a) = r + γ * max(Q(s', a'))
```

Where:
- `Q(s, a)` = Q-value for state `s` and action `a`
- `r` = immediate reward
- `γ` = discount factor (how much we value future rewards)
- `s'` = next state
- `max(Q(s', a'))` = maximum Q-value for the next state

### Why Deep Networks?

Traditional Q-learning uses a table to store Q-values for each state-action pair. In Scotland Yard:
- **State space is enormous**: Combinations of player positions, tickets, game phase, etc.
- **Continuous learning**: Need to generalize to unseen states
- **Feature relationships**: Deep networks can learn complex patterns in game states

---

## Scotland Yard Game Context

### Game Characteristics

Scotland Yard presents unique challenges for reinforcement learning:

1. **Two-Player Asymmetric Game**
   - Mr. X (fugitive): Hidden movement, special tickets, goal is to evade
   - Detectives (pursuers): Visible movement, limited tickets, goal is to catch

2. **Partial Observability**
   - Mr. X's position is hidden most of the time
   - Detectives must reason about possible locations

3. **Resource Management**
   - Limited tickets for different transport types
   - Strategic ticket usage is crucial

4. **Turn-Based with Time Pressure**
   - Fixed number of turns (typically 24)
   - Tension between exploration and capture

### State Representation

The game state includes:
- **Player Positions**: Current locations on the London map
- **Ticket Counts**: Remaining taxi, bus, underground, black tickets
- **Game Phase**: Turn number, visibility status
- **Board Connectivity**: Transport network structure
- **Possible Positions**: For hidden Mr. X

---

## DQN Architecture

### Neural Network Structure

Our DQN model (`DQNModel` in `training/deep_q/dqn_model.py`) uses a feedforward architecture:

```python
Input Layer (Feature Vector)
    ↓
Hidden Layer 1 (512 neurons) + ReLU + Dropout
    ↓
Hidden Layer 2 (256 neurons) + ReLU + Dropout
    ↓
Hidden Layer 3 (128 neurons) + ReLU
    ↓
Output Layer (Q-values for all possible actions)
```

### Action Space Handling

**Challenge**: Variable action space (different valid moves each turn)

**Solution**: Fixed output layer + action masking
- Output layer has Q-values for all possible (destination, transport) combinations
- Invalid actions are masked with `-∞` during training and inference
- Action selection uses `argmax` over masked Q-values

```python
# Example from dqn_model.py
def get_masked_q_values(self, state_features, valid_moves):
    q_values = self.forward(state_features)
    mask = torch.full_like(q_values, float('-inf'))
    
    # Set valid actions to 0 (no masking)
    for dest, transport in valid_moves:
        action_idx = self.get_action_index(dest, transport)
        mask[i, action_idx] = 0.0
    
    return q_values + mask  # Invalid actions become -inf
```

### Key Design Decisions

1. **Unified Action Encoding**: `action_index = destination * num_transports + transport_type`
2. **Flexible Architecture**: Configurable hidden layers and dropout
3. **Xavier Initialization**: Proper weight initialization for stable training

---

## Training Process

### High-Level Training Loop

The training process (`DQNTrainer` in `training/deep_q/dqn_trainer.py`) follows this structure:

```
1. Initialize neural networks (main + target)
2. Initialize replay buffer
3. For each episode:
   a. Play full game collecting experiences
   b. Store experiences in replay buffer
   c. Sample batch from replay buffer
   d. Compute Q-learning loss
   e. Update main network
   f. Periodically update target network
   g. Decay exploration rate (epsilon)
4. Save trained model
```

### Episode Structure

Each training episode involves:

1. **Game Setup**: Create new game with random starting positions
2. **Agent vs Agent**: Our training agent plays against a baseline opponent
3. **Experience Collection**: Store (state, action, reward, next_state) tuples
4. **Outcome Evaluation**: Assign rewards based on game result

### Reward Structure

The reward function encourages winning behavior:

```python
# Simplified reward structure
if result.winner == our_agent:
    final_reward = +10.0    # Win bonus
elif result.winner == "timeout":
    final_reward = -1.0     # Slight penalty for timeout
else:
    final_reward = -5.0     # Loss penalty
```

**Future Enhancement**: More sophisticated reward shaping could include:
- Distance-based rewards (getting closer to/farther from opponent)
- Ticket efficiency rewards
- Strategic position rewards

---

## Feature Extraction

### Feature Engineering Pipeline

The `GameFeatureExtractor` converts complex game states into numerical vectors:

#### 1. Game Phase Features (4 dimensions)
- Current turn number (normalized)
- Total turns remaining
- Mr. X visibility status
- Game phase indicator

#### 2. Board State Features (200 dimensions)
- One-hot encoding of all player positions
- Handles up to 200 nodes on the game board

#### 3. Distance Features (18 dimensions)
- Minimum distance from Mr. X to each detective
- Maximum distance from Mr. X to each detective  
- Average distance from Mr. X to each detective
- Overall min/max/average distances

#### 4. Ticket Features (20 dimensions)
- Mr. X: taxi, bus, underground, black, double tickets
- Each detective: taxi, bus, underground tickets
- Normalized by maximum possible tickets

#### 5. Transport Connectivity (12 dimensions)
- For each transport type (taxi, bus, underground):
  - Number of accessible nodes
  - Average connectivity
  - Strategic value score
  - Escape route density

#### 6. Possible Positions (3 dimensions, for Mr. X when hidden)
- Number of possible Mr. X locations
- Minimum distance to possible positions
- Connectivity score of possible positions

### Feature Vector Construction

```python
# Example from feature_extractor.py
def extract_features(self, game, player):
    features = []
    
    if self.config.include_game_phase:
        features.extend(self._extract_game_phase_features(game))
    
    if self.config.include_board_state:
        features.extend(self._extract_position_features(game, player))
    
    if self.config.include_distances:
        features.extend(self._extract_distance_features(game, player))
    
    # ... more feature types
    
    return np.array(features, dtype=np.float32)
```

---

## Experience Replay

### Replay Buffer Design

The `ReplayBuffer` class stores and samples training experiences:

```python
class Experience:
    state: np.ndarray           # Current state features
    action: (int, TransportType) # Action taken (dest, transport)
    reward: float               # Immediate reward
    next_state: np.ndarray      # Next state features
    done: bool                  # Episode terminated?
    valid_moves: Set            # Valid actions from state
    next_valid_moves: Set       # Valid actions from next_state
```

### Benefits of Experience Replay

1. **Sample Efficiency**: Learn from each experience multiple times
2. **Stability**: Break correlation between consecutive experiences
3. **Diverse Learning**: Sample from diverse past experiences

### Buffer Operations

```python
# Adding experience
buffer.push(state, action, reward, next_state, done, valid_moves, next_valid_moves)

# Sampling for training
experiences = buffer.sample(batch_size=32)
```

### Advanced: Prioritized Experience Replay

The implementation includes `PrioritizedReplayBuffer` for future enhancement:
- Samples experiences based on TD error magnitude
- Learns more from "surprising" experiences
- Uses importance sampling to correct for bias

---

## Implementation Details

### Training Step: Q-Learning Update

The core learning happens in `_train_step()`:

```python
def _train_step(self):
    # Sample batch from replay buffer
    experiences = self.replay_buffer.sample(self.batch_size)
    
    # Convert to tensors
    states = torch.FloatTensor([exp.state for exp in experiences])
    actions = [exp.action for exp in experiences]
    rewards = torch.FloatTensor([exp.reward for exp in experiences])
    next_states = torch.FloatTensor([exp.next_state for exp in experiences])
    dones = torch.BoolTensor([exp.done for exp in experiences])
    
    # Current Q-values for chosen actions
    current_q_values = self.main_network.get_masked_q_values(states, valid_moves)
    current_q_values = current_q_values.gather(1, action_indices)
    
    # Target Q-values using target network
    next_q_values = self.target_network.get_masked_q_values(next_states, next_valid_moves)
    target_q_values = rewards + (self.gamma * next_q_values.max(1)[0] * ~dones)
    
    # Compute loss and update
    loss = F.mse_loss(current_q_values, target_q_values.detach())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Epsilon-Greedy Exploration

Balances exploration vs exploitation:

```python
def select_action(self, state, valid_moves, epsilon):
    if random.random() < epsilon:
        return random.choice(list(valid_moves))  # Explore
    else:
        # Exploit: choose action with highest Q-value
        q_values = self.model.get_masked_q_values(state, valid_moves)
        return self.model.get_action_from_index(q_values.argmax())
```

### Target Network Updates

Stabilizes training by providing consistent targets:

```python
# Every N episodes, copy main network weights to target network
if episode % self.target_update_frequency == 0:
    self.target_network.load_state_dict(self.main_network.state_dict())
```

---

## Training Configuration

### Key Hyperparameters

From `training/configs/dqn_config.json`:

```json
{
  "training_parameters": {
    "num_episodes": 5000,        // Total training episodes
    "batch_size": 32,            // Mini-batch size for SGD
    "learning_rate": 0.001,      // Adam optimizer learning rate
    "gamma": 0.95,               // Discount factor for future rewards
    "epsilon_start": 1.0,        // Initial exploration rate
    "epsilon_end": 0.01,         // Final exploration rate
    "epsilon_decay": 0.995,      // Exploration decay per episode
    "target_update_frequency": 100, // Update target network every N episodes
    "replay_buffer_size": 10000, // Maximum experiences in buffer
    "min_replay_buffer_size": 1000 // Start training when buffer has N experiences
  }
}
```

### Network Architecture

```json
{
  "network_parameters": {
    "hidden_layers": [512, 256, 128], // Layer sizes
    "activation": "relu",             // Activation function
    "dropout_rate": 0.1               // Regularization
  }
}
```

---

## Model Loading and Inference

### Trained Agent Usage

The `DQNMrXAgent` and `DQNDetectiveAgent` classes load trained models:

```python
class DQNMrXAgent(MrXAgent):
    def __init__(self, model_path=None, epsilon=0.05):
        # Load trained model
        self.load_model(model_path)
    
    def choose_move(self, game):
        # Extract features from game state
        features = self.feature_extractor.extract_features(game, Player.MRX)
        
        # Get Q-values and select action
        q_values = self.model.get_masked_q_values(features, valid_moves)
        action_idx = q_values.argmax()
        return self.model.get_action_from_index(action_idx)
```

### Model Checkpointing

Models are saved with complete training information:

```python
checkpoint = {
    'main_network': self.main_network.state_dict(),
    'target_network': self.target_network.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'config': self.config,
    'training_stats': {
        'episode_rewards': self.episode_rewards,
        'final_epsilon': self.current_epsilon,
        'total_steps': self.step_count
    }
}
torch.save(checkpoint, model_path)
```

---

## Training Process Walkthrough

### 1. Initialization Phase
```python
trainer = DQNTrainer(player_role="mr_x", config_path="dqn_config.json")
```
- Loads configuration from JSON file
- Initializes feature extractor with specified parameters
- Sets up training environment

### 2. Network Creation
```python
# Based on sample game, determine feature vector size
feature_size = feature_extractor.get_feature_size(sample_game)
main_network = DQNModel(input_size=feature_size, ...)
target_network = DQNModel(input_size=feature_size, ...)
```
- Creates main and target networks with identical architecture
- Initializes replay buffer for experience storage

### 3. Training Loop Execution
For each episode:
```python
# Episode execution
our_agent = DQNAgent(trainer, Player.MRX)
opponent = RandomAgent()  # Baseline opponent
result, experiences = env.run_episode(our_agent, opponent)

# Experience processing
episode_reward = process_experiences(result, experiences)
replay_buffer.push(experiences)

# Neural network training
if len(replay_buffer) >= min_replay_size:
    loss = train_step()
    losses.append(loss)

# Network updates
if episode % target_update_frequency == 0:
    target_network.load_state_dict(main_network.state_dict())
```

### 4. Progress Monitoring
```python
if episode % 50 == 0:
    avg_reward = np.mean(episode_rewards[-50:])
    print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
          f"Epsilon={current_epsilon:.3f}, Buffer={len(replay_buffer)}")
```

### 5. Model Persistence
```python
# Save trained model with metadata
model_path = save_model(episode, performance_metrics)
print(f"Model saved to: {model_path}")
```

---

## Key Insights and Design Choices

### 1. Action Space Management
**Challenge**: Scotland Yard has a variable action space that changes each turn.
**Solution**: Fixed neural network output with action masking ensures consistent architecture while handling invalid actions.

### 2. Feature Engineering
**Challenge**: Game states are complex and multifaceted.
**Solution**: Comprehensive feature extraction covering positions, distances, tickets, and strategic elements provides rich input for learning.

### 3. Reward Design
**Challenge**: Sparse rewards (only at game end) make learning difficult.
**Solution**: Current implementation uses terminal rewards, with future potential for reward shaping.

### 4. Exploration Strategy
**Challenge**: Balance between exploring new strategies and exploiting learned knowledge.
**Solution**: Epsilon-greedy with decay ensures exploration early in training, exploitation later.

### 5. Training Stability
**Challenge**: Q-learning can be unstable with neural networks.
**Solution**: Experience replay, target networks, and proper hyperparameter tuning stabilize training.

---

This implementation provides a solid foundation for training intelligent Scotland Yard agents using deep reinforcement learning, with the flexibility to extend and improve the approach as needed.
