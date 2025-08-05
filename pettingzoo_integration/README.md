# Shadow Chase PettingZoo Integration

This directory contains a PettingZoo-compatible environment for the Shadow Chase game, enabling multi-agent reinforcement learning experiments.

## Overview

The Shadow Chase PettingZoo environment wraps the existing Shadow Chase game to be compatible with the PettingZoo API. This allows you to:

- Train RL agents using any PettingZoo-compatible algorithm
- Run multi-agent experiments with different agent types
- Integrate with popular RL libraries (Stable-Baselines3, RLLib, etc.)
- Compare different approaches on the same standardized interface

## Key Design Decisions

### Agent Structure
The environment treats Shadow Chase as a **2-player asymmetric game**:

- **"mrx"**: Controls Mr. X with actions (destination, transport, use_double_move)
- **"detectives"**: Controls all detectives as a single agent with sequential moves

This design choice was made to:
1. **Simplify the action space**: Instead of 5+ agents with complex coordination
2. **Handle sequential detective moves**: The environment internally manages the sequential movement restriction
3. **Preserve game mechanics**: Double moves and movement restrictions are handled correctly
4. **Enable standard RL algorithms**: Most multi-agent RL works best with 2-3 agents

### Action Spaces

**Mr. X Action Space** (`MultiDiscrete`):
```python
[num_nodes, num_transport_types, 2]
# [destination, transport_type, use_double_move]
```

**Detective Action Space** (`MultiDiscrete`):
```python
[num_nodes, num_transport_types] * num_detectives
# [dest1, transport1, dest2, transport2, ...]
```

### Observation Space
Both agents receive the same observation - full game state features extracted using the existing `GameFeatureExtractor`. This includes:
- Positions (when visible)
- Ticket counts
- Turn information
- Distance features (optional)
- Move history (optional)

### How Sequential Detective Movement is Handled

The environment internally processes detective actions sequentially:

1. The "detectives" agent provides all detective moves at once
2. The environment processes them one by one, using `pending_moves` to prevent collisions
3. Invalid moves (due to tickets or collisions) result in the detective staying in place
4. This preserves the original game's movement constraints while presenting a simple interface

### Double Move Complexity

Mr. X's double moves are handled by:
1. The action includes a `use_double_move` flag
2. If true and available, Mr. X gets another turn immediately
3. The environment manages the turn order correctly
4. Double move tickets are consumed appropriately

## Installation

```bash
# Install PettingZoo and Gymnasium
pip install pettingzoo gymnasium

# The Shadow Chase environment uses the existing game infrastructure
# No additional installation needed
```

## Usage

### Basic Usage

```python
from pettingzoo_integration import create_shadow_chase_env

# Create environment
env = create_shadow_chase_env(
    map_size="test",        # "test" (10 nodes) or "full" (199 nodes)
    num_detectives=2,       # 2-5 detectives
    max_turns=24,          # Maximum turns before truncation
    render_mode="human"     # "human", "ansi", or None
)

# Reset environment
observations, infos = env.reset()

# Game loop
while env.agents:
    agent = env.agent_selection
    observation = env.observe(agent)
    action_mask = env.action_mask(agent)
    
    # Choose action (implement your policy here)
    action = your_policy(observation, action_mask)
    
    # Step environment
    env.step(action)
    
    # Optional: render
    env.render()

env.close()
```

### Pre-configured Environments

```python
from pettingzoo_integration import create_test_env, create_training_env, create_evaluation_env

# Quick testing (small map, few turns)
test_env = create_test_env()

# Optimized for training (no rendering, good features)
train_env = create_training_env()

# Full evaluation (large map, all detectives, rich features)
eval_env = create_evaluation_env()
```

### Action Masks

The environment provides action masks to ensure only valid actions are taken:

```python
agent = env.agent_selection
action_mask = env.action_mask(agent)

# For Mr. X: mask is shaped [num_nodes, num_transports, 2]
# For Detectives: mask is shaped [num_nodes * num_transports * num_detectives]

# Use mask to filter valid actions
if hasattr(env.action_spaces[agent], 'nvec'):
    # MultiDiscrete space
    valid_actions = []
    for i, n in enumerate(env.action_spaces[agent].nvec):
        if i < len(action_mask) and action_mask[i]:
            valid_actions.append(np.random.randint(0, n))
        else:
            valid_actions.append(0)
    action = np.array(valid_actions)
```

## Integration with RL Libraries

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import parallel_wrapper_fn

# Convert to parallel environment (if needed)
parallel_env = parallel_wrapper_fn(create_training_env)

# Train with your favorite algorithm
# Note: You may need additional wrappers for multi-agent training
```

### RLLib

```python
import ray
from ray.rllib.algorithms.ppo import PPO

# RLLib can handle PettingZoo environments directly
config = {
    "env": "shadow_chase_v1",
    "multiagent": {
        "policies": {
            "mrx_policy": (None, obs_space, action_space, {}),
            "detective_policy": (None, obs_space, action_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "mrx_policy" if agent_id == "mrx" else "detective_policy",
    },
}

trainer = PPO(config=config)
```

## Files

- **`shadow_chase_env.py`**: Main environment implementation
- **`env_utils.py`**: Utility functions for creating configured environments
- **`example_usage.py`**: Example usage and random agent implementation
- **`README.md`**: This documentation

## Limitations and Future Work

### Current Limitations

1. **Sequential Detective Processing**: While the interface is simplified, detective moves are still processed sequentially internally, which may not be ideal for some RL algorithms.

2. **Action Space Size**: For large maps and many detectives, the action space can become quite large.

3. **Observation Complexity**: The feature extraction includes many components which might be overwhelming for simpler algorithms.

4. **Reward Shaping**: Current reward structure is basic - more sophisticated reward shaping could improve learning.

### Potential Improvements

1. **Simultaneous Detective Moves**: Implement true simultaneous movement with conflict resolution.

2. **Individual Detective Agents**: Create separate agents for each detective (would be 6 agents total).

3. **Hierarchical Actions**: Use hierarchical action spaces to reduce complexity.

4. **Custom Reward Functions**: Allow pluggable reward functions for different training objectives.

5. **Curriculum Learning**: Start with simpler maps/configurations and gradually increase complexity.

## Compatibility with Existing Agents

The PettingZoo environment can interface with existing Shadow Chase agents:

```python
from agents import AgentType, get_agent_registry

# Wrap existing agents for PettingZoo
class ExistingAgentWrapper:
    def __init__(self, agent):
        self.agent = agent
    
    def __call__(self, observation, action_mask):
        # Convert PettingZoo observation to game state
        # Call existing agent
        # Convert back to PettingZoo action format
        pass

# Use existing heuristic agents
registry = get_agent_registry()
mrx_agent = registry.create_agent(AgentType.HEURISTIC, Player.MRX)
detective_agent = registry.create_agent(AgentType.HEURISTIC, Player.DETECTIVES, num_detectives=2)

wrapped_mrx = ExistingAgentWrapper(mrx_agent)
wrapped_detectives = ExistingAgentWrapper(detective_agent)
```

## Performance Considerations

- **Map Size**: Use "test" map for faster training, "full" map for evaluation
- **Feature Extraction**: Use minimal features for faster training
- **Rendering**: Disable rendering during training (`render_mode=None`)
- **Max Turns**: Shorter games train faster but may not capture full game dynamics

## Contributing

To extend or improve the PettingZoo integration:

1. **Add New Action Spaces**: Modify `_setup_spaces()` method
2. **Improve Observations**: Extend the feature extractor
3. **Add Reward Shaping**: Modify `_calculate_rewards()` method
4. **Add Wrappers**: Create additional PettingZoo wrappers for specific use cases

The integration is designed to be modular and extensible while maintaining compatibility with the existing Shadow Chase codebase.
