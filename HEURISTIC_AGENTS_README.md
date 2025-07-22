# Heuristic AI Agents and Agent Selection System

## Overview

This implementation adds heuristic-based AI agents for Scotland Yard and creates a comprehensive system for selecting and managing different AI agent types. The system includes both terminal and UI interfaces with proper game saving that records the AI agent types used.

## New Components

### 1. Heuristic Agents (`agents/heuristic_agent.py`)
- **HeuristicMrXAgent**: Mr. X agent that maximizes distance from the closest detective
- **HeuristicMultiDetectiveAgent**: Detective agent that minimizes distance to Mr. X's last known position
- **HeuristicDetectiveAgent**: Single detective agent with the same heuristic strategy

### 2. Agent Registry System (`agents/agent_registry.py`)
- **AgentRegistry**: Central registry for managing different AI agent implementations
- **AgentType**: Enum defining available agent types (RANDOM, HEURISTIC)
- **AgentSelector**: Helper class for terminal and UI agent selection
- Extensible design for adding future agent types (minimax, MCTS, deep learning, etc.)

### 3. Updated Components

#### Terminal Interface Updates
- **simple_play/simple_game.py**: Added agent selection for batch and interactive modes
- **simple_play/game_logic.py**: Updated `get_game_mode()` to include agent selection
- **simple_play/game_utils.py**: Updated all game and saving functions to support agent types

#### UI Interface Updates  
- **cops_and_robbers/ui/setup_controls.py**: Added agent selection dropdowns
- **cops_and_robbers/ui/game_visualizer.py**: Updated to use agent registry system

#### Game Saving Updates
- **cops_and_robbers/services/game_service.py**: Updated saving methods to include agent type metadata
- All save functions now record which AI agent types were used

## Features

### Agent Strategy Details
- **Random Agents**: Make random valid moves (existing)
- **Heuristic Agents**: 
  - Mr. X tries to maximize distance from closest detective
  - Detectives try to minimize distance to Mr. X's last known position
  - Uses shortest path calculations on the game graph
  - Strategic use of special moves (double move, black tickets)

### Agent Selection Interface
- **Terminal**: Interactive menus with agent descriptions
- **UI**: Dropdown menus that appear based on game mode
- **Batch Mode**: Agent selection for automated game runs
- **Dynamic UI**: Agent selection only shows when AI players are involved

### Game Metadata Enhancement
- Saved games now include:
  - `mr_x_agent_type`: Type of AI used for Mr. X (if AI)
  - `detective_agent_type`: Type of AI used for detectives (if AI)
  - Enhanced player type descriptions (e.g., "AI (Heuristic)" instead of just "AI")

## Usage Examples

### Terminal Interface
```bash
# Interactive game with agent selection
python simple_play/simple_game.py

# Batch mode with agent selection
python simple_play/simple_game.py --batch 10 --map-size extracted
```

### Programmatic Usage
```python
from agents import AgentType, create_agents_from_types

# Create agents
mr_x_agent, detective_agent = create_agents_from_types(
    AgentType.HEURISTIC, 
    AgentType.RANDOM, 
    num_detectives=5
)
```

### UI Interface
- Game mode selection automatically shows relevant agent selection dropdowns
- Agent types are saved with the game for replay analysis

## Extensibility

The system is designed for easy extension:

1. **New Agent Types**: Add to `AgentType` enum and register in `AgentRegistry`
2. **New Strategies**: Implement `MrXAgent` or `MultiDetectiveAgent` base classes
3. **Registration**: Use `agent_registry.register_*_agent()` methods

Example of adding a new agent type:
```python
# 1. Add to AgentType enum
class AgentType(Enum):
    MINIMAX = "minimax"

# 2. Register in AgentRegistry
agent_registry.register_mr_x_agent(
    AgentType.MINIMAX, 
    MinimaxMrXAgent, 
    "Minimax Mr. X - Uses game tree search"
)
```

## Testing

- **test_heuristic_agents.py**: Tests agent creation and registry system  
- **test_game_saving.py**: Tests metadata generation and saved game structure

## Files Modified/Created

### New Files
- `agents/heuristic_agent.py`
- `agents/agent_registry.py` 
- `test_heuristic_agents.py`
- `test_game_saving.py`

### Modified Files
- `agents/__init__.py`
- `simple_play/simple_game.py`
- `simple_play/game_logic.py`
- `simple_play/game_utils.py`
- `cops_and_robbers/ui/setup_controls.py`
- `cops_and_robbers/ui/game_visualizer.py`
- `cops_and_robbers/services/game_service.py`

## Benefits

1. **Strategic AI**: Heuristic agents provide more intelligent gameplay than random agents
2. **Extensible Architecture**: Easy to add new agent types in the future
3. **Complete Integration**: Works in both terminal and UI interfaces
4. **Enhanced Metadata**: Game saves include AI strategy information for analysis
5. **User Choice**: Players can select AI difficulty/strategy for different game experiences
6. **Research Ready**: Foundation for implementing more advanced AI techniques

This implementation provides a solid foundation for AI agent research and experimentation in the Scotland Yard domain, while maintaining ease of use for players.
