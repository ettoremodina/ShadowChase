# Simple Play - Terminal Scotland Yard

A clean, terminal-based implementation of Scotland Yard without GUI dependencies.

## Quick Start

```bash
cd ScotlandYardRL
python simple_play/simple_game.py
```

## Features

### Game Modes
- **Human vs Human**: Control both detectives and Mr. X
- **Human Detectives vs AI Mr. X**: Play as detectives against AI
- **AI Detectives vs Human Mr. X**: Play as Mr. X against AI detectives
- **AI vs AI**: Watch the AI play against itself

### Map Options
- **Test Map**: 10 nodes, good for learning the game
- **Full Map**: 199 nodes, complete Scotland Yard experience

### Display Verbosity Levels
1. **Basic**: Just positions and current turn
2. **Standard**: + Available moves and ticket counts
3. **Detailed**: + Move history and transport details
4. **Debug**: + All internal game state information

## Move Input Format

### Basic Moves
- `45` - Move to position 45 (auto-select transport if only one option)
- `45 taxi` - Move to position 45 using taxi
- `45 bus` - Move to position 45 using bus
- `45 underground` - Move to position 45 using underground

### Special Moves (Mr. X only)
- `B45` or `45 black` - Move to position 45 using black ticket
- `DD` - Activate double move (then enter two consecutive moves)

### Commands
- `help` - Show input help
- `quit` - Exit game

## Example Game Flow

```
🕵️ SCOTLAND YARD - SIMPLE TERMINAL GAME 🕵️‍♂️
============================================================

📍 Choose map size:
1. Test map (10 nodes) - Good for learning
2. Full map (199 nodes) - Complete Scotland Yard
Map size (1-2): 1

🎭 Choose play mode:
1. Human vs Human - You control both sides
2. Human Detectives vs AI Mr. X
3. AI Detectives vs Human Mr. X
4. AI vs AI - Watch the game play
Play mode (1-4): 2

📊 Choose display verbosity:
1. Basic - Just positions and turn
2. Standard - + Available moves and tickets
3. Detailed - + Move history and transport details
4. Debug - + All internal game state
Verbosity level (1-4): 2

ℹ️  Created test game (10 nodes)
ℹ️  Game mode: Human Detectives vs AI Mr. X

============================================================
  TURN 1
============================================================

🎯 TURN 1 - ROBBER'S TURN

🕵️ DETECTIVES:
  Detective 1: Position 1
  Detective 2: Position 3

🕵️‍♂️ MR. X:
  Position: Hidden ❓

🎫 TICKETS:
  Detective 1:
    🚕 Taxi: 1
    🚌 Bus: 1
    🚇 Underground: 1
  Detective 2:
    🚕 Taxi: 1
    🚌 Bus: 1
    🚇 Underground: 1
  Mr. X:
    🚕 Taxi: 1
    🚌 Bus: 3
    🚇 Underground: 3
    ⚫ Black: 1
    ⚡ Double_move: 2

🕵️‍♂️ AI MR. X'S TURN
✅ AI Mr. X moved
```

## Project Structure

```
simple_play/
├── __init__.py          # Package initialization
├── display_utils.py     # Terminal display formatting
├── game_logic.py        # Game flow and AI logic
├── simple_game.py       # Main game script
└── README.md           # This file
```

## Integration

The simple play system integrates with the existing Scotland Yard codebase:

- Uses `cops_and_robbers.core.game.ScotlandYardGame` for game logic
- Uses `cops_and_robbers.examples.example_games` for game creation
- Compatible with existing AI solvers (uses random moves for now)

## Customization

### Adding New Display Formats
Extend the `GameDisplay` class in `display_utils.py` to add new formatting options.

### Adding New AI Strategies
Modify the `make_ai_move` method in `GameController` to use more sophisticated AI algorithms.

### Adding New Game Modes
Add new play modes in `game_logic.py` by extending the `get_game_mode` function.

---

**Note**: This is a simplified interface focused on clean terminal gameplay. For full graphical visualization, use the existing GUI system in `cops_and_robbers.ui.game_visualizer`.
