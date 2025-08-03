"""
UI Package for detectives and MrXs Game

Provides a modern, clean GUI for Shadow Chase game with proper separation of concerns.
"""

# Import components only when they're actually used to avoid circular imports
def get_game_visualizer():
    from .game_visualizer import GameVisualizer
    return GameVisualizer



# For direct imports
from .game_visualizer import GameVisualizer

__all__ = [
    "GameVisualizer"
]
