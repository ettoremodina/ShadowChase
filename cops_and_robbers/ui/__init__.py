"""
UI Package for Cops and Robbers Game

Provides a modern, clean GUI for Scotland Yard game with proper separation of concerns.
"""

# Import components only when they're actually used to avoid circular imports
def get_game_visualizer():
    from .game_visualizer import GameVisualizer
    return GameVisualizer

def get_setup_controller():
    from .setup_controller import SetupController
    return SetupController

def get_gameplay_controller():
    from .gameplay_controller import GameplayController
    return GameplayController

def get_graph_renderer():
    from .graph_renderer import GraphRenderer
    return GraphRenderer

def get_info_panel():
    from .info_panel import InfoPanel
    return InfoPanel

def get_control_panel():
    from .control_panel import ControlPanel
    return ControlPanel

# For direct imports
from .game_visualizer import GameVisualizer

__all__ = [
    "GameVisualizer",
    "get_setup_controller", 
    "get_gameplay_controller",
    "get_graph_renderer",
    "get_info_panel",
    "get_control_panel"
]
