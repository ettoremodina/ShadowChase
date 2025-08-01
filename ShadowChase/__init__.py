"""
detectives and MrXs Game Package

A comprehensive implementation of the detectives and MrXs pursuit-evasion game
with flexible rules, solvers, and interactive visualization.
"""

from .core.game import (
    Game, GameState, Player,
    MovementRule, StandardMovement, DistanceKMovement,
    WinCondition, CaptureWinCondition, DistanceKWinCondition
)

# from .solver.base_solver import BaseSolver, Strategy, SolverResult
# from .solver.minimax_solver import MinimaxSolver

from .ui.game_visualizer import GameVisualizer

from .examples.example_games import (
    create_path_graph_game,
    create_cycle_graph_game, 
    create_complete_graph_game,
    create_grid_graph_game,
    create_petersen_graph_game,
    create_distance_k_game,
    create_distance_k_win_game
)
from .services.game_loader import GameLoader, GameRecord

__version__ = "1.0.0"
__author__ = "detectives and MrXs Game"

__all__ = [
    # Core classes
    "Game", "GameState", "Player",
    "MovementRule", "StandardMovement", "DistanceKMovement", 
    "WinCondition", "CaptureWinCondition", "DistanceKWinCondition",
    # "Obstacle", "StaticObstacle",
    
    # Solver classes
    "BaseSolver", "Strategy", "SolverResult", "MinimaxSolver",
    
    # UI classes
    "GameVisualizer",
    
    # Example functions
    "create_path_graph_game", "create_cycle_graph_game",
    "create_complete_graph_game", "create_grid_graph_game", 
    "create_petersen_graph_game", "create_distance_k_game",
    "create_distance_k_win_game"

    # Loading and storage
    "GameLoader", "GameRecord"
]


