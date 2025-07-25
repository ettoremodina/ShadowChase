"""
Core training utilities and data structures.

This module contains fundamental data structures and basic utility functions
used across different training algorithms.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class GameResult:
    """Result of a single game episode."""
    winner: str  # "mr_x", "detectives", or "timeout"
    total_turns: int
    game_length: float  # seconds
    mr_x_final_position: int
    detective_final_positions: List[int]
    moves_history: List[Dict[str, Any]]


def calculate_win_rate(results: List[GameResult], player: str) -> float:
    """
    Calculate win rate for a specific player.
    
    Args:
        results: List of game results
        player: Player name ("mr_x" or "detectives")
        
    Returns:
        Win rate as a float between 0 and 1
    """
    if not results:
        return 0.0
    
    wins = sum(1 for result in results if result.winner == player)
    return wins / len(results)


def calculate_average_game_length(results: List[GameResult]) -> float:
    """
    Calculate average game length in turns.
    
    Args:
        results: List of game results
        
    Returns:
        Average game length in turns
    """
    if not results:
        return 0.0
    
    total_turns = sum(result.total_turns for result in results)
    return total_turns / len(results)
