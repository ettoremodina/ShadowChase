"""
Base agent classes for Shadow Chase game.

This module defines the abstract base classes for agents that can play Shadow Chase.
All agents must inherit from these base classes and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from ShadowChase.core.game import ShadowChaseGame, Player, TransportType


class Agent(ABC):
    """Abstract base class for all Shadow Chase agents"""
    
    def __init__(self, player: Player):
        self.player = player
    
    @abstractmethod
    def choose_move(self, game: ShadowChaseGame) -> Optional[Tuple]:
        """
        Make a move based on the current game state.
        
        Args:
            game: The current ShadowChase game instance
            
        Returns:
            Move data appropriate for the player type, or None if no move possible
        """
        pass
    
    def get_valid_moves(self, game: ShadowChaseGame, position: int = None) -> Set[Tuple[int, TransportType]]:
        """Get valid moves for this agent's player type"""
        if self.player == Player.DETECTIVES:
            if position is None:
                raise ValueError("Must specify position for detective moves")
            return game.get_valid_moves(Player.DETECTIVES, position)
        else:
            return game.get_valid_moves(Player.MRX)


class DetectiveAgent(Agent):
    """Base class for detective agents"""
    
    def __init__(self, detective_id: int):
        super().__init__(Player.DETECTIVES)
        self.detective_id = detective_id
    
    @abstractmethod
    def choose_move(self, game: ShadowChaseGame) -> Optional[Tuple[int, TransportType]]:
        """
        Make a move for this detective.
        
        Args:
            game: The current ShadowChase game instance
            
        Returns:
            Tuple of (destination, transport_type) or None if no move possible
        """
        pass
    
    def get_current_position(self, game: ShadowChaseGame) -> int:
        """Get current position of this detective"""
        return game.game_state.detective_positions[self.detective_id]
    
    def get_available_tickets(self, game: ShadowChaseGame) -> dict:
        """Get available tickets for this detective"""
        return game.get_detective_tickets(self.detective_id)


class MrXAgent(Agent):
    """Base class for Mr. X agents"""
    
    def __init__(self):
        super().__init__(Player.MRX)
    
    @abstractmethod
    def choose_move(self, game: ShadowChaseGame) -> Optional[Tuple[int, TransportType]]:
        """
        Make a move for Mr. X.
        
        Args:
            game: The current ShadowChase game instance
            
        Returns:
            Tuple of (destination, transport_type) or None if no move possible
        """
        pass
    
    # @abstractmethod
    def should_use_double_move(self, game: ShadowChaseGame) -> bool:
        """
        Decide whether to use a double move.
        
        Args:
            game: The current ShadowChase game instance
            
        Returns:
            True if double move should be used, False otherwise
        """
        pass
    
    def get_current_position(self, game: ShadowChaseGame) -> int:
        """Get current position of Mr. X"""
        return game.game_state.MrX_position
    
    def get_available_tickets(self, game: ShadowChaseGame) -> dict:
        """Get available tickets for Mr. X"""
        return game.get_MrX_tickets()
    
    def can_use_double_move(self, game: ShadowChaseGame) -> bool:
        """Check if Mr. X can use double move"""
        return game.can_use_double_move()


class MultiDetectiveAgent(Agent):
    """Base class for agents controlling multiple detectives simultaneously"""
    
    def __init__(self, num_detectives: int):
        super().__init__(Player.DETECTIVES)
        self.num_detectives = num_detectives

    # @abstractmethod
    def choose_move(self, game: ShadowChaseGame) -> List[Tuple[int, TransportType]]:
        """
        Make moves for all detectives.
        
        Args:
            game: The current ShadowChase game instance
            
        Returns:
            List of moves, one for each detective as (destination, transport_type)
        """
        pass
    
    # @abstractmethod
    def choose_all_moves(self, game: ShadowChaseGame) -> List[Tuple[int, TransportType]]:
        """
        Make moves for all detectives.
        
        Args:
            game: The current ShadowChase game instance
            
        Returns:
            List of moves, one for each detective as (destination, transport_type)
        """
        pass
    
    def make_move(self, game: ShadowChaseGame) -> List[Tuple[int, TransportType]]:
        """Wrapper to match base Agent interface"""
        return self.make_all_moves(game)
