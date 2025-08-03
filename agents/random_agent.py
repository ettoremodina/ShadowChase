"""
Random agent implementations for Shadow Chase game.

This module contains basic random agents that make random valid moves.
These can be used for testing and as baseline implementations.
"""

import random
from typing import List, Tuple, Optional, Set
from ShadowChase.core.game import ShadowChaseGame, Player, TransportType, TicketType
from .base_agent import DetectiveAgent, MrXAgent, MultiDetectiveAgent

class RandomMrXAgent(MrXAgent):
    """Mr. X agent that makes random valid moves"""
    
    def choose_move(self, game: ShadowChaseGame) -> Optional[Tuple[int, TransportType, bool]]:
        """Make a random valid move for Mr. X"""
        valid_moves = list(game.get_valid_moves(Player.MRX))
        
        if not valid_moves:
            return (None, None, False)  # No valid moves
        # Randomly decide to use double move 
        use_double = (self.should_use_double_move(game) and random.choice([True, False]))
        # Filter out black ticket moves
        valid_moves_filtered = [move for move in valid_moves if move[1] != TransportType.BLACK]  # Exclude black ticket
        # first check if we have valid moves without black ticket
        if valid_moves_filtered:
            destination, transport = random.choice(valid_moves_filtered)
        else:
            destination, transport = random.choice(valid_moves)
            return (destination, transport, use_double)
        # Decide whether to use black ticket randomly
        if self._should_use_black_ticket(game, transport):
            transport = TransportType.BLACK
        return (destination, transport, use_double)
    
    def should_use_double_move(self, game: ShadowChaseGame) -> bool:
        """Randomly decide to use double move (30% chance)"""
        return (self.can_use_double_move(game) and 
                random.random() < 0.3)
    
    def _should_use_black_ticket(self, game: ShadowChaseGame, required_transport: TransportType) -> bool:
        """Decide whether to use black ticket instead of required transport"""
        tickets = self.get_available_tickets(game)
        
        # If we don't have the required ticket, use black if available
        required_ticket = TicketType[required_transport.name]
        if tickets.get(required_ticket, 0) == 0:
            return tickets.get(TicketType.BLACK, 0) > 0
        else:
            return (tickets.get(TicketType.BLACK, 0) > 0 and random.random() < 0.5)


class RandomMultiDetectiveAgent(MultiDetectiveAgent):
    """Agent that controls all detectives with random moves"""
    
    def choose_all_moves(self, game: ShadowChaseGame) -> List[Tuple[int, TransportType]]:
        """Make random moves for all detectives"""
        detective_moves = []
        pending_moves = []
        
        for i in range(self.num_detectives):
            current_pos = game.game_state.detective_positions[i]
            
            # Get valid moves considering previous detectives' moves
            valid_moves = list(game.get_valid_moves(Player.DETECTIVES, current_pos, pending_moves=pending_moves))
            
            if not valid_moves:
                # Stay in place if no valid moves
                move = (current_pos, None)
            else:
                move = random.choice(valid_moves)
            
            detective_moves.append(move)
            pending_moves.append(move)
        
        return detective_moves

