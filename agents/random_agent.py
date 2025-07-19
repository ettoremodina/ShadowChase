"""
Random agent implementations for Scotland Yard game.

This module contains basic random agents that make random valid moves.
These can be used for testing and as baseline implementations.
"""

import random
from typing import List, Tuple, Optional, Set
from cops_and_robbers.core.game import ScotlandYardGame, Player, TransportType, TicketType
from .base_agent import DetectiveAgent, MrXAgent, MultiDetectiveAgent


class RandomDetectiveAgent(DetectiveAgent):
    """Detective agent that makes random valid moves"""
    
    def make_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType]]:
        """Make a random valid move for this detective"""
        current_pos = self.get_current_position(game)
        valid_moves = list(game.get_valid_moves(Player.COPS, current_pos))
        
        if not valid_moves:
            # Stay in place if no valid moves
            return (current_pos, None)
        
        return random.choice(valid_moves)


class RandomMrXAgent(MrXAgent):
    """Mr. X agent that makes random valid moves"""
    
    def make_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType]]:
        """Make a random valid move for Mr. X"""
        valid_moves = list(game.get_valid_moves(Player.ROBBER))
        
        if not valid_moves:
            return None
        
        destination, transport = random.choice(valid_moves)
        
        # Decide whether to use black ticket randomly
        if self._should_use_black_ticket(game, transport):
            transport = TransportType.BLACK
            
        return (destination, transport)
    
    def should_use_double_move(self, game: ScotlandYardGame) -> bool:
        """Randomly decide to use double move (30% chance)"""
        return (self.can_use_double_move(game) and 
                random.random() < 0.3)
    
    def _should_use_black_ticket(self, game: ScotlandYardGame, required_transport: TransportType) -> bool:
        """Decide whether to use black ticket instead of required transport"""
        tickets = self.get_available_tickets(game)
        
        # If we don't have the required ticket, use black if available
        required_ticket = TicketType[required_transport.name]
        if tickets.get(required_ticket, 0) == 0:
            return tickets.get(TicketType.BLACK, 0) > 0
        
        # Otherwise, randomly use black ticket 20% of the time if available
        return (tickets.get(TicketType.BLACK, 0) > 0 and 
                random.random() < 0.2)


class RandomMultiDetectiveAgent(MultiDetectiveAgent):
    """Agent that controls all detectives with random moves"""
    
    def make_all_moves(self, game: ScotlandYardGame) -> List[Tuple[int, TransportType]]:
        """Make random moves for all detectives"""
        detective_moves = []
        pending_moves = []
        
        for i in range(self.num_detectives):
            current_pos = game.game_state.cop_positions[i]
            
            # Get valid moves considering previous detectives' moves
            valid_moves = list(game.get_valid_moves(Player.COPS, current_pos, pending_moves=pending_moves))
            
            if not valid_moves:
                # Stay in place if no valid moves
                move = (current_pos, None)
            else:
                move = random.choice(valid_moves)
            
            detective_moves.append(move)
            pending_moves.append(move)
        
        return detective_moves


class SmartRandomMrXAgent(MrXAgent):
    """
    Slightly smarter Mr. X agent that still uses random moves but with some basic strategy:
    - Prefers moves that take him further from detectives
    - More likely to use black tickets when detectives are close
    - More strategic about double moves
    """
    
    def make_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType]]:
        """Make a somewhat strategic random move for Mr. X"""
        valid_moves = list(game.get_valid_moves(Player.ROBBER))
        
        if not valid_moves:
            return None
        
        # Calculate distances from detectives for each possible move
        current_pos = self.get_current_position(game)
        detective_positions = game.game_state.cop_positions
        
        # Score moves based on distance from detectives
        move_scores = []
        for destination, transport in valid_moves:
            min_distance_to_detective = min(
                self._shortest_path_length(game.graph, destination, det_pos)
                for det_pos in detective_positions
            )
            move_scores.append((min_distance_to_detective, (destination, transport)))
        
        # Sort by distance (furthest first) and take top 3 moves
        move_scores.sort(reverse=True)
        best_moves = [move for _, move in move_scores[:3]]
        
        # Choose randomly from best moves
        destination, transport = random.choice(best_moves)
        
        # Decide whether to use black ticket strategically
        if self._should_use_black_ticket_strategic(game, transport, detective_positions):
            transport = TransportType.BLACK
            
        return (destination, transport)
    
    def should_use_double_move(self, game: ScotlandYardGame) -> bool:
        """Use double move more strategically"""
        if not self.can_use_double_move(game):
            return False
        
        detective_positions = game.game_state.cop_positions
        current_pos = self.get_current_position(game)
        
        # Calculate minimum distance to any detective
        min_distance = min(
            self._shortest_path_length(game.graph, current_pos, det_pos)
            for det_pos in detective_positions
        )
        
        # Use double move if detectives are close (distance <= 2) with 70% probability
        # Or randomly 20% of the time otherwise
        if min_distance <= 2:
            return random.random() < 0.7
        else:
            return random.random() < 0.2
    
    def _should_use_black_ticket_strategic(self, game: ScotlandYardGame, 
                                         required_transport: TransportType,
                                         detective_positions: List[int]) -> bool:
        """Strategic decision about using black tickets"""
        tickets = self.get_available_tickets(game)
        
        # If we don't have the required ticket, use black if available
        required_ticket = TicketType[required_transport.name]
        if tickets.get(required_ticket, 0) == 0:
            return tickets.get(TicketType.BLACK, 0) > 0
        
        # Use black ticket if detectives are close
        current_pos = self.get_current_position(game)
        min_distance = min(
            self._shortest_path_length(game.graph, current_pos, det_pos)
            for det_pos in detective_positions
        )
        
        if min_distance <= 2 and tickets.get(TicketType.BLACK, 0) > 0:
            return random.random() < 0.6  # 60% chance when close
        elif tickets.get(TicketType.BLACK, 0) > 0:
            return random.random() < 0.1  # 10% chance otherwise
        
        return False
    
    def _shortest_path_length(self, graph, source: int, target: int) -> int:
        """Calculate shortest path length between two nodes"""
        try:
            import networkx as nx
            return nx.shortest_path_length(graph, source, target)
        except nx.NetworkXNoPath:
            return float('inf')


