"""
Simple heuristic calculations for Scotland Yard game.

This module provides basic heuristic functions for evaluating game positions,
specifically distance calculations between Mr. X and detectives.
"""

import networkx as nx
from typing import List, Dict, Optional, Set
from cops_and_robbers.core.game import ScotlandYardGame, GameState, Player


class GameHeuristics:
    """
    Simple heuristic calculator for Scotland Yard game positions.
    
    This class provides basic distance calculations that can be used by AI agents
    to evaluate game positions and make strategic decisions.
    """
    
    def __init__(self, game: ScotlandYardGame):
        """
        Initialize the heuristics calculator.
        
        Args:
            game: The Scotland Yard game instance
        """
        self.game = game
        self.graph = game.graph
        self.game_state = game.game_state
        
        # Pre-calculate shortest path distances for efficiency
        self._distance_cache = {}
        
    def update_game_state(self, game: ScotlandYardGame):
        """
        Update the game state reference when the game advances.
        
        Args:
            game: Updated game instance
        """
        self.game = game
        self.game_state = game.game_state
        
    def calculate_shortest_distance(self, source: int, target: int) -> int:
        """
        Calculate the shortest path distance between two nodes.
        
        Args:
            source: Starting node ID
            target: Target node ID
            
        Returns:
            Shortest path distance, or -1 if no path exists
        """
        # Use cache to avoid recalculating
        cache_key = (source, target)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
            
        try:
            distance = nx.shortest_path_length(self.graph, source, target)
            self._distance_cache[cache_key] = distance
            return distance
        except nx.NetworkXNoPath:
            # No path between nodes
            self._distance_cache[cache_key] = -1
            return -1
    
    def get_mr_x_distances_from_detectives(self) -> List[int]:
        """
        Calculate Mr. X's distance from each detective.
        
        Returns:
            List of distances from Mr. X to each detective.
            Distance is -1 if no path exists.
        """
        if not self.game_state:
            return []
            
        mr_x_position = self.game_state.robber_position
        detective_positions = self.game_state.cop_positions
        
        distances = []
        for detective_pos in detective_positions:
            distance = self.calculate_shortest_distance(mr_x_position, detective_pos)
            distances.append(distance)
            
        return distances
    
    def get_detective_distances_to_mr_x(self) -> List[int]:
        """
        Calculate each detective's distance to Mr. X.
        
        This is essentially the same as get_mr_x_distances_from_detectives()
        but provides a different perspective for detective agents.
        
        Returns:
            List of distances from each detective to Mr. X.
            Distance is -1 if no path exists.
        """
        return self.get_mr_x_distances_from_detectives()
    
    def get_detective_distances_to_last_known_mr_x_position(self) -> List[int]:
        """
        Calculate each detective's distance to Mr. X's last known visible position.
        
        This is useful when Mr. X is hidden and detectives need to work from
        the last position they saw him.
        
        Returns:
            List of distances from each detective to last known Mr. X position.
            Empty list if Mr. X was never visible or position is unknown.
        """
        if not self.game_state:
            return []
            
        # Get last known position
        last_known_position = None
        
        # If Mr. X is currently visible, use current position
        if self.game_state.mr_x_visible:
            last_known_position = self.game_state.robber_position
        # Otherwise, try to get last visible position from game
        elif hasattr(self.game, 'last_visible_position') and self.game.last_visible_position is not None:
            last_known_position = self.game.last_visible_position
        
        if last_known_position is None:
            return []
            
        detective_positions = self.game_state.cop_positions
        
        distances = []
        for detective_pos in detective_positions:
            distance = self.calculate_shortest_distance(detective_pos, last_known_position)
            distances.append(distance)
            
        return distances
    
    def get_minimum_distance_to_mr_x(self) -> int:
        """
        Get the minimum distance from any detective to Mr. X.
        
        Returns:
            Minimum distance from any detective to Mr. X.
            Returns -1 if no detective can reach Mr. X.
        """
        distances = self.get_mr_x_distances_from_detectives()
        
        if not distances:
            return -1
            
        # Filter out -1 values (unreachable) and find minimum
        valid_distances = [d for d in distances if d >= 0]
        
        if not valid_distances:
            return -1
            
        return min(valid_distances)
    
    def get_maximum_distance_to_mr_x(self) -> int:
        """
        Get the maximum distance from any detective to Mr. X.
        
        Returns:
            Maximum distance from any detective to Mr. X.
            Returns -1 if no detective can reach Mr. X.
        """
        distances = self.get_mr_x_distances_from_detectives()
        
        if not distances:
            return -1
            
        # Filter out -1 values (unreachable) and find maximum
        valid_distances = [d for d in distances if d >= 0]
        
        if not valid_distances:
            return -1
            
        return max(valid_distances)
    
    def get_possible_mr_x_positions(self) -> set:
        """
        Calculate Mr. X's possible positions based on ticket history and last known position.
        
        This method analyzes the moves between reveal turns to determine all possible
        positions where Mr. X could currently be located.
        
        Returns:
            Set of possible node IDs where Mr. X could be located.
            Returns empty set if no analysis is possible.
        """
        if not self.game_state or not hasattr(self.game, 'ticket_history'):
            return set()
        
        last_known_position = self.game.get_mr_x_last_visible_position()
        
        # If Mr. X is currently visible, return his exact position
        if self.game_state.mr_x_visible:
            return {last_known_position}
        
        # Find the last reveal turn and build possibilities from there
        if last_known_position is None:
            return set()
        
        # Get all Mr. X moves since the last reveal
        mr_x_moves_since_reveal = self._get_mr_x_moves_since_last_reveal()
        
        # Build possible positions step by step
        possible_positions = {last_known_position}
        for move_data in mr_x_moves_since_reveal:
            transport_used = move_data.get('transport_used')
            if transport_used is None:
                transport_used = move_data.get('ticket_used')
            
            if transport_used is None:
                continue
            
            possible_positions = self._expand_possibilities(possible_positions, transport_used)

        # exclude cop positions from possible positions
        detective_positions = set(self.game_state.cop_positions)
        possible_positions_filtered = possible_positions - detective_positions
        return possible_positions_filtered
    
    def _was_reveal_turn(self, turn_number: int) -> bool:
        """
        Check if a given turn number was a reveal turn.
        
        Args:
            turn_number: The robber turn number to check
            
        Returns:
            True if this was a reveal turn, False otherwise
        """
        return turn_number in self.game.reveal_turns
    
    def _get_mr_x_moves_since_last_reveal(self) -> List[Dict]:
        """
        Get all Mr. X moves since the last reveal turn.
        
        Returns:
            List of move data dictionaries from ticket history.
        """
        ticket_history = getattr(self.game, 'ticket_history', [])
        moves_since_reveal = []
        
        # Track robber turn count and find the last reveal turn
        last_reveal_robber_turn = -1
        robber_turn_count = 0
        
        for turn_data in ticket_history:
            if turn_data.get('player') == 'mr_x':
                robber_turn_count += 1
                if self._was_reveal_turn(robber_turn_count):
                    last_reveal_robber_turn = robber_turn_count
        
        # Now collect all Mr. X moves after the last reveal turn
        current_robber_turn = 0
        for turn_data in ticket_history:
            if turn_data.get('player') == 'mr_x':
                current_robber_turn += 1
                # Only include moves after the last reveal turn
                if current_robber_turn > last_reveal_robber_turn:
                    mr_x_moves = turn_data.get('mr_x_moves', [])
                    moves_since_reveal.extend(mr_x_moves)
        
        return moves_since_reveal

    def _expand_possibilities(self, current_positions: set, transport_used: int) -> set:
        """
        Expand the set of possible positions based on a transport type used.
        
        Args:
            current_positions: Set of current possible positions
            transport_used: Transport type value (1=taxi, 2=bus, 3=underground, 4=black)

        Returns:
            New set of possible positions after this move
        """
        new_positions = set()
        
        for current_pos in current_positions:
            # Get all neighbors accessible by this transport type
            if self.graph.has_node(current_pos):
                for neighbor in self.graph.neighbors(current_pos):
                    # Check edge data to see if this transport type is available
                    if self._can_use_transport(current_pos, neighbor, transport_used):
                        new_positions.add(neighbor)
        
        return new_positions

    def _can_use_transport(self, source: int, dest: int, transport_type: int) -> bool:
        """
        Check if a specific transport type can be used between two nodes.
        
        Args:
            source: Source node ID
            dest: Destination node ID  
            transport_type: Transport type (1=taxi, 2=bus, 3=underground, 4=black)
            
        Returns:
            True if this transport can be used, False otherwise
        """
        if not self.graph.has_edge(source, dest):
            return False
        
        # Black ticket can use any edge
        if transport_type == 4:  # TransportType.BLACK
            return True
        
        # Get edge data
        edge_data = self.graph.get_edge_data(source, dest)
        
        # Handle multigraph case
        if self.graph.is_multigraph():
            # Check all edges between these nodes
            for edge_key, data in edge_data.items():
                edge_type = data.get('edge_type', 1)
                if edge_type == transport_type:
                    return True
        else:
            # Handle multiple ticket types stored in edge data
            if 'transports' in edge_data:
                return transport_type in edge_data['transports']
            else:
                # Single transport type
                edge_type = edge_data.get('edge_type', 1)
                return edge_type == transport_type

        return False
    
    def get_detective_distances_to_possible_mr_x_positions(self) -> Dict[int, List[int]]:
        """
        Calculate each detective's distance to all possible Mr. X positions.
        
        Returns:
            Dictionary mapping detective index to list of distances to each possible position.
            Each distance list corresponds to the possible positions in the same order.
        """
        possible_positions = self.get_possible_mr_x_positions()
        
        if not possible_positions or not self.game_state:
            return {}
        
        detective_distances = {}
        detective_positions = self.game_state.cop_positions
        
        for detective_idx, detective_pos in enumerate(detective_positions):
            distances = []
            for possible_pos in sorted(possible_positions):  # Sort for consistency
                distance = self.calculate_shortest_distance(detective_pos, possible_pos)
                distances.append(distance)
            detective_distances[detective_idx] = distances
        
        return detective_distances
    
    def get_minimum_distance_to_possible_mr_x_positions(self) -> int:
        """
        Get the minimum distance from any detective to any possible Mr. X position.
        
        Returns:
            Minimum distance, or -1 if no valid distances exist.
        """
        detective_distances = self.get_detective_distances_to_possible_mr_x_positions()
        
        if not detective_distances:
            return -1
        
        min_distance = float('inf')
        
        for distances in detective_distances.values():
            for distance in distances:
                if distance >= 0 and distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else -1

    def clear_cache(self):
        """
        Clear the distance calculation cache.
        
        Call this if the graph structure changes (though it shouldn't
        during a normal Scotland Yard game).
        """
        self._distance_cache.clear()
