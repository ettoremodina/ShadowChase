"""
Centralized game service for saving, loading, and managing games.
This service handles all game persistence operations with consistent metadata.
"""
from datetime import datetime
from typing import Dict, Optional
from ..storage.game_loader import GameLoader
from ..core.game import ScotlandYardGame


class GameService:
    """Centralized service for game operations"""
    
    def __init__(self, loader: GameLoader = None):
        self.loader = loader or GameLoader()
    
    def save_game(self, game: ScotlandYardGame, 
                  game_mode: str = "unknown",
                  player_types: Dict[str, str] = None,
                  additional_metadata: Dict = None) -> str:
        """
        Save a game with standardized metadata format.
        
        Args:
            game: The game to save
            game_mode: Game mode (human_vs_human, ai_vs_ai, etc.)
            player_types: Dict with 'detectives' and 'mr_x' player types
            additional_metadata: Any extra metadata to include
        
        Returns:
            str: The generated game ID
        """
        # Create standardized metadata with no duplicates
        metadata = {
            'game_mode': game_mode,
            'player_types': player_types or {
                'detectives': 'Human',
                'mr_x': 'Human'
            },
            'created_at': datetime.now().isoformat(),
            'game_completed': game.is_game_over(),
            'total_turns': game.game_state.turn_count if game.game_state else 0
        }
        
        # Add any additional metadata, avoiding duplicates
        if additional_metadata:
            clean_metadata = self._clean_additional_metadata(additional_metadata, metadata)
            metadata.update(clean_metadata)
        
        # Save using the loader
        return self.loader.save_game(game, additional_metadata=metadata)
    
    def _clean_additional_metadata(self, additional: Dict, base: Dict) -> Dict:
        """Remove duplicate keys and normalize field names in additional metadata"""
        clean = {}
        
        # Field mapping to standardize names
        field_mappings = {
            'saved_at': 'created_at',
            'game_complete': 'game_completed', 
            'final_turn_count': 'total_turns',
            'session_turns': 'total_turns',
            'num_detectives': 'num_cops',
            'game_mode_description': None  # Remove entirely - can be derived
        }
        
        for key, value in additional.items():
            # Skip if key is already in base metadata
            if key in base:
                continue
            
            # Map to standardized field name or skip if mapped to None
            mapped_key = field_mappings.get(key, key)
            if mapped_key is None:
                continue
            
            # Skip if mapped key is already in base metadata
            if mapped_key in base:
                continue
                
            clean[mapped_key] = value
        
        return clean
    
    def save_ui_game(self, game: ScotlandYardGame,
                     game_mode: str,
                     detective_agent = None,
                     mr_x_agent = None) -> str:
        """Save a game from the UI with appropriate metadata"""
        player_types = {
            'detectives': 'AI' if detective_agent else 'Human',
            'mr_x': 'AI' if mr_x_agent else 'Human'
        }
        
        additional_metadata = {
            'source': 'ui_game'
        }
        
        return self.save_game(game, game_mode, player_types, additional_metadata)
    
    def save_terminal_game(self, game: ScotlandYardGame,
                          play_mode: str,
                          map_size: str,
                          num_detectives: int,
                          turn_count: int) -> str:
        """Save a game from terminal play with appropriate metadata"""
        additional_metadata = {
            'source': 'terminal_game',
            'map_size': map_size,
            'num_cops': num_detectives  # Use standard field name
        }
        
        return self.save_game(game, play_mode, self._get_player_types_from_mode(play_mode), additional_metadata)
    
    def _get_player_types_from_mode(self, play_mode: str) -> Dict[str, str]:
        """Convert play mode to player types"""
        mode_map = {
            "human_vs_human": {'detectives': 'Human', 'mr_x': 'Human'},
            "human_det_vs_ai_mrx": {'detectives': 'Human', 'mr_x': 'AI'},
            "ai_det_vs_human_mrx": {'detectives': 'AI', 'mr_x': 'Human'},
            "ai_vs_ai": {'detectives': 'AI', 'mr_x': 'AI'}
        }
        return mode_map.get(play_mode, {'detectives': 'Human', 'mr_x': 'Human'})
