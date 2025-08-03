# ShadowChase/storage/game_loader.py
import os
import json
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import networkx as nx
from ..core.game import Game, GameState, ShadowChaseGame

class GameRecord:
    """Represents a saved game record"""
    
    def __init__(self, game_id: str, metadata: Dict, game_config: Dict, 
                 game_history: List[GameState], ticket_history: List[Dict], solver_result: None):
        self.game_id = game_id
        self.metadata = metadata
        self.game_config = game_config
        self.game_history = game_history
        # self.solver_result = solver_result
        self.ticket_history = ticket_history
        self.created_at = metadata.get('created_at', datetime.now().isoformat())
        self.updated_at = datetime.now().isoformat()

class GameLoader:
    """Handles saving and loading of games with organized folder structure"""
    
    def __init__(self, base_directory: str = "fritto_misto"):
        self.base_dir = "saved_games/" + base_directory
        self.ensure_directory_structure()
    
    def ensure_directory_structure(self):
        """Create organized folder structure"""
        directories = [
            self.base_dir,
            f"{self.base_dir}/games",
            f"{self.base_dir}/games/by_date",
            f"{self.base_dir}/games/by_graph_type",
            f"{self.base_dir}/games/by_outcome",
            f"{self.base_dir}/metadata",
            f"{self.base_dir}/exports",
            f"{self.base_dir}/statistics"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_game(self, game: ShadowChaseGame, game_id: str = None, 
                  additional_metadata: Dict = None) -> str:
        """Save a complete game with organized file structure"""
        
        if game_id is None:
            game_id = self._generate_game_id()
        
        metadata = {
            'game_id': game_id,
            'created_at': datetime.now().isoformat(),
            'graph_type': self._determine_graph_type(game.graph),
            'num_nodes': game.graph.number_of_nodes(),
            'num_edges': game.graph.number_of_edges(),
            'num_detectives': game.num_detectives,
            'total_turns': game.game_state.turn_count if game.game_state else 0,
            'winner': game.get_winner().value if game.get_winner() else None,
            'game_completed': game.is_game_over(),
            'has_ticket_system': isinstance(game, ShadowChaseGame)
        }
        
        # Add additional metadata (including game mode)
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Prepare game configuration
        game_config = {
            'graph_data': nx.node_link_data(game.graph, edges="links"),
            'num_detectives': game.num_detectives,
            'movement_rules': {
                'detective_movement': self._serialize_movement_rule(game.detective_movement),
                'MrX_movement': self._serialize_movement_rule(game.MrX_movement)
            },
            'win_condition': self._serialize_win_condition(game.win_condition),
        }
        
        # Create game record - ticket history is already in game_state
        solver_result = getattr(game, 'solver_result', None)
        record = GameRecord(game_id, metadata, game_config, game.game_history, game.ticket_history, solver_result)
        
        # Save to multiple organized locations
        self._save_to_organized_structure(record)
        
        return game_id
    
    def load_game(self, game_id: str) -> Optional[ShadowChaseGame]:
        """Load a game by ID"""
        record = self.load_game_record(game_id)
        if not record:
            return None
        
        return self._reconstruct_game_from_record(record)
    
    def load_game_record(self, game_id: str) -> Optional[GameRecord]:
        """Load complete game record by ID"""
        game_file = f"{self.base_dir}/games/{game_id}.pkl"
        
        if not os.path.exists(game_file):
            return None
        
        try:
            with open(game_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading game {game_id}: {e}")
            return None
    
    def list_games(self, filter_by: Dict = None) -> List[Dict]:
        """List saved games with optional filtering"""
        games = []
        
        for filename in os.listdir(f"{self.base_dir}/games"):
            if filename.endswith('.pkl') and not filename.startswith('by_'):
                game_id = filename[:-4]  # Remove .pkl extension
                record = self.load_game_record(game_id)
                
                if record and self._matches_filter(record.metadata, filter_by):
                    games.append(record.metadata)
        
        return sorted(games, key=lambda x: x['created_at'], reverse=True)
    
    def delete_game(self, game_id: str) -> bool:
        """Delete a saved game"""
        try:
            # Remove from main games directory
            main_file = f"{self.base_dir}/games/{game_id}.pkl"
            if os.path.exists(main_file):
                os.remove(main_file)
            
            # Remove from organized directories
            self._remove_from_organized_structure(game_id)
            
            return True
        except Exception as e:
            print(f"Error deleting game {game_id}: {e}")
            return False
    
    def export_game(self, game_id: str, format: str = 'json') -> Optional[str]:
        """Export game to various formats with ticket information"""
        record = self.load_game_record(game_id)
        if not record:
            return None
        
        export_data = {
            'metadata': record.metadata,
            'game_config': record.game_config,
            'game_history': [self._serialize_game_state(state) for state in record.game_history],
            'ticket_history': getattr(record, 'ticket_history', [])
        }
        
        if format == 'json':
            export_file = f"{self.base_dir}/exports/{game_id}.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'csv':
            export_file = f"{self.base_dir}/exports/{game_id}.csv"
            self._export_to_csv_with_tickets(record, export_file)
        else:
            return None
        
        return export_file
    
    def _generate_game_id(self) -> str:
        """Generate unique game ID using UUID4 for guaranteed uniqueness"""
        # Generate a UUID4 (random UUID) and take first 8 characters for readability
        unique_id = str(uuid.uuid4()).replace('-', '')[:16]
        
        # Add readable timestamp prefix for easier identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine timestamp with UUID for both readability and guaranteed uniqueness
        return f"game_{timestamp}_{unique_id}"
    
    def _determine_graph_type(self, graph: nx.Graph) -> str:
        """Determine graph type for categorization"""
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        if nx.is_tree(graph):
            if m == n - 1:  # Path graph
                return "path"
            else:
                return "tree"
        elif m == n and nx.is_connected(graph):  # Cycle
            return "cycle"
        elif m == n * (n - 1) // 2:  # Complete graph
            return "complete"
        elif self._is_grid_graph(graph):
            return "grid"
        elif n == 10 and m == 15:  # Petersen graph characteristics
            return "petersen"
        else:
            return "custom"
    
    def _is_grid_graph(self, graph: nx.Graph) -> bool:
        """Check if graph is a grid graph"""
        try:
            # Simple heuristic: check if nodes can be arranged in grid pattern
            nodes = list(graph.nodes())
            if not all(isinstance(node, int) for node in nodes):
                return False
            
            # Check degree distribution typical of grid graphs
            degrees = [graph.degree(node) for node in nodes]
            corner_nodes = sum(1 for d in degrees if d == 2)
            edge_nodes = sum(1 for d in degrees if d == 3)
            internal_nodes = sum(1 for d in degrees if d == 4)
            
            return corner_nodes == 4 and (edge_nodes + internal_nodes) > 0
        except:
            return False
    
    def _save_to_organized_structure(self, record: GameRecord):
        """Save game record to organized folder structure"""
        game_id = record.game_id
        
        # Main game file
        main_file = f"{self.base_dir}/games/{game_id}.pkl"
        with open(main_file, 'wb') as f:
            pickle.dump(record, f)
        
        # By date
        date_str = record.created_at[:10]
        date_dir = f"{self.base_dir}/games/by_date/{date_str}"
        os.makedirs(date_dir, exist_ok=True)
        os.link(main_file, f"{date_dir}/{game_id}.pkl")
        
        # By graph type
        graph_type = record.metadata['graph_type']
        type_dir = f"{self.base_dir}/games/by_graph_type/{graph_type}"
        os.makedirs(type_dir, exist_ok=True)
        os.link(main_file, f"{type_dir}/{game_id}.pkl")
        
        # By outcome
        if record.metadata['winner']:
            outcome_dir = f"{self.base_dir}/games/by_outcome/{record.metadata['winner']}_wins"
            os.makedirs(outcome_dir, exist_ok=True)
            os.link(main_file, f"{outcome_dir}/{game_id}.pkl")
        
        # Metadata file
        metadata_file = f"{self.base_dir}/metadata/{game_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(record.metadata, f, indent=2)
    
    def _remove_from_organized_structure(self, game_id: str):
        """Remove game from organized folder structure"""
        # Remove from organized directories
        subdirs = ['by_date', 'by_graph_type', 'by_outcome']
        
        for subdir in subdirs:
            base_path = f"{self.base_dir}/games/{subdir}"
            for root, dirs, files in os.walk(base_path):
                if f"{game_id}.pkl" in files:
                    os.remove(os.path.join(root, f"{game_id}.pkl"))
        
        # Remove metadata
        metadata_file = f"{self.base_dir}/metadata/{game_id}.json"
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
    
    def _serialize_movement_rule(self, rule) -> Dict:
        """Serialize movement rule for storage"""
        return {
            'type': rule.__class__.__name__,
            'parameters': getattr(rule, '__dict__', {})
        }
    
    def _serialize_win_condition(self, condition) -> Dict:
        """Serialize win condition for storage"""
        return {
            'type': condition.__class__.__name__,
            'parameters': getattr(condition, '__dict__', {})
        }
    
    def _serialize_game_state(self, state: GameState) -> Dict:
        """Serialize game state for export"""
        serialized = {
            'detective_positions': state.detective_positions,
            'MrX_position': state.MrX_position,
            'turn': state.turn.value,
            'turn_count': state.turn_count,
            'mr_x_visible': getattr(state, 'mr_x_visible', True),
            'double_move_active': getattr(state, 'double_move_active', False)
        }
        
        # Add ticket information if available
        if hasattr(state, 'detective_tickets'):
            serialized['detective_tickets'] = {
                i: {k.value if hasattr(k, 'value') else str(k): v for k, v in tickets.items()}
                for i, tickets in state.detective_tickets.items()
            }
        
        if hasattr(state, 'mr_x_tickets'):
            serialized['mr_x_tickets'] = {
                k.value if hasattr(k, 'value') else str(k): v 
                for k, v in state.mr_x_tickets.items()
            }
        
        if hasattr(state, 'ticket_history'):
            serialized['ticket_history'] = state.ticket_history
            
        return serialized
    
    def _export_to_csv_with_tickets(self, record: GameRecord, filename: str):
        """Export game history to CSV format using stored ticket_history"""
        return False
    
    
    def _matches_filter(self, metadata: Dict, filter_by: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        if not filter_by:
            return True
        
        for key, value in filter_by.items():
            if key not in metadata or metadata[key] != value:
                return False
        
        return True
    

    def _export_to_csv(self, record: GameRecord, filename: str):
        """Export game history to CSV format"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Turn', 'Player', 'detective_Positions', 'MrX_Position'])
            
            for i, state in enumerate(record.game_history):
                writer.writerow([
                    i,
                    state.turn.value,
                    ','.join(map(str, state.detective_positions)),
                    state.MrX_position
                ])
    
    def _reconstruct_game_from_record(self, record: GameRecord) -> Game:
        """Reconstruct Game object from saved record"""
        # Reconstruct graph with explicit edges parameter to avoid warning
        graph = nx.node_link_graph(record.game_config['graph_data'], edges='links')
        
        # Create game with basic configuration
        game = ShadowChaseGame(graph, record.game_config['num_detectives'])
        
        # Restore game history
        if record.game_history:
            initial_state = record.game_history[0]
            game.initialize_game(initial_state.detective_positions, initial_state.MrX_position)
            game.game_history = record.game_history.copy()
            game.game_state = record.game_history[-1].copy()
            game.ticket_history = record.ticket_history.copy()
        
        return game

    def delete_all_games(self, confirm: bool = False) -> bool:
        """Delete all saved games and metadata with confirmation"""
        if not confirm:
            print("⚠️  WARNING: This will delete ALL saved games and metadata!")
            print("This action cannot be undone.")
            response = input("Are you sure you want to continue? Type 'DELETE ALL' to confirm: ").strip()
            
            if response != "DELETE ALL":
                print("Operation cancelled.")
                return False
        
        try:
            deleted_count = 0
            
            # Delete all .pkl files in games directory and subdirectories
            games_dir = f"{self.base_dir}/games"
            if os.path.exists(games_dir):
                for root, dirs, files in os.walk(games_dir):
                    for file in files:
                        if file.endswith('.pkl'):
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            deleted_count += 1
            
            # Delete all metadata files
            metadata_dir = f"{self.base_dir}/metadata"
            if os.path.exists(metadata_dir):
                for file in os.listdir(metadata_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(metadata_dir, file))
            
            # Delete all export files
            exports_dir = f"{self.base_dir}/exports"
            if os.path.exists(exports_dir):
                for file in os.listdir(exports_dir):
                    os.remove(os.path.join(exports_dir, file))
            
            # Delete all statistics files
            stats_dir = f"{self.base_dir}/statistics"
            if os.path.exists(stats_dir):
                for file in os.listdir(stats_dir):
                    os.remove(os.path.join(stats_dir, file))
            
            # Clean up empty subdirectories
            self._clean_empty_directories()
            
            print(f"✅ Successfully deleted {deleted_count} games and all associated metadata.")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting games: {e}")
            return False

    def _clean_empty_directories(self):
        """Remove empty subdirectories after bulk deletion"""
        subdirs_to_clean = [
            f"{self.base_dir}/games/by_date",
            f"{self.base_dir}/games/by_graph_type", 
            f"{self.base_dir}/games/by_outcome"
        ]
        
        for subdir in subdirs_to_clean:
            if os.path.exists(subdir):
                # Walk bottom-up to remove empty directories
                for root, dirs, files in os.walk(subdir, topdown=False):
                    try:
                        if not dirs and not files:  # Directory is empty
                            os.rmdir(root)
                    except OSError:
                        pass  # Directory not empty or other issue

