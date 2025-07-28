"""
DQN agents for Scotland Yard.

These agents use trained Deep Q-Networks for action selection.
They can be loaded from saved models and used in games.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Set
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ScotlandYard.core.game import ScotlandYardGame, Player, TransportType
from agents.base_agent import MrXAgent, MultiDetectiveAgent, DetectiveAgent
from training.feature_extractor import GameFeatureExtractor, FeatureConfig
from training.deep_q.dqn_model import create_dqn_model
from training.deep_q.dqn_model import DQNModel


class DQNMrXAgent(MrXAgent):
    """
    DQN-based Mr. X agent that loads a trained model.
    """
    
    def __init__(self, model_path: Optional[str] = None, epsilon: float = 0.05):
        """
        Initialize DQN Mr. X agent.
        
        Args:
            model_path: Path to trained model file (.pth)
            epsilon: Exploration rate for epsilon-greedy action selection
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agents. Please install PyTorch.")
        
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and feature extractor will be loaded
        self.model = None
        self.feature_extractor = None
        
        # Find and load model
        if model_path is None:
            model_path = self._find_latest_model("mr_x")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: No trained DQN model found at {model_path}")
            print("Agent will use random moves until a model is loaded.")
    
    def _find_latest_model(self, player_role: str) -> Optional[str]:
        """Find the latest trained model for the given player role."""
        model_dir = Path("training_results")
        if not model_dir.exists():
            return None
        
        # Look for DQN models for this player role
        pattern = f"dqn_{player_role}_*.pth"
        models = list(model_dir.glob(pattern))
        
        if models:
            # Return the most recently modified
            latest_model = max(models, key=lambda p: p.stat().st_mtime)
            return str(latest_model)
        
        return None
    
    def load_model(self, model_path: str):
        """Load a trained DQN model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load configuration
            config = checkpoint['config']
            
            # Initialize feature extractor
            feature_extraction_config = config.get('feature_extraction', {})
            # Remove input_size if present (it's added during training but not part of FeatureConfig)
            feature_extraction_config = {k: v for k, v in feature_extraction_config.items() if k != 'input_size'}
            feature_config = FeatureConfig(**feature_extraction_config)
            self.feature_extractor = GameFeatureExtractor(feature_config)
            
            # Create and load model
            
            self.model = create_dqn_model(config).to(self.device)
            self.model.load_state_dict(checkpoint['main_network'])
            self.model.eval()
            
            print(f"✅ Loaded DQN model from {model_path}")
            
            # Print training stats if available
            if 'training_stats' in checkpoint:
                stats = checkpoint['training_stats']
                print(f"   Training episodes: {len(stats.get('episode_rewards', []))}")
                print(f"   Final epsilon: {stats.get('final_epsilon', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            self.model = None
            self.feature_extractor = None
    
    def choose_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType, bool]]:
        """
        Choose a move using the trained DQN model.
        
        Args:
            game: Current game state
            
        Returns:
            Tuple of (destination, transport, use_double_move) or None
        """
        # Fallback to random if model not loaded
        if self.model is None or self.feature_extractor is None:
            return self._random_move(game)
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(game, self.player)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get valid moves
            valid_moves = game.get_valid_moves(Player.MRX)
            if not valid_moves:
                return None
            
            # Select action using epsilon-greedy
            with torch.no_grad():
                dest, transport = self.model.select_action(features_tensor, valid_moves, epsilon=0.0)
            
            # Decide on double move (simple heuristic)
            use_double_move = False
            if hasattr(game, 'can_use_double_move') and game.can_use_double_move():
                # Use double move 10% of the time when available
                use_double_move = np.random.random() < 0.1
            
            return (dest, transport, use_double_move)
            
        except Exception as e:
            print(f"Error in DQN move selection: {e}")
            return self._random_move(game)
    
    def _random_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType, bool]]:
        """Fallback random move selection."""
        valid_moves = game.get_valid_moves(Player.MRX)
        if not valid_moves:
            return None
        
        dest, transport = np.random.choice(list(valid_moves))
        use_double_move = False
        
        if hasattr(game, 'can_use_double_move') and game.can_use_double_move():
            use_double_move = np.random.random() < 0.05  # 5% chance
        
        return (dest, transport, use_double_move)


class DQNDetectiveAgent(DetectiveAgent):
    """
    DQN-based detective agent for a single detective.
    """
    
    def __init__(self, detective_id: int, model_path: Optional[str] = None, epsilon: float = 0.05):
        """
        Initialize DQN detective agent.
        
        Args:
            detective_id: ID of this detective (0, 1, 2, ...)
            model_path: Path to trained model file
            epsilon: Exploration rate
        """
        super().__init__(detective_id)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agents. Please install PyTorch.")
        
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and feature extractor
        self.model = None
        self.feature_extractor = None
        
        # Find and load model
        if model_path is None:
            model_path = self._find_latest_model("detectives")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: No trained DQN detective model found at {model_path}")
    
    def _find_latest_model(self, player_role: str) -> Optional[str]:
        """Find the latest trained model for detectives."""
        model_dir = Path("training_results")
        if not model_dir.exists():
            return None
        
        pattern = f"dqn_{player_role}_*.pth"
        models = list(model_dir.glob(pattern))
        
        if models:
            latest_model = max(models, key=lambda p: p.stat().st_mtime)
            return str(latest_model)
        
        return None
    
    def load_model(self, model_path: str):
        """Load a trained DQN model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            # Initialize feature extractor
            feature_extraction_config = config.get('feature_extraction', {})
            # Remove input_size if present (it's added during training but not part of FeatureConfig)
            feature_extraction_config = {k: v for k, v in feature_extraction_config.items() if k != 'input_size'}
            feature_config = FeatureConfig(**feature_extraction_config)
            self.feature_extractor = GameFeatureExtractor(feature_config)
            
            
            self.model = create_dqn_model(config).to(self.device)
            self.model.load_state_dict(checkpoint['main_network'])
            self.model.eval()
            
            print(f"✅ Loaded DQN detective model from {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load detective model: {e}")
            self.model = None
            self.feature_extractor = None
    
    def choose_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType]]:
        """Choose move for this detective using DQN."""
        if self.model is None or self.feature_extractor is None:
            return self._random_move(game)
        
        try:
            # Get current position
            position = self.get_current_position(game)
            
            # Get valid moves
            valid_moves = game.get_valid_moves(Player.DETECTIVES, position)
            if not valid_moves:
                return None
            
            # Extract features
            features = self.feature_extractor.extract_features(game, self.player)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Select action
            with torch.no_grad():
                dest, transport = self.model.select_action(features_tensor, valid_moves, epsilon=0.0)
            
            return (dest, transport)
            
        except Exception as e:
            print(f"Error in DQN detective move selection: {e}")
            return self._random_move(game)
    
    def _random_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType]]:
        """Fallback random move."""
        position = self.get_current_position(game)
        valid_moves = game.get_valid_moves(Player.DETECTIVES, position)
        if not valid_moves:
            return None
        return np.random.choice(list(valid_moves))


class DQNMultiDetectiveAgent(MultiDetectiveAgent):
    """
    DQN-based agent that controls multiple detectives.
    """
    
    def __init__(self, num_detectives: int = 2, model_path: Optional[str] = None, epsilon: float = 0.05):
        """
        Initialize DQN multi-detective agent.
        
        Args:
            num_detectives: Number of detectives to control
            model_path: Path to trained model
            epsilon: Exploration rate
        """
        super().__init__(num_detectives)
        
        # Create individual detective agents
        self.detective_agents = [
            DQNDetectiveAgent(i, model_path, epsilon) 
            for i in range(num_detectives)
        ]
    
    def choose_moves(self, game: ScotlandYardGame) -> List[Tuple[int, TransportType]]:
        """
        Choose moves for all detectives.
        
        Args:
            game: Current game state
            
        Returns:
            List of (destination, transport) tuples for each detective
        """
        moves = []
        for agent in self.detective_agents:
            move = agent.choose_move(game)
            if move is not None:
                moves.append(move)
            else:
                # Stay in place if no valid move
                position = agent.get_current_position(game)
                moves.append((position, TransportType.TAXI))
        
        return moves


# Factory functions for agent registry
def create_dqn_mr_x_agent(model_path: Optional[str] = None) -> DQNMrXAgent:
    """Create a DQN Mr. X agent."""
    return DQNMrXAgent(model_path)


def create_dqn_multi_detective_agent(num_detectives: int, model_path: Optional[str] = None) -> DQNMultiDetectiveAgent:
    """Create a DQN multi-detective agent."""
    return DQNMultiDetectiveAgent(num_detectives, model_path)
