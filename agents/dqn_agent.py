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

import torch


from ScotlandYard.core.game import ScotlandYardGame, Player, TransportType
from training.feature_extractor_simple import GameFeatureExtractor, FeatureConfig
from training.deep_q.dqn_model import create_dqn_model

from agents.base_agent import MrXAgent, MultiDetectiveAgent, DetectiveAgent

class DQNMrXAgent(MrXAgent):
    """
    DQN-based Mr. X agent that can work in training or inference mode.
    """
    
    def __init__(self, model_path: Optional[str] = None, trainer=None, epsilon: float = 0):
        """
        Initialize DQN Mr. X agent.
        
        Args:
            model_path: Path to trained model file (.pth) - for inference mode
            trainer: DQNTrainer instance - for training mode
            epsilon: Exploration rate for epsilon-greedy action selection
        """
        super().__init__()

        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        
        # Model and feature extractor will be loaded
        self.model = None
        self.feature_extractor = None
        self.trainer = trainer
        self.training_mode = trainer is not None
        
        if self.training_mode:
            # Training mode: use trainer's components
            self.model = trainer.main_network
            self.feature_extractor = trainer.feature_extractor
            # print("🎯 DQN Mr. X agent initialized in training mode")
        else:
            # Inference mode: load from saved model
            if model_path is None:
                model_path = self._find_latest_model("mr_x")
            
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
            else:
                print(f"Warning: No trained DQN model found at {model_path}")
                print("Agent will use random moves until a model is loaded.")
    

    def finalize_episode(self, game: ScotlandYardGame, final_reward: float):
        """Store the final experience for the episode with the actual reward."""
        if self.training_mode and hasattr(self, '_previous_state'):
            # Extract current final state features
            final_state_features = self.feature_extractor.extract_features(game, self.player)
            
            # Store final experience with the actual episode reward
            self.trainer.replay_buffer.push(
                state=self._previous_state,
                action=self._previous_action,
                reward=final_reward,
                next_state=final_state_features,
                done=True,
                next_valid_moves=set()  # No valid moves in terminal state
            )
            
            # Clear previous state
            delattr(self, '_previous_state')
            delattr(self, '_previous_action')
            delattr(self, '_previous_valid_moves')
    
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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
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
            
            # print(f"✅ Loaded DQN model from {model_path}")
            
            # Print training stats if available
            # if 'training_stats' in checkpoint:
            #     stats = checkpoint['training_stats']
            #     print(f"   Training episodes: {len(stats.get('episode_rewards', []))}")
            #     print(f"   Final epsilon: {stats.get('final_epsilon', 'N/A')}")
                
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
            # Extract features from current state
            current_state_features = self.feature_extractor.extract_features(game, self.player)
            features_tensor = torch.FloatTensor(current_state_features).unsqueeze(0).to(self.device)
            
            # Get valid moves
            valid_moves = game.get_valid_moves(Player.MRX)
            if not valid_moves:
                return None
            
            # Store previous state if we have one (for experience collection)
            if self.training_mode and hasattr(self, '_previous_state'):
                # We have a previous state, so we can create an experience
                
                # Calculate reward (simple step reward - the final reward will be assigned at episode end)
                step_reward = 0.0
                
                # Check if game ended
                done = game.is_game_over()
                
                # Get current valid moves for next_valid_moves

                
                # Store experience in replay buffer
                self.trainer.replay_buffer.push(
                    state=self._previous_state,
                    action=self._previous_action,
                    reward=step_reward,
                    next_state=current_state_features,
                    done=done,
                    next_valid_moves=valid_moves
                )

            
            # Select action using epsilon-greedy
            # Use trainer's epsilon if in training mode, otherwise use agent's epsilon
            current_epsilon = self.trainer.current_epsilon if self.training_mode else self.epsilon
            
            with torch.no_grad():
                dest, transport = self.model.select_action(features_tensor, valid_moves, epsilon=current_epsilon)
            
            # Store current state and action for next experience
            if self.training_mode:
                self._previous_state = current_state_features
                self._previous_action = (dest, transport)
            
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
        
        chosen_move = np.random.choice(len(valid_moves))
        dest, transport = list(valid_moves)[chosen_move]
        use_double_move = False
        
        if hasattr(game, 'can_use_double_move') and game.can_use_double_move():
            use_double_move = np.random.random() < 0.05  # 5% chance
        
        return (dest, transport, use_double_move)


class DQNDetectiveAgent(DetectiveAgent):
    """
    DQN-based detective agent for a single detective.
    """
    
    def __init__(self, detective_id: int, model_path: Optional[str] = None, trainer=None, epsilon: float = 0):
        """
        Initialize DQN detective agent.
        
        Args:
            detective_id: ID of this detective (0, 1, 2, ...)
            model_path: Path to trained model file - for inference mode
            trainer: DQNTrainer instance - for training mode
            epsilon: Exploration rate
        """
        super().__init__(detective_id)
        
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        
        # Model and feature extractor
        self.model = None
        self.feature_extractor = None
        self.trainer = trainer
        self.training_mode = trainer is not None
        
        if self.training_mode:
            # Training mode: use trainer's components
            self.model = trainer.main_network
            self.feature_extractor = trainer.feature_extractor
            # print(f"🎯 DQN Detective agent {detective_id} initialized in training mode")
        else:
            # Inference mode: load from saved model
            if model_path is None:
                model_path = self._find_latest_model("detectives")
            
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
            else:
                print(f"Warning: No trained DQN detective model found at {model_path}")
    
    def store_experience(self, state, action, reward, next_state, done, next_valid_moves):
        """Store experience in trainer's replay buffer (only in training mode)."""
        if self.training_mode and self.trainer:
            self.trainer.replay_buffer.push(state, action, reward, next_state, done, next_valid_moves)
    
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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
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
            
            # print(f"✅ Loaded DQN detective model from {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load detective model: {e}")
            self.model = None
            self.feature_extractor = None
    
    def choose_move(self, game: ScotlandYardGame, pending_moves) -> Optional[Tuple[int, TransportType]]:
        """Choose move for this detective using DQN."""
        if self.model is None or self.feature_extractor is None:
            return self._random_move(game)
        
        try:
            # Get current position
            position = self.get_current_position(game)
            
            # Get valid moves
            valid_moves = game.get_valid_moves(Player.DETECTIVES, position, pending_moves)
            if not valid_moves:
                return None
            
            # Extract features
            current_state_features = self.feature_extractor.extract_features(game, self.player)
            features_tensor = torch.FloatTensor(current_state_features).unsqueeze(0).to(self.device)

            if self.training_mode and hasattr(self, '_previous_state') and hasattr(self, '_previous_action'):
                # We have a previous state, so we can create an experience
                
                # Calculate reward (simple step reward - the final reward will be assigned at episode end)
                step_reward = 0.0
                
                # Check if game ended
                done = game.is_game_over()
                
                # Get current valid moves for next_valid_moves
                self.trainer.replay_buffer.push(
                    state=self._previous_state,
                    action=self._previous_action,
                    reward=step_reward,
                    next_state=current_state_features,
                    done=done,
                    next_valid_moves=valid_moves
                )

            # Select action
            # Use trainer's epsilon if in training mode, otherwise use agent's epsilon
            current_epsilon = self.trainer.current_epsilon if self.training_mode else self.epsilon
            
            with torch.no_grad():
                dest, transport = self.model.select_action(features_tensor, valid_moves, epsilon=current_epsilon)
                
            if self.training_mode:
                self._previous_state = current_state_features
                self._previous_action = (dest, transport)

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
        chosen_move = np.random.choice(len(valid_moves))
        return list(valid_moves)[chosen_move]


class DQNMultiDetectiveAgent(MultiDetectiveAgent):
    """
    DQN-based agent that controls multiple detectives.
    """
    
    def __init__(self, num_detectives: int = 2, model_path: Optional[str] = None, trainer=None, epsilon: float = 0):
        """
        Initialize DQN multi-detective agent.
        
        Args:
            num_detectives: Number of detectives to control
            model_path: Path to trained model - for inference mode
            trainer: DQNTrainer instance - for training mode
            epsilon: Exploration rate
        """
        super().__init__(num_detectives)
        
        # Create individual detective agents
        self.detective_agents = [
            DQNDetectiveAgent(i, model_path, trainer, epsilon) 
            for i in range(num_detectives)
        ]
    
    def choose_all_moves(self, game: ScotlandYardGame) -> List[Tuple[int, TransportType]]:
        """
        Choose moves for all detectives.
        
        Args:
            game: Current game state
            
        Returns:
            List of (destination, transport) tuples for each detective
        """
        moves = []
        pending_moves = []
        for agent in self.detective_agents:
            move = agent.choose_move(game, pending_moves)
            if move is not None:
                moves.append(move)
                pending_moves.append(move)
            else:
                # Stay in place if no valid move
                move = (agent.get_current_position(game), None)
                moves.append(move)
            pending_moves.append(move)
        return moves


# Factory functions for agent registry
def create_dqn_mr_x_agent(model_path: Optional[str] = None, trainer=None) -> DQNMrXAgent:
    """Create a DQN Mr. X agent."""
    return DQNMrXAgent(model_path, trainer)


def create_dqn_multi_detective_agent(num_detectives: int, model_path: Optional[str] = None, trainer=None) -> DQNMultiDetectiveAgent:
    """Create a DQN multi-detective agent."""
    return DQNMultiDetectiveAgent(num_detectives, model_path, trainer)
