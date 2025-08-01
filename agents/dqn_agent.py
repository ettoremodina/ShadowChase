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


from ShadowChase.core.game import ScotlandYardGame, Player, TransportType
from training.feature_extractor_simple import GameFeatureExtractor, FeatureConfig
from training.deep_q.dqn_model import create_dqn_model
from agents.heuristics import GameHeuristics
from agents.base_agent import MrXAgent, MultiDetectiveAgent, DetectiveAgent

class DQNAgentMixin:
    """Mixin class for shared DQN agent functionality."""
    
    def _infer_model_config_from_state_dict(self, state_dict: dict) -> dict:
        """
        Attempt to infer model configuration from state dictionary.
        This is a fallback when config is not available in checkpoint.
        """
        try:
            # Get the first layer to infer input size
            first_layer_key = None
            for key in state_dict.keys():
                if 'feature_network.0.weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key is None:
                raise ValueError("Cannot find first layer in state dict")
            
            # Input size is the second dimension of the first layer weights
            input_size = state_dict[first_layer_key].shape[1]
            
            # Infer hidden layers by examining the network structure
            hidden_layers = []
            layer_idx = 0
            while True:
                weight_key = f'feature_network.{layer_idx * 3}.weight'  # Every 3rd layer is Linear
                if weight_key in state_dict:
                    hidden_layers.append(state_dict[weight_key].shape[0])
                    layer_idx += 1
                else:
                    break
            
            # Default action size (destination + transport)
            action_size = 2
            
            # State size is input size minus action size
            state_size = input_size - action_size
            
            # Create minimal config
            inferred_config = {
                'network_parameters': {
                    'action_size': action_size,
                    'hidden_layers': hidden_layers,
                    'dropout_rate': 0.1  # Default value
                },
                'feature_extraction': {
                    'input_size': state_size,
                    # Add some reasonable defaults for feature extraction
                    'include_positions': True,
                    'include_distances': True,
                    'include_transport_info': True,
                    'include_game_state': True
                }
            }
            
            print(f"⚠️  Inferred model config from state dict: state_size={state_size}, hidden_layers={hidden_layers}")
            return inferred_config
            
        except Exception as e:
            print(f"❌ Failed to infer model config from state dict: {e}")
            raise ValueError("Cannot infer model structure from state dict")
    
    def load_model(self, model_path: str):
        """Load a trained DQN model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Try to get configuration from checkpoint
            config = checkpoint.get('config')
            
            if config is None:
                print("⚠️  No config found in checkpoint, attempting to infer from state dict...")
                # Fallback: try to infer config from state dict
                config = self._infer_model_config_from_state_dict(checkpoint['main_network'])
            
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
            
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            self.model = None
            self.feature_extractor = None
    
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
            if hasattr(self, '_previous_valid_moves'):
                delattr(self, '_previous_valid_moves')


class DQNMrXAgent(MrXAgent, DQNAgentMixin):
    """
    DQN-based Mr. X agent that can work in training or inference mode.
    """
    
    def __init__(self, model_path: Optional[str] = None, trainer=None, epsilon: float = 0, device=None):
        """
        Initialize DQN Mr. X agent.
        
        Args:
            model_path: Path to trained model file (.pth) - for inference mode
            trainer: DQNTrainer instance - for training mode
            epsilon: Exploration rate for epsilon-greedy action selection
            device: PyTorch device to use (if None, will auto-detect)
        """
        super().__init__()

        self.epsilon = epsilon
        
        # Set device - use trainer's device if available, otherwise use provided device or auto-detect
        if trainer is not None:
            self.device = trainer.device
        elif device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    

    def _calculate_step_reward(self, game: ScotlandYardGame) -> float:
        """Calculate step-wise reward for Mr. X to encourage good behavior."""
        
        heuristics = GameHeuristics(game)
        min_distance = heuristics.get_minimum_distance_to_mr_x()
        
        # Reward staying far from detectives
        distance_reward = min_distance * 0.1
        
        # Small survival bonus
        survival_bonus = 0.01
        
        # Penalty if too close to detectives
        danger_penalty = 0.0
        if min_distance <= 2:
            danger_penalty = -0.5
        elif min_distance <= 1:
            danger_penalty = -1.0
        
        return distance_reward + survival_bonus + danger_penalty
    
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
        
        # try:
        # Extract features from current state
        current_state_features = self.feature_extractor.extract_features(game, self.player)
        features_tensor = torch.FloatTensor(current_state_features).unsqueeze(0).to(self.device)
        
        # Get valid moves
        valid_moves = game.get_valid_moves(Player.MRX)
        if not valid_moves:
            return (None, None, False)  # No valid moves
        
        # Check if double move is available
        can_use_double_move = game.can_use_double_move()
        
        # Store previous state if we have one (for experience collection)
        if self.training_mode and hasattr(self, '_previous_state'):
            # We have a previous state, so we can create an experience
            
            # Calculate step reward with better shaping
            step_reward = self._calculate_step_reward(game)
            
            # Check if game ended
            done = game.is_game_over()
            
            # Get current valid moves for next_valid_moves
            # Convert 2-tuples to 3-tuples for Mr. X
            if hasattr(self.model, 'action_size') and self.model.action_size == 3:
                # Convert valid moves to 3-tuples
                valid_moves_3d = set()
                for dest, transport in valid_moves:
                    valid_moves_3d.add((dest, transport, False))  # Default no double move
                    if can_use_double_move:
                        valid_moves_3d.add((dest, transport, True))   # With double move if available
                next_valid_moves = valid_moves_3d
            else:
                next_valid_moves = valid_moves
            
            # Store experience in replay buffer
            self.trainer.replay_buffer.push(
                state=self._previous_state,
                action=self._previous_action,
                reward=step_reward,
                next_state=current_state_features,
                done=done,
                next_valid_moves=next_valid_moves
            )

        
        # Select action using epsilon-greedy
        # Use trainer's epsilon if in training mode, otherwise use agent's epsilon
        current_epsilon = self.trainer.current_epsilon if self.training_mode else self.epsilon
        
        with torch.no_grad():
            action_result = self.model.select_action(
                features_tensor, 
                valid_moves, 
                epsilon=current_epsilon,
                can_use_double_move=can_use_double_move
            )
        
        dest, transport, use_double_move = action_result

        # Store current state and action for next experience
        if self.training_mode:
            self._previous_state = current_state_features
            self._previous_action = (dest, transport, use_double_move)
        
        return (dest, transport, use_double_move)
            
    
    def _random_move(self, game: ScotlandYardGame) -> Optional[Tuple[int, TransportType, bool]]:
        """Fallback random move selection."""
        valid_moves = game.get_valid_moves(Player.MRX)
        if not valid_moves:
            return None
        
        chosen_move = np.random.choice(len(valid_moves))
        dest, transport = list(valid_moves)[chosen_move]
        
        # Randomly decide on double move if available
        can_use_double_move = game.can_use_double_move()
        use_double_move = can_use_double_move and np.random.random() < 0.2  # 20% chance to use double move
        
        return (dest, transport, use_double_move)


class DQNDetectiveAgent(DetectiveAgent, DQNAgentMixin):
    """
    DQN-based detective agent for a single detective.
    """
    
    def __init__(self, detective_id: int, model_path: Optional[str] = None, trainer=None, epsilon: float = 0, device=None):
        """
        Initialize DQN detective agent.
        
        Args:
            detective_id: ID of this detective (0, 1, 2, ...)
            model_path: Path to trained model file - for inference mode
            trainer: DQNTrainer instance - for training mode
            epsilon: Exploration rate
            device: PyTorch device to use (if None, will auto-detect)
        """
        super().__init__(detective_id)
        
        # Store detective ID for reward calculations
        self.detective_id = detective_id
        
        self.epsilon = epsilon
        
        # Set device - use trainer's device if available, otherwise use provided device or auto-detect
        if trainer is not None:
            self.device = trainer.device
        elif device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
                
                # Calculate step reward with better shaping
                step_reward = self._calculate_step_reward(game)
                
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

            # Select action (detectives never use double moves)
            # Use trainer's epsilon if in training mode, otherwise use agent's epsilon
            current_epsilon = self.trainer.current_epsilon if self.training_mode else self.epsilon
            
            with torch.no_grad():
                action_result = self.model.select_action(
                    features_tensor, 
                    valid_moves, 
                    epsilon=current_epsilon
                )
                dest, transport = action_result
                
            if self.training_mode:
                self._previous_state = current_state_features
                self._previous_action = (dest, transport)

            return (dest, transport)
            
        except Exception as e:
            print(f"Error in DQN detective move selection: {e}")
            return self._random_move(game)
        
    def _calculate_step_reward(self, game: ScotlandYardGame) -> float:
        """Calculate step-wise reward for detective to encourage good behavior."""        
        heuristics = GameHeuristics(game)
        
        # Get this detective's current position
        detective_pos = self.get_current_position(game)
        
        # Get all possible Mr. X positions
        possible_mr_x_positions = heuristics.get_possible_mr_x_positions()
        
        if not possible_mr_x_positions:
            # If no possible positions available, return small penalty
            return -0.01
        
        # Calculate distances from this detective to all possible Mr. X positions
        distances_to_possible_positions = []
        for possible_pos in possible_mr_x_positions:
            distance = heuristics.calculate_shortest_distance(detective_pos, possible_pos)
            if distance >= 0:  # Only include valid paths
                distances_to_possible_positions.append(distance)
        
        if not distances_to_possible_positions:
            # No valid paths to any possible position - large penalty
            sum_distance_to_mr_x = 100  # Large penalty proportional to sum
        else:
            # Use sum of distances to all possible Mr. X positions
            sum_distance_to_mr_x = sum(distances_to_possible_positions)
        
        # Reward getting closer to possible Mr. X positions (inverse sum)
        distance_reward = -sum_distance_to_mr_x * 0.01  # Negative because we want smaller sum, scaled down since it's a sum
        
        # Small step penalty to encourage ending games quickly
        step_penalty = -0.01
        
        # Bonus for being very close to any possible Mr. X position (use min for proximity bonus)
        proximity_bonus = 0.0
        if distances_to_possible_positions:
            min_distance = min(distances_to_possible_positions)
            if min_distance <= 1:
                proximity_bonus = 1.0  # Very close - excellent!
            elif min_distance <= 2:
                proximity_bonus = 0.5  # Close - good
            elif min_distance <= 3:
                proximity_bonus = 0.2  # Reasonably close
        
        return distance_reward + step_penalty + proximity_bonus
    
    
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
    
    def __init__(self, num_detectives: int = 2, model_path: Optional[str] = None, trainer=None, epsilon: float = 0, device=None):
        """
        Initialize DQN multi-detective agent.
        
        Args:
            num_detectives: Number of detectives to control
            model_path: Path to trained model - for inference mode
            trainer: DQNTrainer instance - for training mode
            epsilon: Exploration rate
            device: PyTorch device to use (if None, will auto-detect)
        """
        super().__init__(num_detectives)
        
        # Create individual detective agents
        self.detective_agents = [
            DQNDetectiveAgent(i, model_path, trainer, epsilon, device) 
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

    def finalize_episode(self, game: ScotlandYardGame, final_reward: float):
        """Finalize episode for all detective agents."""
        for agent in self.detective_agents:
            agent.finalize_episode(game, final_reward)


# Factory functions for agent registry
def create_dqn_mr_x_agent(model_path: Optional[str] = None, trainer=None, device=None) -> DQNMrXAgent:
    """Create a DQN Mr. X agent."""
    return DQNMrXAgent(model_path, trainer, device=device)


def create_dqn_multi_detective_agent(num_detectives: int, model_path: Optional[str] = None, trainer=None, device=None) -> DQNMultiDetectiveAgent:
    """Create a DQN multi-detective agent."""
    return DQNMultiDetectiveAgent(num_detectives, model_path, trainer, device=device)
