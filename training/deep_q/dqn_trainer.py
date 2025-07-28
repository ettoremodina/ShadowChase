"""
Deep Q-Network trainer for Scotland Yard.

This module implements the DQN training algorithm using the existing training infrastructure.
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DQN training will not work.")

from ScotlandYard.core.game import ScotlandYardGame, Player, TransportType
from training.base_trainer import BaseTrainer, TrainingResult
from training.feature_extractor import GameFeatureExtractor, FeatureConfig
from training.utils.training_environment import TrainingEnvironment
from agents.base_agent import Agent, MrXAgent, DetectiveAgent

if TORCH_AVAILABLE:
    from .dqn_model import DQNModel, create_dqn_model
    from .replay_buffer import ReplayBuffer, create_replay_buffer


class DQNTrainer(BaseTrainer):
    """
    Deep Q-Network trainer for Scotland Yard agents.
    
    Implements DQN with experience replay and target networks.
    """
    
    def __init__(self, 
                 player_role: str = "mr_x",  # "mr_x" or "detectives"
                 config_path: str = "training/configs/dqn_config.json",
                 save_dir: str = "training_results"):
        """
        Initialize DQN trainer.
        
        Args:
            player_role: Which player the agent will control ("mr_x" or "detectives")
            config_path: Path to DQN configuration file
            save_dir: Directory to save training results
        """
        super().__init__("dqn", save_dir)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN training. Please install PyTorch.")
        
        self.player_role = player_role
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize training parameters
        training_params = self.config.get('training_parameters', {})
        self.batch_size = training_params.get('batch_size', 32)
        self.learning_rate = training_params.get('learning_rate', 0.001)
        self.gamma = training_params.get('gamma', 0.95)
        self.epsilon_start = training_params.get('epsilon_start', 1.0)
        self.epsilon_end = training_params.get('epsilon_end', 0.01)
        self.epsilon_decay = training_params.get('epsilon_decay', 0.995)
        self.target_update_frequency = training_params.get('target_update_frequency', 100)
        self.min_replay_size = training_params.get('min_replay_buffer_size', 1000)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize feature extractor
        feature_config = FeatureConfig(**self.config.get('feature_extraction', {}))
        self.feature_extractor = GameFeatureExtractor(feature_config)
        
        # Initialize components (will be created in train())
        self.main_network = None
        self.target_network = None
        self.optimizer = None
        self.replay_buffer = None
        self.current_epsilon = self.epsilon_start
        
        # Training state
        self.step_count = 0
        self.episode_rewards = []
        self.losses = []
    
    def _initialize_networks(self, sample_game: ScotlandYardGame):
        """Initialize the neural networks based on a sample game."""
        # Get feature size from a sample state
        feature_size = self.feature_extractor.get_feature_size(sample_game)
        print(f"Feature vector size: {feature_size}")
        
        # Update config with actual feature size
        self.config['feature_extraction']['input_size'] = feature_size
        
        # Create networks
        self.main_network = create_dqn_model(self.config).to(self.device)
        self.target_network = create_dqn_model(self.config).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = create_replay_buffer(self.config)
        
        print(f"Initialized DQN with {sum(p.numel() for p in self.main_network.parameters())} parameters")
    
    def train(self, 
              num_episodes: int,
              map_size: str = "test",
              num_detectives: int = 2,
              max_turns_per_game: int = 24,
              **kwargs) -> TrainingResult:
        """
        Train the DQN agent.
        
        Args:
            num_episodes: Number of training episodes
            map_size: Game map size
            num_detectives: Number of detective agents
            max_turns_per_game: Maximum turns per game
            
        Returns:
            TrainingResult with training metrics
        """
        start_time = time.time()
        
        # Create training environment
        env = TrainingEnvironment(
            map_size=map_size,
            num_detectives=num_detectives,
            max_turns=max_turns_per_game
        )
        
        # Create a sample game to initialize networks
        from simple_play.game_utils import create_and_initialize_game
        sample_game = create_and_initialize_game(map_size, num_detectives)
        
        # Initialize networks if not already done
        if self.main_network is None:
            self._initialize_networks(sample_game)
        
        # Create baseline opponent
        from agents import AgentType, get_agent_registry
        registry = get_agent_registry()
        
        if self.player_role == "mr_x":
            opponent_agent = registry.create_multi_detective_agent(AgentType.RANDOM, num_detectives)
        else:
            opponent_agent = registry.create_mr_x_agent(AgentType.RANDOM)
        
        print(f"\n🚀 Starting DQN training for {self.player_role}")
        print(f"Episodes: {num_episodes}, Map: {map_size}, Detectives: {num_detectives}")
        print("=" * 60)
        
        # Training loop
        for episode in range(num_episodes):
            episode_reward = self._train_episode(env, opponent_agent)
            self.episode_rewards.append(episode_reward)
            
            # Update target network
            if episode % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.main_network.state_dict())
            
            # Decay epsilon
            self.current_epsilon = max(
                self.epsilon_end, 
                self.current_epsilon * self.epsilon_decay
            )
            
            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_loss = np.mean(self.losses[-50:]) if self.losses else 0
                print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | "
                      f"Epsilon: {self.current_epsilon:.3f} | Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.replay_buffer)}")
            
            # Log training step
            self._log_training_step(episode, {
                'episode_reward': episode_reward,
                'epsilon': self.current_epsilon,
                'buffer_size': len(self.replay_buffer),
                'avg_loss': np.mean(self.losses[-10:]) if self.losses else 0
            })
        
        training_duration = time.time() - start_time
        
        # Calculate final performance
        final_performance = {
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'final_epsilon': self.current_epsilon,
            'total_steps': self.step_count,
            'final_buffer_size': len(self.replay_buffer)
        }
        
        print(f"\n✅ Training completed in {training_duration:.2f} seconds")
        print(f"Final average reward: {final_performance['avg_episode_reward']:.2f}")
        
        # Save model
        model_path = self.save_model()
        
        return TrainingResult(
            algorithm="dqn",
            total_episodes=num_episodes,
            training_duration=training_duration,
            final_performance=final_performance,
            training_history=self.training_history,
            model_path=model_path
        )
    
    def _train_episode(self, env: TrainingEnvironment, opponent_agent) -> float:
        """Train for a single episode."""
        episode_reward = 0.0
        
        # Create our training agent
        if self.player_role == "mr_x":
            our_agent = DQNAgent(self, Player.MRX)
            result, experiences = env.run_episode(our_agent, opponent_agent, collect_experience=True)
        else:
            our_agent = DQNAgent(self, Player.DETECTIVES)
            result, experiences = env.run_episode(opponent_agent, our_agent, collect_experience=True)
        
        # Process experiences and add to replay buffer
        episode_reward = self._process_episode_experiences(result, experiences)
        
        # Train on batch if we have enough experiences
        if len(self.replay_buffer) >= self.min_replay_size:
            loss = self._train_step()
            if loss is not None:
                self.losses.append(loss)
        
        return episode_reward
    
    def _process_episode_experiences(self, result, experiences) -> float:
        """Process episode experiences and add to replay buffer."""
        episode_reward = 0.0
        
        # Simple reward shaping based on game outcome
        if result.winner == self.player_role.replace("_", ""):
            final_reward = 10.0  # Win
        elif result.winner == "timeout":
            final_reward = 0.0   # Neutral
        else:
            final_reward = -10.0  # Loss
        
        episode_reward = final_reward
        
        # For now, just assign the final reward to the last step
        # In a more sophisticated implementation, we'd do reward shaping
        # throughout the episode
        
        return episode_reward
    
    def _train_step(self) -> Optional[float]:
        """Perform one training step using batch from replay buffer."""
        if not self.replay_buffer.can_sample(self.batch_size):
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = [exp.action for exp in experiences]
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        valid_moves = [exp.valid_moves for exp in experiences]
        next_valid_moves = [exp.next_valid_moves for exp in experiences]
        
        # Get current Q-values
        current_q_values = self.main_network.get_masked_q_values(states, valid_moves)
        action_indices = torch.LongTensor([
            self.main_network.get_action_index(dest, transport) 
            for dest, transport in actions
        ]).to(self.device)
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Get next Q-values
        with torch.no_grad():
            next_q_values = self.target_network.get_masked_q_values(next_states, next_valid_moves)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.step_count += 1
        return loss.item()
    
    def get_trained_agent(self, player: Player) -> Agent:
        """Get the trained agent for the specified player."""
        if self.main_network is None:
            raise RuntimeError("No trained model available. Train first.")
        
        if self.player_role == "mr_x" and player == Player.MRX:
            return DQNMrXAgent(self)
        elif self.player_role == "detectives" and player == Player.DETECTIVES:
            return DQNMultiDetectiveAgent(self)
        else:
            return None
    
    def find_latest_model(self, player_role: Optional[str] = None) -> Optional[str]:
        """
        Find the most recent trained model for the given player role.
        
        Args:
            player_role: Player role ("mr_x" or "detectives"). If None, uses self.player_role.
            
        Returns:
            Path to the most recent model file, or None if no models found.
        """
        if player_role is None:
            player_role = self.player_role
        
        # Look for DQN models with the naming pattern: dqn_{player_role}_{timestamp}.pth
        pattern = f"dqn_{player_role}_*.pth"
        models = list(self.save_dir.glob(pattern))
        
        if models:
            # Return the most recently modified
            latest_model = max(models, key=lambda p: p.stat().st_mtime)
            return str(latest_model)
        
        return None
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save the trained model."""
        if model_name is None:
            model_name = f"dqn_{self.player_role}_{int(time.time())}"
        
        model_path = self.save_dir / f"{model_name}.pth"
        
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'player_role': self.player_role,
            'training_stats': {
                'episode_rewards': self.episode_rewards,
                'losses': self.losses,
                'step_count': self.step_count,
                'final_epsilon': self.current_epsilon
            }
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        return str(model_path)


class DQNAgent(Agent):
    """Base DQN agent that uses trained model for action selection."""
    
    def __init__(self, trainer: DQNTrainer, player: Player):
        super().__init__(player)
        self.trainer = trainer
        self.epsilon = 0.05  # Small epsilon for inference
    
    def choose_move(self, game: ScotlandYardGame) -> Optional[Tuple]:
        """Choose move using trained DQN model."""
        # Extract features
        features = self.trainer.feature_extractor.extract_features(game, self.player)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.trainer.device)
        
        # Get valid moves
        if self.player == Player.MRX:
            valid_moves = game.get_valid_moves(Player.MRX)
            if not valid_moves:
                return None
            
            # Select action
            dest, transport = self.trainer.main_network.select_action(
                features_tensor, valid_moves, self.epsilon
            )
            
            # Check if double move is possible (simplified)
            can_double = game.can_use_double_move() if hasattr(game, 'can_use_double_move') else False
            use_double = can_double and random.random() < 0.1  # 10% chance to use double move
            
            return (dest, transport, use_double)
        
        else:  # Detective
            # For detective, we need to know which detective we are
            # This is a simplified implementation
            position = game.game_state.detective_positions[0]  # Use first detective
            valid_moves = game.get_valid_moves(Player.DETECTIVES, position)
            if not valid_moves:
                return None
            
            dest, transport = self.trainer.main_network.select_action(
                features_tensor, valid_moves, self.epsilon
            )
            return (dest, transport)


class DQNMrXAgent(MrXAgent, DQNAgent):
    """DQN agent for Mr. X."""
    
    def __init__(self, trainer: DQNTrainer):
        super().__init__()
        DQNAgent.__init__(self, trainer, Player.MRX)


class DQNMultiDetectiveAgent(DQNAgent):
    """DQN agent for multiple detectives."""
    
    def __init__(self, trainer: DQNTrainer):
        super().__init__(trainer, Player.DETECTIVES)
        self.num_detectives = 2  # Default
    
    def choose_moves(self, game: ScotlandYardGame) -> List[Tuple[int, TransportType]]:
        """Choose moves for all detectives."""
        moves = []
        for i in range(self.num_detectives):
            if i < len(game.game_state.detective_positions):
                position = game.game_state.detective_positions[i]
                valid_moves = game.get_valid_moves(Player.DETECTIVES, position)
                
                if valid_moves:
                    features = self.trainer.feature_extractor.extract_features(game, self.player)
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.trainer.device)
                    
                    dest, transport = self.trainer.main_network.select_action(
                        features_tensor, valid_moves, self.epsilon
                    )
                    moves.append((dest, transport))
                else:
                    # Stay in place if no valid moves
                    moves.append((position, TransportType.TAXI))
            
        return moves
