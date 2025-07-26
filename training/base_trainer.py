"""
Base trainer abstract class and data structures for Scotland Yard AI training.

This module provides the foundational classes that all training algorithms
(MCTS, Deep Q-Learning, etc.) should inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
from pathlib import Path

from cops_and_robbers.core.game import ScotlandYardGame, Player
from .utils.evaluation import AgentEvaluator, EvaluationConfig
from agents.base_agent import Agent
import pickle

@dataclass
class TrainingResult:
    """Results from a training session."""
    algorithm: str
    total_episodes: int
    training_duration: float  # seconds
    final_performance: Dict[str, float]  # win rates, avg game length, etc.
    training_history: List[Dict[str, Any]]  # episode-by-episode metrics
    model_path: Optional[str] = None
    
    def save_to_file(self, filepath: str) -> None:
        """Save training results to JSON file."""
        data = {
            'algorithm': self.algorithm,
            'total_episodes': self.total_episodes,
            'training_duration': self.training_duration,
            'final_performance': self.final_performance,
            'training_history': self.training_history,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingResult':
        """Load training results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            algorithm=data['algorithm'],
            total_episodes=data['total_episodes'],
            training_duration=data['training_duration'],
            final_performance=data['final_performance'],
            training_history=data['training_history'],
            model_path=data.get('model_path')
        )


@dataclass
class EvaluationResult:
    """Results from evaluating a trained agent."""
    agent_name: str
    opponent_agent: str
    total_games: int
    games_won: int
    games_lost: int
    win_rate: float
    avg_game_length: float
    detailed_results: List[Dict[str, Any]]
    
    def save_to_file(self, filepath: str) -> None:
        """Save evaluation results to JSON file."""
        data = {
            'agent_name': self.agent_name,
            'opponent_agent': self.opponent_agent,
            'total_games': self.total_games,
            'games_won': self.games_won,
            'games_lost': self.games_lost,
            'win_rate': self.win_rate,
            'avg_game_length': self.avg_game_length,
            'detailed_results': self.detailed_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class BaseTrainer(ABC):
    """
    Abstract base class for all training algorithms.
    
    This class defines the interface that all trainers (MCTS, DQN, etc.)
    must implement, providing a consistent API for training and evaluation.
    """
    
    def __init__(self, 
                 algorithm_name: str,
                 save_dir: str = "training_results",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base trainer.
        
        Args:
            algorithm_name: Name of the training algorithm
            save_dir: Directory to save training results and models
            config: Algorithm-specific configuration parameters
        """
        self.algorithm_name = algorithm_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.current_episode = 0
        
    @abstractmethod
    def train(self, 
              num_episodes: int,
              map_size: str = "test",
              num_detectives: int = 2,
              max_turns_per_game: int = 24,
              **kwargs) -> TrainingResult:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            map_size: Game map size ("test", "full", "extracted")
            num_detectives: Number of detective agents
            max_turns_per_game: Maximum turns per game
            **kwargs: Additional training parameters
            
        Returns:
            TrainingResult object with training metrics and history
        """
        pass
    
    @abstractmethod
    def get_trained_agent(self, player: Player) -> Agent:
        """
        Get a trained agent for the specified player.
        
        Args:
            player: Player type (ROBBER for Mr. X, COPS for detectives)
            
        Returns:
            Trained agent instance
            
        Raises:
            RuntimeError: If training hasn't been completed
        """
        pass
    
    def evaluate(self,
                 num_games: int,
                 opponent_agent_type: str = "random",
                 map_size: str = "test",
                 num_detectives: int = 2,
                 max_turns_per_game: int = 24) -> EvaluationResult:
        """
        Evaluate the trained agent against a baseline opponent.
        
        Args:
            num_games: Number of evaluation games
            opponent_agent_type: Type of opponent ("random", "heuristic")
            map_size: Game map size
            num_detectives: Number of detective agents
            max_turns_per_game: Maximum turns per game
            
        Returns:
            EvaluationResult with performance metrics
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before evaluation")
        
        # Import here to avoid circular dependencies
        
        
        # Determine which player the trained agent plays
        trained_agent_mr_x = self.get_trained_agent(Player.ROBBER)
        trained_agent_det = self.get_trained_agent(Player.COPS)
        
        if trained_agent_mr_x is not None:
            player_role = "mr_x"
            trained_agent = trained_agent_mr_x
        elif trained_agent_det is not None:
            player_role = "detectives"
            trained_agent = trained_agent_det
        else:
            raise RuntimeError("No trained agent available for either role")
        
        # Create evaluator and run evaluation
        eval_config = EvaluationConfig(
            num_games=num_games,
            map_size=map_size,
            num_detectives=num_detectives,
            max_turns_per_game=max_turns_per_game,
            baseline_agents=[opponent_agent_type]
        )
        
        evaluator = AgentEvaluator(eval_config)
        eval_results = evaluator.evaluate_agent(trained_agent, player_role, opponent_agent_type)
        
        # Convert to EvaluationResult format
        return EvaluationResult(
            agent_name=f"{self.algorithm_name}_agent",
            opponent_agent=opponent_agent_type,
            total_games=eval_results['total_games'],
            games_won=eval_results['agent_wins'],
            games_lost=eval_results['baseline_wins'],
            win_rate=eval_results['agent_win_rate'],
            avg_game_length=eval_results['avg_game_length'],
            detailed_results=[r.__dict__ for r in eval_results['results']]
        )
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_name: Optional custom name for the model file
            
        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.algorithm_name}_model_{timestamp}"
        
        model_path = self.save_dir / f"{model_name}.pkl"
        
        # Subclasses should override this method to save their specific model data
        # This is a placeholder implementation

        model_data = {
            'algorithm': self.algorithm_name,
            'config': self.config,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        # Subclasses should override this method to load their specific model data
        import pickle
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.algorithm_name = model_data['algorithm']
        self.config = model_data['config']
        self.training_history = model_data.get('training_history', [])
        self.is_trained = model_data.get('is_trained', False)
    
    def _log_training_step(self, episode: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a training step.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metrics to log
        """
        step_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.training_history.append(step_data)
        
        # Print progress every 100 episodes or at the end
        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode}: {metrics}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training progress.
        
        Returns:
            Dictionary with training summary statistics
        """
        if not self.training_history:
            return {}
        
        total_episodes = len(self.training_history)
        
        # Calculate average metrics from recent episodes
        recent_episodes = self.training_history[-min(100, total_episodes):]
        
        summary = {
            'total_episodes': total_episodes,
            'algorithm': self.algorithm_name,
            'is_trained': self.is_trained,
            'recent_performance': {}
        }
        
        # Calculate averages for numeric metrics from recent episodes
        if recent_episodes:
            numeric_keys = set()
            for episode in recent_episodes:
                for key, value in episode.items():
                    if isinstance(value, (int, float)) and key != 'episode':
                        numeric_keys.add(key)
            
            for key in numeric_keys:
                values = [ep[key] for ep in recent_episodes if key in ep]
                if values:
                    summary['recent_performance'][f'avg_{key}'] = sum(values) / len(values)
        
        return summary
