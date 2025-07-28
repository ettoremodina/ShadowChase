"""
Agent evaluation utilities for Scotland Yard training.

This module provides functions for evaluating trained agents against
baseline opponents. It leverages the existing analyze_games.py infrastructure
for comprehensive analysis and reporting.
"""

from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

from agents import AgentType, get_agent_registry
from .training_environment import TrainingEnvironment


@dataclass
class GameResult:
    """Result of a single game episode."""
    winner: str  # "mr_x", "detectives", or "timeout"
    total_turns: int
    game_length: float  # seconds
    mr_x_final_position: int
    detective_final_positions: List[int]
    moves_history: List[Dict[str, Any]]

@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation."""
    num_games: int = 100
    map_size: str = "test"
    num_detectives: int = 2
    max_turns_per_game: int = 24
    baseline_agents: List[str] = None
    
    def __post_init__(self):
        if self.baseline_agents is None:
            self.baseline_agents = ["random", "heuristic"]


class AgentEvaluator:
    """
    Evaluates trained agents against baseline opponents.
    
    This class provides comprehensive evaluation of trained agents,
    leveraging the existing analyze_games.py infrastructure for
    detailed analysis and reporting.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the agent evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.registry = get_agent_registry()
        
    def evaluate_agent(self,
                      trained_agent,
                      agent_plays_as: str,  # "mr_x" or "detectives"
                      baseline_agent_type: str = "random") -> Dict[str, any]:
        """
        Evaluate a trained agent against a baseline opponent.
        
        Args:
            trained_agent: The trained agent to evaluate
            agent_plays_as: Whether the agent plays as "mr_x" or "detectives"
            baseline_agent_type: Type of baseline opponent
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {agent_plays_as} agent vs {baseline_agent_type} baseline...")
        
        # Set up environment
        env = TrainingEnvironment(
            map_size=self.config.map_size,
            num_detectives=self.config.num_detectives,
            max_turns=self.config.max_turns_per_game
        )
        
        # Create baseline agent
        baseline_agent_enum = AgentType(baseline_agent_type.lower())
        
        if agent_plays_as == "mr_x":
            # Trained agent is Mr. X, baseline agent is detectives
            mr_x_agent = trained_agent
            detective_agent = self.registry.create_multi_detective_agent(
                baseline_agent_enum, self.config.num_detectives
            )
        else:
            # Trained agent is detectives, baseline agent is Mr. X
            mr_x_agent = self.registry.create_mr_x_agent(baseline_agent_enum)
            detective_agent = trained_agent
        
        # Run evaluation games
        results = []
        start_time = time.time()
        
        for i in range(self.config.num_games):
            if i % 20 == 0:
                print(f"  Game {i+1}/{self.config.num_games}")
            
            result, _ = env.run_episode(mr_x_agent, detective_agent, collect_experience=False)
            results.append(result)
        
        end_time = time.time()
        
        # Calculate basic metrics
        if agent_plays_as == "mr_x":
            agent_wins = sum(1 for r in results if r.winner == "mr_x")
            baseline_wins = sum(1 for r in results if r.winner == "detectives")
        else:
            agent_wins = sum(1 for r in results if r.winner == "detectives")
            baseline_wins = sum(1 for r in results if r.winner == "mr_x")
        
        timeouts = sum(1 for r in results if r.winner == "timeout")
        
        return {
            'trained_agent_role': agent_plays_as,
            'baseline_agent_type': baseline_agent_type,
            'total_games': self.config.num_games,
            'agent_wins': agent_wins,
            'baseline_wins': baseline_wins,
            'timeouts': timeouts,
            'agent_win_rate': agent_wins / self.config.num_games,
            'baseline_win_rate': baseline_wins / self.config.num_games,
            'timeout_rate': timeouts / self.config.num_games,
            'avg_game_length': sum(r.total_turns for r in results) / len(results),
            'evaluation_time': end_time - start_time,
            'results': results
        }
    
    def comprehensive_evaluation_with_analysis(self,
                                              trained_agent,
                                              agent_plays_as: str,
                                              algorithm_name: str = "trained_agent") -> str:
        """
        Run comprehensive evaluation and generate full analysis using analyze_games.py.
        
        Args:
            trained_agent: The trained agent to evaluate
            agent_plays_as: Whether the agent plays as "mr_x" or "detectives"
            algorithm_name: Name for the algorithm/agent being evaluated
            
        Returns:
            Path to the generated analysis directory
        """
        print(f"\n🔍 Running comprehensive evaluation with full analysis")
        print("=" * 60)
        
        # Create temporary directory for saving games
        temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_{algorithm_name}_"))
        print(f"📁 Temporary analysis directory: {temp_dir}")
        
        try:
            # Run evaluation against all baseline agents and save games
            self._run_and_save_evaluation_games(trained_agent, agent_plays_as, temp_dir)
            
            # Use existing analyze_games.py to generate comprehensive analysis
            analysis_dir = self._generate_comprehensive_analysis(temp_dir, algorithm_name)
            
            return str(analysis_dir)
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            # Clean up temporary directory on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def _run_and_save_evaluation_games(self, trained_agent, agent_plays_as: str, save_dir: Path):
        """Run evaluation games and save them for analysis."""
        from simple_play.game_utils import save_game_automatically
        from simple_play.display_utils import GameDisplay, VerbosityLevel
        
        for baseline_type in self.config.baseline_agents:
            combo_name = f"trained_{agent_plays_as}_vs_{baseline_type}"
            combo_dir = save_dir / combo_name
            combo_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Running {self.config.num_games} games: {combo_name}")
            
            # Set up environment
            env = TrainingEnvironment(
                map_size=self.config.map_size,
                num_detectives=self.config.num_detectives,
                max_turns=self.config.max_turns_per_game,
                verbosity=VerbosityLevel.SILENT
            )
            
            # Create baseline agent
            baseline_agent_enum = AgentType(baseline_type.lower())
            
            if agent_plays_as == "mr_x":
                mr_x_agent = trained_agent
                detective_agent = self.registry.create_multi_detective_agent(
                    baseline_agent_enum, self.config.num_detectives
                )
            else:
                mr_x_agent = self.registry.create_mr_x_agent(baseline_agent_enum)
                detective_agent = trained_agent
            
            # Run and save games
            for i in range(self.config.num_games):
                if i % 20 == 0:
                    print(f"  Game {i+1}/{self.config.num_games}")
                
                # Run episode and get the game instance
                result, _ = env.run_episode(mr_x_agent, detective_agent, collect_experience=False)
                
                # Note: In a full implementation, we'd need to modify TrainingEnvironment
                # to return the actual game instance for saving. For now, this is a placeholder.
                # The games would need to be saved in the format expected by analyze_games.py
    
    def _generate_comprehensive_analysis(self, data_dir: Path, algorithm_name: str) -> Path:
        """Generate comprehensive analysis using the existing analyze_games.py infrastructure."""
        try:
            # Import the existing GameAnalyzer
            import sys
            import os
            sys.path.append(os.getcwd())  # Ensure we can import from root
            from ScotlandYard.services.analyze_games import GameAnalyzer
            
            # Create analyzer and generate full analysis
            analyzer = GameAnalyzer(str(data_dir))
            
            if analyzer.load_all_games():
                print(f"📊 Generating comprehensive analysis for {algorithm_name}...")
                analyzer.generate_comprehensive_analysis()
                
                # Move results to a permanent location
                analysis_dir = Path(f"training_results/evaluation_{algorithm_name}")
                analysis_dir.mkdir(parents=True, exist_ok=True)
                
                # detective analysis results
                if analyzer.graphs_dir.exists():
                    shutil.detectivetree(analyzer.graphs_dir, analysis_dir / "graphs", dirs_exist_ok=True)
                
                # detective report if it exists
                report_file = data_dir / "analysis_report.txt"
                if report_file.exists():
                    shutil.detective2(report_file, analysis_dir / "analysis_report.txt")
                
                print(f"✅ Analysis complete! Results saved to: {analysis_dir}")
                return analysis_dir
            else:
                raise RuntimeError("No games found for analysis")
                
        except ImportError:
            print("⚠️  Could not import analyze_games.py - falling back to basic analysis")
            return data_dir


def quick_evaluation_summary(results: List[GameResult], algorithm_name: str) -> None:
    """
    Print a quick evaluation summary (simplified version of analyze_games functionality).
    
    Args:
        results: List of game results
        algorithm_name: Name of the algorithm being evaluated
    """
    if not results:
        print("No results to display")
        return
    
    total_games = len(results)
    mr_x_wins = sum(1 for r in results if r.winner == "mr_x")
    detective_wins = sum(1 for r in results if r.winner == "detectives")
    timeouts = sum(1 for r in results if r.winner == "timeout")
    
    avg_length = sum(r.total_turns for r in results) / len(results)
    avg_game_time = sum(r.game_length for r in results) / len(results)
    
    print(f"\n📊 {algorithm_name} Evaluation Summary:")
    print("=" * 50)
    print(f"Total games: {total_games}")
    print(f"Mr. X wins: {mr_x_wins} ({mr_x_wins/total_games:.1%})")
    print(f"Detective wins: {detective_wins} ({detective_wins/total_games:.1%})")
    print(f"Timeouts: {timeouts} ({timeouts/total_games:.1%})")
    print(f"Average game length: {avg_length:.1f} turns")
    print(f"Average game time: {avg_game_time:.3f} seconds")
    print("=" * 50)
