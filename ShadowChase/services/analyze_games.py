"""
Game Statistics Analyzer for Shadow Chase
Specialized class for analyzing game results and generating comprehensive visualizations.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Set style for better-looking plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# Agent name mapping for better display names
AGENT_NAME_MAPPING = {
    "random": "Random Agent",
    "heuristic": "Heuristic Agent", 
    "optimized_mcts": "Optimized MCTS",
    "epsilon_greedy_mcts": "eps-Greedy MCTS",
    "deep_q": "Deep Q-Network"
}
increase = 7
# Font size configuration variables for easy debugging and testing
FONT_SIZE_LARGE_TITLE = 24   +increase       # Main titles and suptitles
FONT_SIZE_TITLE = 20         +increase           # Plot titles
FONT_SIZE_SUBTITLE = 18      +increase          # Subtitles
FONT_SIZE_LABEL = 16         +increase         # Axis labels
FONT_SIZE_TICK = 14          +increase          # Tick labels
FONT_SIZE_LEGEND = 14        +increase      # Legend text
FONT_SIZE_ANNOTATION = 13    +increase      # Value annotations on bars/plots
FONT_SIZE_TEXT_STATS = 14    +increase     # Statistics text boxes
FONT_SIZE_HEATMAP_ANNOT = 18 +increase   # Heatmap annotations
FONT_SIZE_PIE_LABEL = 14     +increase    # Pie chart labels
FONT_SIZE_PIE_PERCENT = 12   +increase    # Pie chart percentages

def get_display_name(agent_name: str) -> str:
    """Convert internal agent name to display name"""
    return AGENT_NAME_MAPPING.get(agent_name, agent_name.replace("_", " ").title())


def add_line_breaks_to_name(name: str) -> str:
    """Add line breaks at spaces in long agent names to prevent overlap"""
    if ' ' in name:
        return name.replace(' ', '\n')
    return name


def calculate_proportion_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a proportion using Wilson score interval.
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) as percentages
    """
    if total == 0:
        return 0.0, 0.0
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval (better for small samples and edge cases)
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    
    lower = max(0, centre - margin) * 100
    upper = min(100, centre + margin) * 100
    
    return lower, upper


def calculate_mean_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a mean.
    
    Args:
        values: List of values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    
    if n == 1:
        return mean, mean, mean
    
    # Use t-distribution for small samples
    if n < 30:
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_critical * stats.sem(values)
    else:
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * stats.sem(values)
    
    return mean, mean - margin, mean + margin


class GameStatistics:
    """Container for game statistics"""
    
    def __init__(self):
        self.games = []
        self.agent_combinations = {}
        self.win_rates = {}
        self.game_lengths = {}
        self.execution_times = {}
        self.temporal_data = {}
        self.performance_metrics = {}
        
    def add_game(self, game_data: Dict):
        """Add a game to the statistics"""
        self.games.append(game_data)
    
    def get_summary(self) -> Dict:
        """Get summary statistics with confidence intervals"""
        if not self.games:
            return {}
        
        total_games = len(self.games)
        detective_wins = sum(1 for g in self.games if g.get('winner') == 'detectives')
        MrX_wins = sum(1 for g in self.games if g.get('winner') != 'detectives' )
        incomplete_games = total_games - detective_wins - MrX_wins
        
        # Calculate confidence intervals for win rates
        detective_ci_lower, detective_ci_upper = calculate_proportion_confidence_interval(
            detective_wins, total_games)
        MrX_ci_lower, MrX_ci_upper = calculate_proportion_confidence_interval(
            MrX_wins, total_games)
        
        # Game length statistics with confidence intervals
        game_lengths = [g.get('total_turns', 0) for g in self.games if g.get('game_completed', False)]
        if game_lengths:
            avg_length, length_ci_lower, length_ci_upper = calculate_mean_confidence_interval(game_lengths)
        else:
            avg_length, length_ci_lower, length_ci_upper = 0, 0, 0
        
        # Execution time statistics with confidence intervals
        execution_times = [g.get('execution_time_seconds', 0) for g in self.games if g.get('execution_time_seconds') is not None]
        if execution_times:
            avg_execution_time, exec_ci_lower, exec_ci_upper = calculate_mean_confidence_interval(execution_times)
        else:
            avg_execution_time, exec_ci_lower, exec_ci_upper = 0, 0, 0
        
        # Calculate average time per turn with confidence interval
        time_per_turn_values = []
        for game in self.games:
            exec_time = game.get('execution_time_seconds')
            turns = game.get('total_turns', 0)
            if exec_time is not None and turns > 0:
                time_per_turn_values.append(exec_time / turns)
        
        if time_per_turn_values:
            avg_time_per_turn, time_per_turn_ci_lower, time_per_turn_ci_upper = calculate_mean_confidence_interval(time_per_turn_values)
        else:
            avg_time_per_turn, time_per_turn_ci_lower, time_per_turn_ci_upper = 0, 0, 0
        
        # Completion rate confidence interval
        completed_games = total_games - incomplete_games
        completion_ci_lower, completion_ci_upper = calculate_proportion_confidence_interval(
            completed_games, total_games)
        
        return {
            'total_games': total_games,
            'detective_wins': detective_wins,
            'MrX_wins': MrX_wins,
            'incomplete_games': incomplete_games,
            'detective_win_rate': detective_wins / total_games * 100 if total_games > 0 else 0,
            'detective_win_rate_ci': (detective_ci_lower, detective_ci_upper),
            'MrX_win_rate': MrX_wins / total_games * 100 if total_games > 0 else 0,
            'MrX_win_rate_ci': (MrX_ci_lower, MrX_ci_upper),
            'average_game_length': avg_length,
            'average_game_length_ci': (length_ci_lower, length_ci_upper),
            'average_execution_time': avg_execution_time,
            'average_execution_time_ci': (exec_ci_lower, exec_ci_upper),
            'average_time_per_turn': avg_time_per_turn,
            'average_time_per_turn_ci': (time_per_turn_ci_lower, time_per_turn_ci_upper),
            'games_with_timing': len(execution_times),
            'completion_rate': completed_games / total_games * 100 if total_games > 0 else 0,
            'completion_rate_ci': (completion_ci_lower, completion_ci_upper)
        }


class GameAnalyzer:
    """Comprehensive game analyzer with visualization capabilities"""
    
    def __init__(self, base_directory: str):
        self.base_dir = Path(base_directory)
        print(f"📂 Initializing GameAnalyzer with base directory: {self.base_dir}")
        self.statistics = GameStatistics()
        self.graphs_dir = self.base_dir / "analysis_graphs"
        self.graphs_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib for better output using font size variables
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = FONT_SIZE_TICK
        plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
        plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
        plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
        plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
        plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
        plt.rcParams['figure.titlesize'] = FONT_SIZE_LARGE_TITLE
    
    def load_all_games(self) -> bool:
        """Load all games from the directory structure"""
        print(f"🔍 Scanning for games in: {self.base_dir}")
        
        games_loaded = 0
        
        # Look for game combinations in subdirectories
        for combo_dir in self.base_dir.iterdir():
            if combo_dir.is_dir() and not combo_dir.name.startswith('.') and combo_dir.name != 'analysis_graphs':
                print(f"   📁 Processing: {combo_dir.name}")
                games_in_combo = self._load_games_from_directory(combo_dir)
                games_loaded += games_in_combo
                print(f"      Loaded {games_in_combo} games")
        
        print(f"✅ Total games loaded: {games_loaded}")
        return games_loaded > 0
    
    def _load_games_from_directory(self, directory: Path) -> int:
        """Load games from a specific directory"""
        games_loaded = 0
        
        # Look for metadata files
        metadata_dir = directory / "metadata"
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        game_data = json.load(f)
                    
                    # Extract agent combination from directory name
                    if "_vs_" in directory.name:
                        MrX_agent, detective_agent = directory.name.split("_vs_")
                        game_data['MrX_agent'] = MrX_agent
                        game_data['detective_agent'] = detective_agent
                        game_data['agent_combination'] = directory.name
                    
                    self.statistics.add_game(game_data)
                    games_loaded += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Error loading {metadata_file.name}: {e}")
        
        return games_loaded
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis and all visualizations"""
        if not self.statistics.games:
            print("❌ No games found to analyze")
            return
        
        print(f"📊 Analyzing {len(self.statistics.games)} games...")
        
        # Generate all analyses
        self._analyze_agent_performance()
        self._analyze_game_lengths()
        self._analyze_execution_times()
        self._analyze_win_rates()
        self._analyze_temporal_patterns()
        
        # Generate all visualizations
        self._create_win_rate_comparison()
        self._create_game_length_analysis()
        self._create_execution_time_analysis()
        self._create_agent_performance_matrix()
        self._create_temporal_analysis()
        self._create_comprehensive_dashboard()
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"✅ Analysis complete! Graphs saved to: {self.graphs_dir}")
    
    def _analyze_agent_performance(self):
        """Analyze performance by agent combination with confidence intervals"""
        combinations = defaultdict(lambda: {'games': 0, 'detective_wins': 0, 'MrX_wins': 0, 
                                          'total_turns': 0, 'completed_games': 0, 'game_lengths': []})
        
        for game in self.statistics.games:
            combo = game.get('agent_combination', 'unknown')
            combinations[combo]['games'] += 1
            
            if game.get('game_completed', False):
                combinations[combo]['completed_games'] += 1
                turn_count = game.get('total_turns', 0)
                combinations[combo]['total_turns'] += turn_count
                combinations[combo]['game_lengths'].append(turn_count)
                
                if game.get('winner') == 'detectives':
                    combinations[combo]['detective_wins'] += 1
                elif game.get('winner') == 'MrX':
                    combinations[combo]['MrX_wins'] += 1
        
        # Calculate performance metrics with confidence intervals
        for combo, stats in combinations.items():
            if stats['games'] > 0:
                # Win rate confidence intervals
                detective_ci_lower, detective_ci_upper = calculate_proportion_confidence_interval(
                    stats['detective_wins'], stats['games'])
                MrX_ci_lower, MrX_ci_upper = calculate_proportion_confidence_interval(
                    stats['MrX_wins'], stats['games'])
                completion_ci_lower, completion_ci_upper = calculate_proportion_confidence_interval(
                    stats['completed_games'], stats['games'])
                
                stats['detective_win_rate'] = stats['detective_wins'] / stats['games'] * 100
                stats['detective_win_rate_ci'] = (detective_ci_lower, detective_ci_upper)
                stats['MrX_win_rate'] = stats['MrX_wins'] / stats['games'] * 100
                stats['MrX_win_rate_ci'] = (MrX_ci_lower, MrX_ci_upper)
                stats['completion_rate'] = stats['completed_games'] / stats['games'] * 100
                stats['completion_rate_ci'] = (completion_ci_lower, completion_ci_upper)
                
                # Game length confidence interval
                if stats['game_lengths']:
                    avg_length, length_ci_lower, length_ci_upper = calculate_mean_confidence_interval(
                        stats['game_lengths'])
                    stats['avg_game_length'] = avg_length
                    stats['avg_game_length_ci'] = (length_ci_lower, length_ci_upper)
                else:
                    stats['avg_game_length'] = 0
                    stats['avg_game_length_ci'] = (0, 0)
        
        self.statistics.agent_combinations = dict(combinations)
    
    def _analyze_game_lengths(self):
        """Analyze game length patterns"""
        length_data = defaultdict(list)
        
        for game in self.statistics.games:
            if game.get('game_completed', False):
                combo = game.get('agent_combination', 'unknown')
                length_data[combo].append(game.get('total_turns', 0))
        
        self.statistics.game_lengths = dict(length_data)
    
    def _analyze_execution_times(self):
        """Analyze execution time patterns"""
        execution_data = defaultdict(list)
        
        for game in self.statistics.games:
            if game.get('execution_time_seconds') is not None:
                combo = game.get('agent_combination', 'unknown')
                execution_data[combo].append(game.get('execution_time_seconds'))
        
        self.statistics.execution_times = dict(execution_data)
    
    def _analyze_win_rates(self):
        """Analyze win rates by agent type with confidence intervals"""
        MrX_performance = defaultdict(lambda: {'games': 0, 'wins': 0})
        detective_performance = defaultdict(lambda: {'games': 0, 'wins': 0})
        
        for game in self.statistics.games:
            MrX_agent = game.get('MrX_agent', 'unknown')
            detective_agent = game.get('detective_agent', 'unknown')
            
            MrX_performance[MrX_agent]['games'] += 1
            detective_performance[detective_agent]['games'] += 1
            
            if game.get('winner') == 'MrX':
                MrX_performance[MrX_agent]['wins'] += 1
            elif game.get('winner') == 'detectives':
                detective_performance[detective_agent]['wins'] += 1
        
        # Calculate win rates with confidence intervals
        for agent, stats in MrX_performance.items():
            if stats['games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['games'] * 100
                ci_lower, ci_upper = calculate_proportion_confidence_interval(
                    stats['wins'], stats['games'])
                stats['win_rate_ci'] = (ci_lower, ci_upper)
        
        for agent, stats in detective_performance.items():
            if stats['games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['games'] * 100
                ci_lower, ci_upper = calculate_proportion_confidence_interval(
                    stats['wins'], stats['games'])
                stats['win_rate_ci'] = (ci_lower, ci_upper)
        
        self.statistics.win_rates = {
            'MrX': dict(MrX_performance),
            'detective': dict(detective_performance)
        }
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in games"""
        # Group games by creation time (if available)
        temporal_data = defaultdict(int)
        
        for game in self.statistics.games:
            created_at = game.get('created_at', '')
            if created_at:
                try:
                    date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).date()
                    temporal_data[date.isoformat()] += 1
                except:
                    pass
        
        self.statistics.temporal_data = dict(temporal_data)
    
    def _create_win_rate_comparison(self):
        """Create win rate comparison chart with confidence intervals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Mr. X win rates
        if self.statistics.win_rates.get('MrX'):
            MrX_data = self.statistics.win_rates['MrX']
            agents = [get_display_name(agent) for agent in MrX_data.keys()]
            win_rates = [data['win_rate'] for data in MrX_data.values()]
            games = [data['games'] for data in MrX_data.values()]
            
            # Get confidence intervals
            ci_data = [data.get('win_rate_ci', (0, 0)) for data in MrX_data.values()]
            
            # Create boxplot-style visualization with error bars
            x_pos = range(len(agents))
            bars1 = ax1.bar(x_pos, win_rates, color=sns.color_palette("Set2", len(agents)),
                           alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add confidence interval error bars
            lower_errors = [max(0, rate - ci[0]) for rate, ci in zip(win_rates, ci_data)]
            upper_errors = [min(100 - rate, ci[1] - rate) for rate, ci in zip(win_rates, ci_data)]
            ax1.errorbar(x_pos, win_rates, yerr=[lower_errors, upper_errors], 
                        fmt='none', capsize=8, capthick=2, ecolor='black', elinewidth=2)
            
            ax1.set_title('Mr. X Win Rates by Agent Type (95% CI)', fontsize=FONT_SIZE_TITLE, pad=30, fontweight='bold')
            ax1.set_ylabel('Win Rate (%)', fontsize=FONT_SIZE_LABEL)
            ax1.set_ylim(0, 100)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(agents, fontsize=FONT_SIZE_TICK)
            ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
            
            # Add value labels with confidence intervals
            for i, (rate, game_count, ci) in enumerate(zip(win_rates, games, ci_data)):
                ax1.text(i, rate + upper_errors[i] + 3,
                        f'{rate:.1f}%\n[{ci[0]:.1f}%, {ci[1]:.1f}%]\n({game_count} games)',
                        ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        # Detective win rates
        if self.statistics.win_rates.get('detective'):
            detective_data = self.statistics.win_rates['detective']
            agents = [get_display_name(agent) for agent in detective_data.keys()]
            win_rates = [data['win_rate'] for data in detective_data.values()]
            games = [data['games'] for data in detective_data.values()]
            
            # Get confidence intervals
            ci_data = [data.get('win_rate_ci', (0, 0)) for data in detective_data.values()]
            
            # Create boxplot-style visualization with error bars
            x_pos = range(len(agents))
            bars2 = ax2.bar(x_pos, win_rates, color=sns.color_palette("Set1", len(agents)),
                           alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add confidence interval error bars
            lower_errors = [max(0, rate - ci[0]) for rate, ci in zip(win_rates, ci_data)]
            upper_errors = [min(100 - rate, ci[1] - rate) for rate, ci in zip(win_rates, ci_data)]
            ax2.errorbar(x_pos, win_rates, yerr=[lower_errors, upper_errors], 
                        fmt='none', capsize=8, capthick=2, ecolor='black', elinewidth=2)
            
            ax2.set_title('Detective Win Rates by Agent Type (95% CI)', fontsize=FONT_SIZE_TITLE, pad=30, fontweight='bold')
            ax2.set_ylabel('Win Rate (%)', fontsize=FONT_SIZE_LABEL)
            ax2.set_ylim(0, 100)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(agents, fontsize=FONT_SIZE_TICK)
            ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
            
            # Add value labels with confidence intervals
            for i, (rate, game_count, ci) in enumerate(zip(win_rates, games, ci_data)):
                ax2.text(i, rate + upper_errors[i] + 3,
                        f'{rate:.1f}%\n[{ci[0]:.1f}%, {ci[1]:.1f}%]\n({game_count} games)',
                        ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "win_rates_by_agent.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📈 Generated: win_rates_by_agent.jpg")
    
    def _create_game_length_analysis(self):
        """Create game length analysis"""
        if not self.statistics.game_lengths:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Box plot of game lengths by combination
        combinations = list(self.statistics.game_lengths.keys())
        display_combinations = [f"{get_display_name(combo.split('_vs_')[0])}\nvs\n{get_display_name(combo.split('_vs_')[1])}" 
                              if "_vs_" in combo else combo for combo in combinations]
        lengths_data = [self.statistics.game_lengths[combo] for combo in combinations]
        
        if lengths_data:
            box_plot = ax1.boxplot(lengths_data, labels=display_combinations, patch_artist=True)
            colors = sns.color_palette("Set3", len(combinations))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_title('Game Length Distribution by Agent Combination', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
            ax1.set_ylabel('Game Length (turns)', fontsize=FONT_SIZE_LABEL)
            ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK-1)
            ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Histogram of all game lengths
        all_lengths = [length for lengths in lengths_data for length in lengths]
        if all_lengths:
            ax2.hist(all_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Overall Game Length Distribution', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
            ax2.set_xlabel('Game Length (turns)', fontsize=FONT_SIZE_LABEL)
            ax2.set_ylabel('Number of Games', fontsize=FONT_SIZE_LABEL)
            ax2.axvline(np.mean(all_lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(all_lengths):.1f}')
            ax2.legend(fontsize=FONT_SIZE_LEGEND)
            ax2.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
            ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Average game length by combination
        avg_lengths = [np.mean(lengths) if lengths else 0 for lengths in lengths_data]
        if avg_lengths:
            bars = ax3.bar(range(len(combinations)), avg_lengths, 
                          color=sns.color_palette("viridis", len(combinations)))
            ax3.set_title('Average Game Length by Agent Combination', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
            ax3.set_xlabel('Agent Combination', fontsize=FONT_SIZE_LABEL)
            ax3.set_ylabel('Average Game Length (turns)', fontsize=FONT_SIZE_LABEL)
            ax3.set_xticks(range(len(combinations)))
            ax3.set_xticklabels(display_combinations, fontsize=FONT_SIZE_TICK-1)
            ax3.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
            
            # Add value labels
            for bar, avg_len in zip(bars, avg_lengths):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{avg_len:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        # Game completion rates
        completion_rates = [
            self.statistics.agent_combinations[combo]['completion_rate'] 
            for combo in combinations 
            if combo in self.statistics.agent_combinations
        ]
        
        if completion_rates:
            bars = ax4.bar(range(len(combinations)), completion_rates,
                          color=sns.color_palette("plasma", len(combinations)))
            ax4.set_title('Game Completion Rate by Agent Combination', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
            ax4.set_xlabel('Agent Combination', fontsize=FONT_SIZE_LABEL)
            ax4.set_ylabel('Completion Rate (%)', fontsize=FONT_SIZE_LABEL)
            ax4.set_xticks(range(len(combinations)))
            ax4.set_xticklabels(display_combinations, fontsize=FONT_SIZE_TICK-1)
            ax4.set_ylim(0, 100)
            ax4.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
            
            # Add value labels
            for bar, rate in zip(bars, completion_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "game_length_analysis.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Generated: game_length_analysis.jpg")
    
    def _create_execution_time_analysis(self):
        """Create execution time analysis visualization"""
        if not self.statistics.execution_times:
            print("   ⚠️  No execution time data available - skipping execution time analysis")
            return
        
        combinations = list(self.statistics.execution_times.keys())
        if not combinations:
            return
        
        display_combinations = [f"{get_display_name(combo.split('_vs_')[0])}\nvs\n{get_display_name(combo.split('_vs_')[1])}" 
                              if "_vs_" in combo else combo for combo in combinations]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Execution Time Analysis by Agent Combination', fontsize=FONT_SIZE_LARGE_TITLE, fontweight='bold', y=0.98)
        
        # Box plot of execution times
        execution_data = [self.statistics.execution_times[combo] for combo in combinations]
        bp = ax1.boxplot(execution_data, labels=display_combinations, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("viridis", len(combinations))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title('Execution Time Distribution', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
        ax1.set_xlabel('Agent Combination', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=FONT_SIZE_LABEL)
        ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK-1)
        ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Average execution times bar chart
        avg_times = [np.mean(self.statistics.execution_times[combo]) for combo in combinations]
        bars = ax2.bar(range(len(combinations)), avg_times,
                      color=sns.color_palette("plasma", len(combinations)))
        ax2.set_title('Average Execution Time by Agent Combination', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
        ax2.set_xlabel('Agent Combination', fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel('Average Time (seconds)', fontsize=FONT_SIZE_LABEL)
        ax2.set_xticks(range(len(combinations)))
        ax2.set_xticklabels(display_combinations, fontsize=FONT_SIZE_TICK-1)
        ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Add value labels
        for bar, avg_time in zip(bars, avg_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{avg_time:.2f}s', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        # Execution time vs game length scatter plot
        all_exec_times = []
        all_game_lengths = []
        colors_scatter = []
        
        for i, combo in enumerate(combinations):
            if combo in self.statistics.game_lengths:
                # Match execution times with game lengths for this combination
                exec_times = self.statistics.execution_times[combo]
                game_lengths = self.statistics.game_lengths[combo]
                
                # Take minimum length to avoid index errors
                min_len = min(len(exec_times), len(game_lengths))
                all_exec_times.extend(exec_times[:min_len])
                all_game_lengths.extend(game_lengths[:min_len])
                colors_scatter.extend([colors[i]] * min_len)
        
        if all_exec_times and all_game_lengths:
            scatter = ax3.scatter(all_game_lengths, all_exec_times, c=colors_scatter, alpha=0.6, s=50)
            ax3.set_title('Execution Time vs Game Length', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
            ax3.set_xlabel('Game Length (turns)', fontsize=FONT_SIZE_LABEL)
            ax3.set_ylabel('Execution Time (seconds)', fontsize=FONT_SIZE_LABEL)
            ax3.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
            ax3.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
            
            # Add trend line
            if len(all_game_lengths) > 1:
                z = np.polyfit(all_game_lengths, all_exec_times, 1)
                p = np.poly1d(z)
                ax3.plot(sorted(all_game_lengths), p(sorted(all_game_lengths)), 
                        "r--", alpha=0.8, linewidth=2)
        
        # Games per minute calculation
        games_per_min = []
        for combo in combinations:
            if self.statistics.execution_times[combo]:
                avg_time = np.mean(self.statistics.execution_times[combo])
                gpm = 60 / avg_time if avg_time > 0 else 0
                games_per_min.append(gpm)
            else:
                games_per_min.append(0)
        
        bars = ax4.bar(range(len(combinations)), games_per_min,
                      color=sns.color_palette("cool", len(combinations)))
        ax4.set_title('Theoretical Games per Minute', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
        ax4.set_xlabel('Agent Combination', fontsize=FONT_SIZE_LABEL)
        ax4.set_ylabel('Games per Minute', fontsize=FONT_SIZE_LABEL)
        ax4.set_xticks(range(len(combinations)))
        ax4.set_xticklabels(display_combinations, fontsize=FONT_SIZE_TICK-1)
        ax4.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Add value labels
        for bar, gpm in zip(bars, games_per_min):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{gpm:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "execution_time_analysis.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Generated: execution_time_analysis.jpg")
    
    def _create_agent_performance_matrix(self):
        """Create agent performance matrix heatmap"""
        if not self.statistics.agent_combinations:
            return
        
        # Extract agent types
        MrX_agents = set()
        detective_agents = set()
        
        for combo in self.statistics.agent_combinations.keys():
            if "_vs_" in combo:
                MrX, detective = combo.split("_vs_")
                MrX_agents.add(MrX)
                detective_agents.add(detective)
        
        MrX_agents = sorted(list(MrX_agents))
        detective_agents = sorted(list(detective_agents),reverse=True)
        
        if not MrX_agents or not detective_agents:
            return
        
        # Convert to display names
        MrX_display = [get_display_name(agent) for agent in MrX_agents]
        detective_display = [get_display_name(agent) for agent in detective_agents]
        
        # Create performance matrices
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))
        
        # Mr. X win rate matrix
        MrX_matrix = np.zeros((len(MrX_agents), len(detective_agents)))
        detective_matrix = np.zeros((len(MrX_agents), len(detective_agents)))
        game_count_matrix = np.zeros((len(MrX_agents), len(detective_agents)))
        avg_length_matrix = np.zeros((len(MrX_agents), len(detective_agents)))
        
        for i, MrX in enumerate(MrX_agents):
            for j, detective in enumerate(detective_agents):
                combo = f"{MrX}_vs_{detective}"
                if combo in self.statistics.agent_combinations:
                    stats = self.statistics.agent_combinations[combo]
                    MrX_matrix[i, j] = stats['MrX_win_rate']
                    detective_matrix[i, j] = stats['detective_win_rate']
                    game_count_matrix[i, j] = stats['games']
                    avg_length_matrix[i, j] = stats['avg_game_length']
        
        # Mr. X win rate heatmap
        sns.heatmap(MrX_matrix, annot=True, fmt='.1f', 
                   xticklabels=detective_display, yticklabels=MrX_display,
                   cmap='Reds', ax=ax1, cbar_kws={'label': 'Win Rate (%)'}, 
                   annot_kws={'fontsize': FONT_SIZE_HEATMAP_ANNOT, 'fontweight': 'bold'})
        ax1.set_title('Mr. X Win Rate Matrix', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax1.set_xlabel('Detective Agent', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel('Mr. X Agent', fontsize=FONT_SIZE_LABEL)
        ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
        ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Detective win rate heatmap
        sns.heatmap(detective_matrix, annot=True, fmt='.1f',
                   xticklabels=detective_display, yticklabels=MrX_display,
                   cmap='Blues', ax=ax2, cbar_kws={'label': 'Win Rate (%)'}, 
                   annot_kws={'fontsize': FONT_SIZE_HEATMAP_ANNOT, 'fontweight': 'bold'})
        ax2.set_title('Detective Win Rate Matrix', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax2.set_xlabel('Detective Agent', fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel('Mr. X Agent', fontsize=FONT_SIZE_LABEL)
        ax2.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
        ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Game count heatmap
        sns.heatmap(game_count_matrix, annot=True, fmt='.0f',
                   xticklabels=detective_display, yticklabels=MrX_display,
                   cmap='Greens', ax=ax3, cbar_kws={'label': 'Number of Games'}, 
                   annot_kws={'fontsize': FONT_SIZE_HEATMAP_ANNOT, 'fontweight': 'bold'})
        ax3.set_title('Number of Games Matrix', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax3.set_xlabel('Detective Agent', fontsize=FONT_SIZE_LABEL)
        ax3.set_ylabel('Mr. X Agent', fontsize=FONT_SIZE_LABEL)
        ax3.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
        ax3.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # Average game length heatmap
        sns.heatmap(avg_length_matrix, annot=True, fmt='.1f',
                   xticklabels=detective_display, yticklabels=MrX_display,
                   cmap='Purples', ax=ax4, cbar_kws={'label': 'Average Turns'}, 
                   annot_kws={'fontsize': FONT_SIZE_HEATMAP_ANNOT, 'fontweight': 'bold'})
        ax4.set_title('Average Game Length Matrix', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
        ax4.set_xlabel('Detective Agent', fontsize=FONT_SIZE_LABEL)
        ax4.set_ylabel('Mr. X Agent', fontsize=FONT_SIZE_LABEL)
        ax4.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
        ax4.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "agent_performance_matrix.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   🔥 Generated: agent_performance_matrix.jpg")
    
    def _create_temporal_analysis(self):
        """Create temporal analysis if data is available"""
        if not self.statistics.temporal_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Games over time
        dates = sorted(self.statistics.temporal_data.keys())
        counts = [self.statistics.temporal_data[date] for date in dates]
        
        ax1.plot(dates, counts, marker='o', linewidth=3, markersize=8, color='steelblue')
        ax1.set_title('Games Played Over Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel('Number of Games', fontsize=FONT_SIZE_LABEL)
        ax1.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK)
        ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative games
        cumulative = np.cumsum(counts)
        ax2.plot(dates, cumulative, marker='s', linewidth=3, markersize=8, color='darkgreen')
        ax2.set_title('Cumulative Games Played', fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel('Cumulative Games', fontsize=FONT_SIZE_LABEL)
        ax2.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK)
        ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "temporal_analysis.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📅 Generated: temporal_analysis.jpg")
    
    def _create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard summary"""
        summary = self.statistics.get_summary()
        if not summary:
            return
        
        fig = plt.figure(figsize=(32, 24))
        
        # Create a simple 2x2 grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # TOP LEFT: Theoretical Games per Minute for same-type agents
        ax1 = fig.add_subplot(gs[0, 0])
        if self.statistics.execution_times:
            # Filter for same-type agent combinations
            same_type_exec_combinations = []
            same_type_games_per_min = []
            
            for combo in self.statistics.execution_times.keys():
                if "_vs_" in combo:
                    MrX_agent, detective_agent = combo.split("_vs_")
                    if MrX_agent == detective_agent and self.statistics.execution_times[combo]:  # Same type of agent
                        avg_time = np.mean(self.statistics.execution_times[combo])
                        gpm = 60 / avg_time if avg_time > 0 else 0
                        same_type_exec_combinations.append(combo)
                        same_type_games_per_min.append(gpm)
            
            if same_type_exec_combinations and same_type_games_per_min:
                display_combinations = [add_line_breaks_to_name(get_display_name(combo.split('_vs_')[0])) for combo in same_type_exec_combinations]
                
                bars = ax1.bar(range(len(same_type_exec_combinations)), same_type_games_per_min,
                              color=sns.color_palette("plasma", len(same_type_exec_combinations)))
                ax1.set_title('Theoretical Games per Minute\n(Same-Type Agent Combinations)', 
                             fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
                ax1.set_xlabel('Agent Type (Both Players)', fontsize=FONT_SIZE_LABEL)
                ax1.set_ylabel('Games per Minute', fontsize=FONT_SIZE_LABEL)
                ax1.set_xticks(range(len(same_type_exec_combinations)))
                ax1.set_xticklabels(display_combinations, fontsize=FONT_SIZE_TICK)
                ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
                
                # Add value labels
                for bar, gpm in zip(bars, same_type_games_per_min):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{gpm:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')
        
        # BOTTOM LEFT: Game Length Distribution for same-type agents
        ax2 = fig.add_subplot(gs[1, 0])
        if self.statistics.game_lengths:
            # Filter for same-type agent combinations
            same_type_combinations = []
            same_type_lengths_data = []
            
            for combo, lengths in self.statistics.game_lengths.items():
                if "_vs_" in combo:
                    MrX_agent, detective_agent = combo.split("_vs_")
                    if MrX_agent == detective_agent:  # Same type of agent
                        same_type_combinations.append(combo)
                        same_type_lengths_data.append(lengths)
            
            if same_type_combinations and same_type_lengths_data:
                display_combinations = [add_line_breaks_to_name(get_display_name(combo.split('_vs_')[0])) for combo in same_type_combinations]
                
                box_plot = ax2.boxplot(same_type_lengths_data, labels=display_combinations, patch_artist=True)
                colors = sns.color_palette("viridis", len(same_type_combinations))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax2.set_title('Game Length Distribution\n(Same-Type Agent Combinations)', 
                             fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
                ax2.set_xlabel('Agent Type (Both Players)', fontsize=FONT_SIZE_LABEL)
                ax2.set_ylabel('Game Length (turns)', fontsize=FONT_SIZE_LABEL)
                ax2.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
                ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # TOP RIGHT: Performance Heatmap
        ax3 = fig.add_subplot(gs[0, 1])
        if self.statistics.agent_combinations:
            # Extract agent types for heatmap
            MrX_agents = set()
            detective_agents = set()
            
            for combo in self.statistics.agent_combinations.keys():
                if "_vs_" in combo:
                    MrX, detective = combo.split("_vs_")
                    MrX_agents.add(MrX)
                    detective_agents.add(detective)
            
            MrX_agents = sorted(list(MrX_agents))
            detective_agents = sorted(list(detective_agents), reverse=True)
            
            if MrX_agents and detective_agents:
                # Convert to display names
                MrX_display = [get_display_name(agent) for agent in MrX_agents]
                detective_display = [get_display_name(agent) for agent in detective_agents]
                
                # Create Mr. X win rate matrix
                MrX_matrix = np.zeros((len(MrX_agents), len(detective_agents)))
                
                for i, MrX in enumerate(MrX_agents):
                    for j, detective in enumerate(detective_agents):
                        combo = f"{MrX}_vs_{detective}"
                        if combo in self.statistics.agent_combinations:
                            stats = self.statistics.agent_combinations[combo]
                            MrX_matrix[i, j] = stats['MrX_win_rate']
                
                # Create heatmap
                sns.heatmap(MrX_matrix, annot=True, fmt='.1f', 
                           xticklabels=detective_display, yticklabels=MrX_display,
                           cmap='RdYlBu_r', ax=ax3, cbar_kws={'label': 'Mr. X Win Rate (%)'}, 
                           annot_kws={'fontsize': FONT_SIZE_HEATMAP_ANNOT, 'fontweight': 'bold'})
                ax3.set_title('Mr. X Win Rate Matrix', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
                ax3.set_xlabel('Detective Agent', fontsize=FONT_SIZE_LABEL)
                ax3.set_ylabel('Mr. X Agent', fontsize=FONT_SIZE_LABEL)
                ax3.tick_params(axis='x', labelsize=FONT_SIZE_TICK, rotation=30)
                ax3.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        
        # BOTTOM RIGHT: Pie Charts (split into top and bottom halves)
        # Create a sub-grid for the pie charts
        gs_pie = gs[1, 1].subgridspec(2, 1, hspace=0.4)
        
        # Game Outcomes pie chart (top half of bottom right)
        ax4 = fig.add_subplot(gs_pie[0])
        if summary['detective_wins'] > 0 or summary['MrX_wins'] > 0:
            sizes = [summary['detective_wins'], summary['MrX_wins']]
            labels = ['Detective Wins', 'Mr. X Wins']
            colors = ['royalblue', 'crimson']  # Blue for detectives, red for MrX
            
            if summary['incomplete_games'] > 0:
                sizes.append(summary['incomplete_games'])
                labels.append('Incomplete')
                colors.append('lightgray')
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Game Outcomes Distribution', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
            
            # Improve text readability
            for text in texts:
                text.set_fontsize(FONT_SIZE_PIE_LABEL)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(FONT_SIZE_PIE_PERCENT)
        
        # Agent Strength Distribution pie chart (bottom half of bottom right)
        ax5 = fig.add_subplot(gs_pie[1])
        agent_strength = self._calculate_agent_strength()
        if agent_strength:
            agents = list(agent_strength.keys())
            total_wins = [agent_strength[agent]['total_wins'] for agent in agents]
            
            # Only show agents with at least 1 win
            filtered_data = [(agent, wins) for agent, wins in zip(agents, total_wins) if wins > 0]
            if filtered_data:
                agents, total_wins = zip(*filtered_data)
                display_agents = [get_display_name(agent) for agent in agents]
                colors_strength = sns.color_palette("Set3", len(agents))
                
                wedges, texts, autotexts = ax5.pie(total_wins, labels=display_agents, colors=colors_strength, 
                                                  autopct='%1.1f%%', startangle=90)
                ax5.set_title('Agent Strength Distribution\n(Total Wins Across All Roles)', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
                
                # Improve text readability
                for text in texts:
                    text.set_fontsize(FONT_SIZE_PIE_LABEL)
                    text.set_fontweight('bold')
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(FONT_SIZE_PIE_PERCENT)
        
        plt.suptitle('Shadow Chase Game Analysis Dashboard', fontsize=FONT_SIZE_LARGE_TITLE, y=0.95, fontweight='bold')
        plt.savefig(self.graphs_dir / "comprehensive_dashboard.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Generated: comprehensive_dashboard.jpg")
    
    def _calculate_agent_strength(self) -> Dict[str, Dict[str, int]]:
        """Calculate agent strength independent of role (Mr. X vs Detective)"""
        agent_strength = defaultdict(lambda: {'total_wins': 0, 'total_games': 0, 'win_rate': 0.0})
        
        # Aggregate wins from both Mr. X and Detective roles
        if self.statistics.win_rates.get('MrX'):
            for agent, stats in self.statistics.win_rates['MrX'].items():
                agent_strength[agent]['total_wins'] += stats['wins']
                agent_strength[agent]['total_games'] += stats['games']
        
        if self.statistics.win_rates.get('detective'):
            for agent, stats in self.statistics.win_rates['detective'].items():
                agent_strength[agent]['total_wins'] += stats['wins']
                agent_strength[agent]['total_games'] += stats['games']
        
        # Calculate overall win rates
        for agent, stats in agent_strength.items():
            if stats['total_games'] > 0:
                stats['win_rate'] = (stats['total_wins'] / stats['total_games']) * 100
        
        return dict(agent_strength)
    
    def _generate_summary_report(self):
        """Generate a detailed summary report"""
        report_file = self.graphs_dir.parent / "analysis_report.txt"
        
        summary = self.statistics.get_summary()
        
        with open(report_file, 'w') as f:
            f.write("SHADOW CHASE GAME ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Directory: {self.base_dir}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Games: {summary.get('total_games', 0):,}\n")
            
            detective_win_rate = summary.get('detective_win_rate', 0)
            detective_ci = summary.get('detective_win_rate_ci', (0, 0))
            f.write(f"Detective Wins: {summary.get('detective_wins', 0):,} ({detective_win_rate:.1f}% [95% CI: {detective_ci[0]:.1f}%-{detective_ci[1]:.1f}%])\n")
            
            MrX_win_rate = summary.get('MrX_win_rate', 0)
            MrX_ci = summary.get('MrX_win_rate_ci', (0, 0))
            f.write(f"Mr. X Wins: {summary.get('MrX_wins', 0):,} ({MrX_win_rate:.1f}% [95% CI: {MrX_ci[0]:.1f}%-{MrX_ci[1]:.1f}%])\n")
            
            f.write(f"Incomplete Games: {summary.get('incomplete_games', 0):,}\n")
            
            avg_length = summary.get('average_game_length', 0)
            length_ci = summary.get('average_game_length_ci', (0, 0))
            f.write(f"Average Game Length: {avg_length:.1f} turns [95% CI: {length_ci[0]:.1f}-{length_ci[1]:.1f}]\n")
            
            avg_exec_time = summary.get('average_execution_time', 0)
            exec_ci = summary.get('average_execution_time_ci', (0, 0))
            f.write(f"Average Execution Time: {avg_exec_time:.5f} seconds [95% CI: {exec_ci[0]:.5f}-{exec_ci[1]:.5f}]\n")
            
            avg_time_per_turn = summary.get('average_time_per_turn', 0)
            time_per_turn_ci = summary.get('average_time_per_turn_ci', (0, 0))
            f.write(f"Average Time per Turn: {avg_time_per_turn:.5f} seconds [95% CI: {time_per_turn_ci[0]:.5f}-{time_per_turn_ci[1]:.5f}]\n")
            
            f.write(f"Games with Timing Data: {summary.get('games_with_timing', 0):,}\n")
            
            completion_rate = summary.get('completion_rate', 0)
            completion_ci = summary.get('completion_rate_ci', (0, 0))
            f.write(f"Completion Rate: {completion_rate:.1f}% [95% CI: {completion_ci[0]:.1f}%-{completion_ci[1]:.1f}%]\n\n")
            
            # Agent combination details
            if self.statistics.agent_combinations:
                f.write("AGENT COMBINATION PERFORMANCE\n")
                f.write("-" * 32 + "\n")
                for combo, stats in sorted(self.statistics.agent_combinations.items()):
                    # Convert to display names for the report
                    if "_vs_" in combo:
                        MrX_agent, detective_agent = combo.split("_vs_")
                        display_combo = f"{get_display_name(MrX_agent)} vs {get_display_name(detective_agent)}"
                    else:
                        display_combo = combo
                    
                    f.write(f"{display_combo}:\n")
                    f.write(f"  Games: {stats['games']:,}\n")
                    
                    detective_wr = stats['detective_win_rate']
                    detective_ci = stats.get('detective_win_rate_ci', (0, 0))
                    f.write(f"  Detective Win Rate: {detective_wr:.1f}% [95% CI: {detective_ci[0]:.1f}%-{detective_ci[1]:.1f}%]\n")
                    
                    MrX_wr = stats['MrX_win_rate']
                    MrX_ci = stats.get('MrX_win_rate_ci', (0, 0))
                    f.write(f"  Mr. X Win Rate: {MrX_wr:.1f}% [95% CI: {MrX_ci[0]:.1f}%-{MrX_ci[1]:.1f}%]\n")
                    
                    avg_length = stats['avg_game_length']
                    length_ci = stats.get('avg_game_length_ci', (0, 0))
                    f.write(f"  Avg Game Length: {avg_length:.1f} turns [95% CI: {length_ci[0]:.1f}-{length_ci[1]:.1f}]\n")
                    
                    completion_rate = stats['completion_rate']
                    completion_ci = stats.get('completion_rate_ci', (0, 0))
                    f.write(f"  Completion Rate: {completion_rate:.1f}% [95% CI: {completion_ci[0]:.1f}%-{completion_ci[1]:.1f}%]\n")
                    
                    # Add execution time info if available
                    if combo in self.statistics.execution_times:
                        exec_times = self.statistics.execution_times[combo]
                        if exec_times:
                            avg_exec_time = np.mean(exec_times)
                            f.write(f"  Avg Execution Time: {avg_exec_time:.5f} seconds\n")
                            f.write(f"  Games per Minute: {60/avg_exec_time:.1f}\n")
                            
                            # Calculate average time per turn for this combination
                            if combo in self.statistics.game_lengths:
                                game_lengths = self.statistics.game_lengths[combo]
                                if len(exec_times) == len(game_lengths):
                                    total_time = sum(exec_times)
                                    total_turns = sum(game_lengths)
                                    if total_turns > 0:
                                        avg_time_per_turn = total_time / total_turns
                                        f.write(f"  Avg Time per Turn: {avg_time_per_turn:.5f} seconds\n")
                    
                    f.write("\n")
            
            # Agent Strength Analysis (independent of role)
            agent_strength = self._calculate_agent_strength()
            if agent_strength:
                f.write("AGENT STRENGTH ANALYSIS (Independent of Role)\n")
                f.write("-" * 45 + "\n")
                # Sort agents by total wins for better presentation
                sorted_agents = sorted(agent_strength.items(), 
                                     key=lambda x: x[1]['total_wins'], reverse=True)
                for agent, stats in sorted_agents:
                    display_agent = get_display_name(agent)
                    f.write(f"{display_agent}:\n")
                    f.write(f"  Total Wins: {stats['total_wins']:,}\n")
                    f.write(f"  Total Games: {stats['total_games']:,}\n")
                    f.write(f"  Overall Win Rate: {stats['win_rate']:.1f}%\n")
                    f.write("\n")
            
            # Generated files
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 25 + "\n")
            graph_files = list(self.graphs_dir.glob("*.jpg"))
            for graph_file in sorted(graph_files):
                f.write(f"- {graph_file.name}\n")
        
        print(f"Generated: {report_file.name}")


def main():
    """Main analysis function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_games.py <test_directory>")
        print("Example: python analyze_games.py test0_random_vs_heuristic")
        return

    test_dir = "saved_games/" + sys.argv[1]

    if not os.path.exists(test_dir):
        print(f"❌ Directory not found: {test_dir}")
        return
    
    print(f"🔍 SHADOW CHASE GAME ANALYSIS")
    print("=" * 40)
    print(f"📁 Analyzing directory: {test_dir}")
    
    try:
        analyzer = GameAnalyzer(test_dir)
        
        if analyzer.load_all_games():
            analyzer.generate_comprehensive_analysis()
            print(f"\n🎉 Analysis complete!")
            print(f"📊 Graphs saved to: {analyzer.graphs_dir}")
            print(f"📝 Report saved to: {analyzer.base_dir}/analysis_report.txt")
        else:
            print("❌ No games found to analyze")
    
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
