"""
Game Statistics Analyzer for Scotland Yard
Specialized class for analyzing game results and generating comprehensive visualizations.
"""

import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter
import glob

# Set style for better-looking plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


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
        """Get summary statistics"""
        if not self.games:
            return {}
        
        total_games = len(self.games)
        detective_wins = sum(1 for g in self.games if g.get('winner') == 'detectives')
        mr_x_wins = sum(1 for g in self.games if g.get('winner') == 'mr_x')
        incomplete_games = total_games - detective_wins - mr_x_wins
        
        avg_length = np.mean([g.get('total_turns', 0) for g in self.games if g.get('game_completed', False)])
        
        # Calculate average execution time for games that have this data
        execution_times = [g.get('execution_time_seconds', 0) for g in self.games if g.get('execution_time_seconds') is not None]
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        
        # Calculate average time per turn
        total_time = 0
        total_turns = 0
        for game in self.games:
            if game.get('execution_time_seconds') is not None and game.get('total_turns', 0) > 0:
                total_time += game.get('execution_time_seconds')
                total_turns += game.get('total_turns')
        
        avg_time_per_turn = total_time / total_turns if total_turns > 0 else 0
        
        return {
            'total_games': total_games,
            'detective_wins': detective_wins,
            'mr_x_wins': mr_x_wins,
            'incomplete_games': incomplete_games,
            'detective_win_rate': detective_wins / total_games * 100 if total_games > 0 else 0,
            'mr_x_win_rate': mr_x_wins / total_games * 100 if total_games > 0 else 0,
            'average_game_length': avg_length,
            'average_execution_time': avg_execution_time,
            'average_time_per_turn': avg_time_per_turn,
            'games_with_timing': len(execution_times),
            'completion_rate': (total_games - incomplete_games) / total_games * 100 if total_games > 0 else 0
        }


class GameAnalyzer:
    """Comprehensive game analyzer with visualization capabilities"""
    
    def __init__(self, base_directory: str):
        self.base_dir = Path(base_directory)
        print(f"📂 Initializing GameAnalyzer with base directory: {self.base_dir}")
        self.statistics = GameStatistics()
        self.graphs_dir = self.base_dir / "analysis_graphs"
        self.graphs_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib for better output
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
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
                        mr_x_agent, detective_agent = directory.name.split("_vs_")
                        game_data['mr_x_agent'] = mr_x_agent
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
        """Analyze performance by agent combination"""
        combinations = defaultdict(lambda: {'games': 0, 'detective_wins': 0, 'mr_x_wins': 0, 
                                          'total_turns': 0, 'completed_games': 0})
        
        for game in self.statistics.games:
            combo = game.get('agent_combination', 'unknown')
            combinations[combo]['games'] += 1
            
            if game.get('game_completed', False):
                combinations[combo]['completed_games'] += 1
                combinations[combo]['total_turns'] += game.get('total_turns', 0)
                
                if game.get('winner') == 'detectives':
                    combinations[combo]['detective_wins'] += 1
                elif game.get('winner') == 'mr_x':
                    combinations[combo]['mr_x_wins'] += 1
        
        # Calculate performance metrics
        for combo, stats in combinations.items():
            if stats['games'] > 0:
                stats['detective_win_rate'] = stats['detective_wins'] / stats['games'] * 100
                stats['mr_x_win_rate'] = stats['mr_x_wins'] / stats['games'] * 100
                stats['completion_rate'] = stats['completed_games'] / stats['games'] * 100
                stats['avg_game_length'] = (stats['total_turns'] / stats['completed_games'] 
                                          if stats['completed_games'] > 0 else 0)
        
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
        """Analyze win rates by agent type"""
        mr_x_performance = defaultdict(lambda: {'games': 0, 'wins': 0})
        detective_performance = defaultdict(lambda: {'games': 0, 'wins': 0})
        
        for game in self.statistics.games:
            mr_x_agent = game.get('mr_x_agent', 'unknown')
            detective_agent = game.get('detective_agent', 'unknown')
            
            mr_x_performance[mr_x_agent]['games'] += 1
            detective_performance[detective_agent]['games'] += 1
            
            if game.get('winner') == 'mr_x':
                mr_x_performance[mr_x_agent]['wins'] += 1
            elif game.get('winner') == 'detectives':
                detective_performance[detective_agent]['wins'] += 1
        
        # Calculate win rates
        for agent, stats in mr_x_performance.items():
            if stats['games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['games'] * 100
        
        for agent, stats in detective_performance.items():
            if stats['games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['games'] * 100
        
        self.statistics.win_rates = {
            'mr_x': dict(mr_x_performance),
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
        """Create win rate comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Mr. X win rates
        if self.statistics.win_rates.get('mr_x'):
            mr_x_data = self.statistics.win_rates['mr_x']
            agents = list(mr_x_data.keys())
            win_rates = [data['win_rate'] for data in mr_x_data.values()]
            games = [data['games'] for data in mr_x_data.values()]
            
            bars1 = ax1.bar(agents, win_rates, color=sns.color_palette("Set2", len(agents)))
            ax1.set_title('Mr. X Win Rates by Agent Type', fontsize=14, pad=20)
            ax1.set_ylabel('Win Rate (%)')
            ax1.set_ylim(0, 100)
            
            # Add value labels and game counts
            for i, (bar, rate, game_count) in enumerate(zip(bars1, win_rates, games)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%\\n({game_count} games)',
                        ha='center', va='bottom', fontsize=10)
        
        # Detective win rates
        if self.statistics.win_rates.get('detective'):
            detective_data = self.statistics.win_rates['detective']
            agents = list(detective_data.keys())
            win_rates = [data['win_rate'] for data in detective_data.values()]
            games = [data['games'] for data in detective_data.values()]
            
            bars2 = ax2.bar(agents, win_rates, color=sns.color_palette("Set1", len(agents)))
            ax2.set_title('Detective Win Rates by Agent Type', fontsize=14, pad=20)
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_ylim(0, 100)
            
            # Add value labels and game counts
            for i, (bar, rate, game_count) in enumerate(zip(bars2, win_rates, games)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%\\n({game_count} games)',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "win_rates_by_agent.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📈 Generated: win_rates_by_agent.jpg")
    
    def _create_game_length_analysis(self):
        """Create game length analysis"""
        if not self.statistics.game_lengths:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot of game lengths by combination
        combinations = list(self.statistics.game_lengths.keys())
        lengths_data = [self.statistics.game_lengths[combo] for combo in combinations]
        
        if lengths_data:
            box_plot = ax1.boxplot(lengths_data, labels=combinations, patch_artist=True)
            colors = sns.color_palette("Set3", len(combinations))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_title('Game Length Distribution by Agent Combination')
            ax1.set_ylabel('Game Length (turns)')
            ax1.tick_params(axis='x', rotation=45)
        
        # Histogram of all game lengths
        all_lengths = [length for lengths in lengths_data for length in lengths]
        if all_lengths:
            ax2.hist(all_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Overall Game Length Distribution')
            ax2.set_xlabel('Game Length (turns)')
            ax2.set_ylabel('Number of Games')
            ax2.axvline(np.mean(all_lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_lengths):.1f}')
            ax2.legend()
        
        # Average game length by combination
        avg_lengths = [np.mean(lengths) if lengths else 0 for lengths in lengths_data]
        if avg_lengths:
            bars = ax3.bar(range(len(combinations)), avg_lengths, 
                          color=sns.color_palette("viridis", len(combinations)))
            ax3.set_title('Average Game Length by Agent Combination')
            ax3.set_xlabel('Agent Combination')
            ax3.set_ylabel('Average Game Length (turns)')
            ax3.set_xticks(range(len(combinations)))
            ax3.set_xticklabels(combinations, rotation=45, ha='right')
            
            # Add value labels
            for bar, avg_len in zip(bars, avg_lengths):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{avg_len:.1f}', ha='center', va='bottom')
        
        # Game completion rates
        completion_rates = [
            self.statistics.agent_combinations[combo]['completion_rate'] 
            for combo in combinations 
            if combo in self.statistics.agent_combinations
        ]
        
        if completion_rates:
            bars = ax4.bar(range(len(combinations)), completion_rates,
                          color=sns.color_palette("plasma", len(combinations)))
            ax4.set_title('Game Completion Rate by Agent Combination')
            ax4.set_xlabel('Agent Combination')
            ax4.set_ylabel('Completion Rate (%)')
            ax4.set_xticks(range(len(combinations)))
            ax4.set_xticklabels(combinations, rotation=45, ha='right')
            ax4.set_ylim(0, 100)
            
            # Add value labels
            for bar, rate in zip(bars, completion_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Execution Time Analysis by Agent Combination', fontsize=16, fontweight='bold')
        
        # Box plot of execution times
        execution_data = [self.statistics.execution_times[combo] for combo in combinations]
        bp = ax1.boxplot(execution_data, labels=combinations, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("viridis", len(combinations))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title('Execution Time Distribution')
        ax1.set_xlabel('Agent Combination')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average execution times bar chart
        avg_times = [np.mean(self.statistics.execution_times[combo]) for combo in combinations]
        bars = ax2.bar(range(len(combinations)), avg_times,
                      color=sns.color_palette("plasma", len(combinations)))
        ax2.set_title('Average Execution Time by Agent Combination')
        ax2.set_xlabel('Agent Combination')
        ax2.set_ylabel('Average Time (seconds)')
        ax2.set_xticks(range(len(combinations)))
        ax2.set_xticklabels(combinations, rotation=45, ha='right')
        
        # Add value labels
        for bar, avg_time in zip(bars, avg_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{avg_time:.2f}s', ha='center', va='bottom')
        
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
            scatter = ax3.scatter(all_game_lengths, all_exec_times, c=colors_scatter, alpha=0.6)
            ax3.set_title('Execution Time vs Game Length')
            ax3.set_xlabel('Game Length (turns)')
            ax3.set_ylabel('Execution Time (seconds)')
            
            # Add trend line
            if len(all_game_lengths) > 1:
                z = np.polyfit(all_game_lengths, all_exec_times, 1)
                p = np.poly1d(z)
                ax3.plot(sorted(all_game_lengths), p(sorted(all_game_lengths)), 
                        "r--", alpha=0.8, linewidth=1)
        
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
        ax4.set_title('Theoretical Games per Minute')
        ax4.set_xlabel('Agent Combination')
        ax4.set_ylabel('Games per Minute')
        ax4.set_xticks(range(len(combinations)))
        ax4.set_xticklabels(combinations, rotation=45, ha='right')
        
        # Add value labels
        for bar, gpm in zip(bars, games_per_min):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{gpm:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "execution_time_analysis.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Generated: execution_time_analysis.jpg")
    
    def _create_agent_performance_matrix(self):
        """Create agent performance matrix heatmap"""
        if not self.statistics.agent_combinations:
            return
        
        # Extract agent types
        mr_x_agents = set()
        detective_agents = set()
        
        for combo in self.statistics.agent_combinations.keys():
            if "_vs_" in combo:
                mr_x, detective = combo.split("_vs_")
                mr_x_agents.add(mr_x)
                detective_agents.add(detective)
        
        mr_x_agents = sorted(list(mr_x_agents))
        detective_agents = sorted(list(detective_agents),reverse=True)
        
        if not mr_x_agents or not detective_agents:
            return
        
        # Create performance matrices
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mr. X win rate matrix
        mr_x_matrix = np.zeros((len(mr_x_agents), len(detective_agents)))
        detective_matrix = np.zeros((len(mr_x_agents), len(detective_agents)))
        game_count_matrix = np.zeros((len(mr_x_agents), len(detective_agents)))
        avg_length_matrix = np.zeros((len(mr_x_agents), len(detective_agents)))
        
        for i, mr_x in enumerate(mr_x_agents):
            for j, detective in enumerate(detective_agents):
                combo = f"{mr_x}_vs_{detective}"
                if combo in self.statistics.agent_combinations:
                    stats = self.statistics.agent_combinations[combo]
                    mr_x_matrix[i, j] = stats['mr_x_win_rate']
                    detective_matrix[i, j] = stats['detective_win_rate']
                    game_count_matrix[i, j] = stats['games']
                    avg_length_matrix[i, j] = stats['avg_game_length']
        
        # Mr. X win rate heatmap
        sns.heatmap(mr_x_matrix, annot=True, fmt='.1f', 
                   xticklabels=detective_agents, yticklabels=mr_x_agents,
                   cmap='Reds', ax=ax1, cbar_kws={'label': 'Win Rate (%)'})
        ax1.set_title('Mr. X Win Rate Matrix')
        ax1.set_xlabel('Detective Agent')
        ax1.set_ylabel('Mr. X Agent')
        
        # Detective win rate heatmap
        sns.heatmap(detective_matrix, annot=True, fmt='.1f',
                   xticklabels=detective_agents, yticklabels=mr_x_agents,
                   cmap='Blues', ax=ax2, cbar_kws={'label': 'Win Rate (%)'})
        ax2.set_title('Detective Win Rate Matrix')
        ax2.set_xlabel('Detective Agent')
        ax2.set_ylabel('Mr. X Agent')
        
        # Game count heatmap
        sns.heatmap(game_count_matrix, annot=True, fmt='.0f',
                   xticklabels=detective_agents, yticklabels=mr_x_agents,
                   cmap='Greens', ax=ax3, cbar_kws={'label': 'Number of Games'})
        ax3.set_title('Number of Games Matrix')
        ax3.set_xlabel('Detective Agent')
        ax3.set_ylabel('Mr. X Agent')
        
        # Average game length heatmap
        sns.heatmap(avg_length_matrix, annot=True, fmt='.1f',
                   xticklabels=detective_agents, yticklabels=mr_x_agents,
                   cmap='Purples', ax=ax4, cbar_kws={'label': 'Average Turns'})
        ax4.set_title('Average Game Length Matrix')
        ax4.set_xlabel('Detective Agent')
        ax4.set_ylabel('Mr. X Agent')
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "agent_performance_matrix.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   🔥 Generated: agent_performance_matrix.jpg")
    
    def _create_temporal_analysis(self):
        """Create temporal analysis if data is available"""
        if not self.statistics.temporal_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Games over time
        dates = sorted(self.statistics.temporal_data.keys())
        counts = [self.statistics.temporal_data[date] for date in dates]
        
        ax1.plot(dates, counts, marker='o', linewidth=2, markersize=6)
        ax1.set_title('Games Played Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Games')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cumulative games
        cumulative = np.cumsum(counts)
        ax2.plot(dates, cumulative, marker='s', linewidth=2, markersize=6, color='green')
        ax2.set_title('Cumulative Games Played')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Games')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.graphs_dir / "temporal_analysis.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📅 Generated: temporal_analysis.jpg")
    
    def _create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard summary"""
        summary = self.statistics.get_summary()
        if not summary:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Overall statistics (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        stats_text = f"""
OVERALL STATISTICS
==================
Total Games: {summary['total_games']:,}
Completed Games: {summary['total_games'] - summary['incomplete_games']:,}
Completion Rate: {summary['completion_rate']:.1f}%

WINNER BREAKDOWN
================
Detective Wins: {summary['detective_wins']:,} ({summary['detective_win_rate']:.1f}%)
Mr. X Wins: {summary['mr_x_wins']:,} ({summary['mr_x_win_rate']:.1f}%)
Incomplete: {summary['incomplete_games']:,}

GAME METRICS
============
Average Game Length: {summary['average_game_length']:.1f} turns
        """
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.axis('off')
        
        # Win distribution pie chart (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if summary['detective_wins'] > 0 or summary['mr_x_wins'] > 0:
            sizes = [summary['detective_wins'], summary['mr_x_wins']]
            labels = ['Detective Wins', 'Mr. X Wins']
            colors = ['lightcoral', 'lightskyblue']
            
            if summary['incomplete_games'] > 0:
                sizes.append(summary['incomplete_games'])
                labels.append('Incomplete')
                colors.append('lightgray')
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Game Outcomes Distribution', pad=20)
        
        # Agent combination performance (middle)
        if self.statistics.agent_combinations:
            ax3 = fig.add_subplot(gs[1, :])
            combinations = list(self.statistics.agent_combinations.keys())[:10]  # Top 10
            detective_rates = [self.statistics.agent_combinations[combo]['detective_win_rate'] 
                             for combo in combinations]
            mr_x_rates = [self.statistics.agent_combinations[combo]['mr_x_win_rate'] 
                         for combo in combinations]
            
            x = np.arange(len(combinations))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, detective_rates, width, label='Detective Win Rate', 
                           color='lightcoral', alpha=0.8)
            bars2 = ax3.bar(x + width/2, mr_x_rates, width, label='Mr. X Win Rate', 
                           color='lightskyblue', alpha=0.8)
            
            ax3.set_title('Win Rates by Agent Combination')
            ax3.set_xlabel('Agent Combination')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(combinations, rotation=45, ha='right')
            ax3.legend()
            ax3.set_ylim(0, 100)
        
        # Game length distribution (bottom left)
        if self.statistics.game_lengths:
            ax4 = fig.add_subplot(gs[2, :2])
            all_lengths = [length for lengths in self.statistics.game_lengths.values() 
                          for length in lengths]
            if all_lengths:
                ax4.hist(all_lengths, bins=20, alpha=0.7, color='mediumseagreen', edgecolor='black')
                ax4.set_title('Game Length Distribution')
                ax4.set_xlabel('Game Length (turns)')
                ax4.set_ylabel('Frequency')
                ax4.axvline(np.mean(all_lengths), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_lengths):.1f}')
                ax4.legend()
        
        # Agent performance summary (bottom right)
        ax5 = fig.add_subplot(gs[2, 2:])
        if self.statistics.win_rates.get('mr_x') and self.statistics.win_rates.get('detective'):
            mr_x_agents = list(self.statistics.win_rates['mr_x'].keys())
            mr_x_rates = [self.statistics.win_rates['mr_x'][agent]['win_rate'] for agent in mr_x_agents]
            
            detective_agents = list(self.statistics.win_rates['detective'].keys())
            detective_rates = [self.statistics.win_rates['detective'][agent]['win_rate'] 
                             for agent in detective_agents]
            
            x1 = np.arange(len(mr_x_agents))
            x2 = np.arange(len(detective_agents)) + len(mr_x_agents) + 0.5
            
            bars1 = ax5.bar(x1, mr_x_rates, color='lightskyblue', alpha=0.8, label='Mr. X Agents')
            bars2 = ax5.bar(x2, detective_rates, color='lightcoral', alpha=0.8, label='Detective Agents')
            
            ax5.set_title('Agent Type Performance')
            ax5.set_ylabel('Win Rate (%)')
            ax5.set_xticks(list(x1) + list(x2))
            ax5.set_xticklabels(mr_x_agents + detective_agents, rotation=45, ha='right')
            ax5.legend()
            ax5.set_ylim(0, 100)
        
        plt.suptitle('Scotland Yard Game Analysis Dashboard', fontsize=16, y=0.98)
        plt.savefig(self.graphs_dir / "comprehensive_dashboard.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Generated: comprehensive_dashboard.jpg")
    
    def _generate_summary_report(self):
        """Generate a detailed summary report"""
        report_file = self.graphs_dir.parent / "analysis_report.txt"
        
        summary = self.statistics.get_summary()
        
        with open(report_file, 'w') as f:
            f.write("SCOTLAND YARD GAME ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Directory: {self.base_dir}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Games: {summary.get('total_games', 0):,}\n")
            f.write(f"Detective Wins: {summary.get('detective_wins', 0):,} ({summary.get('detective_win_rate', 0):.1f}%)\n")
            f.write(f"Mr. X Wins: {summary.get('mr_x_wins', 0):,} ({summary.get('mr_x_win_rate', 0):.1f}%)\n")
            f.write(f"Incomplete Games: {summary.get('incomplete_games', 0):,}\n")
            f.write(f"Average Game Length: {summary.get('average_game_length', 0):.1f} turns\n")
            f.write(f"Average Execution Time: {summary.get('average_execution_time', 0):.5f} seconds\n")
            f.write(f"Average Time per Turn: {summary.get('average_time_per_turn', 0):.5f} seconds\n")
            f.write(f"Games with Timing Data: {summary.get('games_with_timing', 0):,}\n")
            f.write(f"Completion Rate: {summary.get('completion_rate', 0):.1f}%\n\n")
            
            # Agent combination details
            if self.statistics.agent_combinations:
                f.write("AGENT COMBINATION PERFORMANCE\n")
                f.write("-" * 32 + "\n")
                for combo, stats in sorted(self.statistics.agent_combinations.items()):
                    f.write(f"{combo}:\n")
                    f.write(f"  Games: {stats['games']:,}\n")
                    f.write(f"  Detective Win Rate: {stats['detective_win_rate']:.1f}%\n")
                    f.write(f"  Mr. X Win Rate: {stats['mr_x_win_rate']:.1f}%\n")
                    f.write(f"  Avg Game Length: {stats['avg_game_length']:.1f} turns\n")
                    f.write(f"  Completion Rate: {stats['completion_rate']:.1f}%\n")
                    
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
            
            # Generated files
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 25 + "\n")
            graph_files = list(self.graphs_dir.glob("*.jpg"))
            for graph_file in sorted(graph_files):
                f.write(f"- {graph_file.name}\n")
        
        print(f"📝 Generated: {report_file.name}")


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
    
    print(f"🔍 SCOTLAND YARD GAME ANALYSIS")
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
