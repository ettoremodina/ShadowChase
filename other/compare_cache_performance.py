#!/usr/bin/env python3
"""
Cache Performance Comparison Script

This script compares game performance with and without cache to determine
if the caching system improves or degrades performance.

Features:
- Run identical games with cache enabled/disabled
- Compare execution times, memory usage, and decision quality
- Statistical analysis of performance differences
- Generate comparison reports
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from simple_play.game_utils import play_single_game, play_multiple_games
from simple_play.display_utils import VerbosityLevel
from agents import AgentType
from ScotlandYard.services.cache_system import enable_cache, disable_cache, is_cache_enabled, get_global_cache


class CachePerformanceComparison:
    """Compare performance with and without cache."""
    
    def __init__(self, save_dir: str = "cache_performance_comparison"):
        self.save_dir = save_dir
        self.results_dir = Path(save_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.cache_enabled_results = []
        self.cache_disabled_results = []
        
    def run_performance_test(self, 
                           num_games: int = 10,
                           map_size: str = "test",
                           mr_x_agent: AgentType = AgentType.OPTIMIZED_MCTS,
                           detective_agent: AgentType = AgentType.OPTIMIZED_MCTS,
                           max_turns: int = 24) -> Dict[str, Any]:
        """
        Run performance comparison between cache enabled and disabled.
        
        Args:
            num_games: Number of games to run for each test
            map_size: Size of the game map
            mr_x_agent: Agent type for Mr. X
            detective_agent: Agent type for detectives
            max_turns: Maximum turns per game
            
        Returns:
            Dictionary with comparison results
        """
        print("🔬 CACHE PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"Testing {num_games} games with each cache setting")
        print(f"Map: {map_size}, Max turns: {max_turns}")
        print(f"Mr. X: {mr_x_agent.value}, Detectives: {detective_agent.value}")
        
        # Test 1: Cache enabled
        print(f"\n🟢 PHASE 1: Testing with cache ENABLED")
        enable_cache()
        cache_enabled_results = self._run_test_batch(
            num_games, map_size, mr_x_agent, detective_agent, max_turns, "cache_enabled"
        )
        
        # Clear cache for fair comparison
        cache = get_global_cache()
        cache.clear_all()
        
        # Test 2: Cache disabled
        print(f"\n🔴 PHASE 2: Testing with cache DISABLED")
        disable_cache()
        cache_disabled_results = self._run_test_batch(
            num_games, map_size, mr_x_agent, detective_agent, max_turns, "cache_disabled"
        )
        
        # Re-enable cache for normal operation
        enable_cache()
        
        # Analyze results
        analysis = self._analyze_performance_difference(cache_enabled_results, cache_disabled_results)
        
        # Save results
        self._save_comparison_report(cache_enabled_results, cache_disabled_results, analysis)
        
        return {
            'cache_enabled': cache_enabled_results,
            'cache_disabled': cache_disabled_results,
            'analysis': analysis
        }
    
    def _run_test_batch(self, num_games: int, map_size: str, mr_x_agent: AgentType, 
                       detective_agent: AgentType, max_turns: int, test_name: str) -> Dict[str, Any]:
        """Run a batch of games and collect performance metrics."""
        print(f"   Running {num_games} games...")
        
        game_results = []
        total_start_time = time.time()
        
        for game_num in tqdm(range(num_games), desc=f"{test_name} games"):
            game_start_time = time.time()
            
            # Run single game
            game_id, turn_count, completed = play_single_game(
                map_size=map_size,
                play_mode="ai_vs_ai",
                num_detectives=2,
                verbosity=VerbosityLevel.SILENT,
                auto_save=True,
                max_turns=max_turns,
                mr_x_agent_type=mr_x_agent,
                detective_agent_type=detective_agent,
                save_dir=self.save_dir
            )
            
            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            
            game_results.append({
                'game_id': game_id,
                'game_number': game_num + 1,
                'turn_count': turn_count,
                'completed': completed,
                'duration': game_duration,
                'cache_enabled': is_cache_enabled()
            })
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Get cache stats if cache is enabled
        cache_stats = None
        if is_cache_enabled():
            cache = get_global_cache()
            cache_stats = cache.get_global_stats()
        
        return {
            'test_name': test_name,
            'num_games': num_games,
            'total_duration': total_duration,
            'avg_duration_per_game': total_duration / num_games,
            'games': game_results,
            'cache_stats': cache_stats,
            'cache_enabled': is_cache_enabled()
        }
    
    def _analyze_performance_difference(self, cache_enabled: Dict, cache_disabled: Dict) -> Dict[str, Any]:
        """Analyze the performance difference between cache enabled and disabled."""
        
        # Extract timing data
        cache_enabled_times = [game['duration'] for game in cache_enabled['games']]
        cache_disabled_times = [game['duration'] for game in cache_disabled['games']]
        
        cache_enabled_turns = [game['turn_count'] for game in cache_enabled['games']]
        cache_disabled_turns = [game['turn_count'] for game in cache_disabled['games']]
        
        # Calculate statistics
        analysis = {
            'timing_analysis': {
                'cache_enabled_avg': np.mean(cache_enabled_times),
                'cache_disabled_avg': np.mean(cache_disabled_times),
                'cache_enabled_std': np.std(cache_enabled_times),
                'cache_disabled_std': np.std(cache_disabled_times),
                'speedup_factor': np.mean(cache_disabled_times) / np.mean(cache_enabled_times),
                'time_difference_seconds': np.mean(cache_disabled_times) - np.mean(cache_enabled_times),
                'time_difference_percent': ((np.mean(cache_disabled_times) - np.mean(cache_enabled_times)) / np.mean(cache_disabled_times)) * 100
            },
            'game_length_analysis': {
                'cache_enabled_avg_turns': np.mean(cache_enabled_turns),
                'cache_disabled_avg_turns': np.mean(cache_disabled_turns),
                'turn_difference': np.mean(cache_enabled_turns) - np.mean(cache_disabled_turns)
            },
            'cache_effectiveness': {},
            'recommendation': ''
        }
        
        # Cache effectiveness analysis
        if cache_enabled['cache_stats']:
            stats = cache_enabled['cache_stats']['global_stats']
            analysis['cache_effectiveness'] = {
                'hit_rate': stats.get('hit_rate', 0) * 100,
                'total_requests': stats.get('total_requests', 0),
                'total_hits': stats.get('hits', 0),
                'total_misses': stats.get('misses', 0),
                'cache_size_mb': stats.get('cache_size_mb', 0),
                'total_entries': stats.get('total_entries', 0)
            }
        
        # Generate recommendation
        speedup = analysis['timing_analysis']['speedup_factor']
        time_diff_pct = analysis['timing_analysis']['time_difference_percent']
        
        if speedup > 1.1:  # Cache provides >10% speedup
            analysis['recommendation'] = f"🟢 ENABLE CACHE: {speedup:.2f}x speedup ({time_diff_pct:.1f}% faster)"
        elif speedup < 0.9:  # Cache causes >10% slowdown
            analysis['recommendation'] = f"🔴 DISABLE CACHE: {1/speedup:.2f}x slowdown ({abs(time_diff_pct):.1f}% slower)"
        else:
            analysis['recommendation'] = f"🟡 NEUTRAL: Minimal performance impact ({time_diff_pct:.1f}% difference)"
        
        return analysis
    
    def _save_comparison_report(self, cache_enabled: Dict, cache_disabled: Dict, analysis: Dict):
        """Save detailed comparison report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"cache_performance_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("CACHE PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test configuration
            f.write("TEST CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of games per test: {cache_enabled['num_games']}\n")
            f.write(f"Cache enabled during test 1: {cache_enabled['cache_enabled']}\n")
            f.write(f"Cache enabled during test 2: {cache_disabled['cache_enabled']}\n\n")
            
            # Performance summary
            timing = analysis['timing_analysis']
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Cache ENABLED average time:  {timing['cache_enabled_avg']:.3f}s ± {timing['cache_enabled_std']:.3f}s\n")
            f.write(f"Cache DISABLED average time: {timing['cache_disabled_avg']:.3f}s ± {timing['cache_disabled_std']:.3f}s\n")
            f.write(f"Speedup factor:              {timing['speedup_factor']:.3f}x\n")
            f.write(f"Time difference:             {timing['time_difference_seconds']:+.3f}s ({timing['time_difference_percent']:+.1f}%)\n\n")
            
            # Game length analysis
            game_length = analysis['game_length_analysis']
            f.write("GAME LENGTH ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Cache ENABLED avg turns:     {game_length['cache_enabled_avg_turns']:.1f}\n")
            f.write(f"Cache DISABLED avg turns:    {game_length['cache_disabled_avg_turns']:.1f}\n")
            f.write(f"Turn difference:             {game_length['turn_difference']:+.1f}\n\n")
            
            # Cache effectiveness
            if analysis['cache_effectiveness']:
                cache_eff = analysis['cache_effectiveness']
                f.write("CACHE EFFECTIVENESS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Hit rate:                    {cache_eff['hit_rate']:.1f}%\n")
                f.write(f"Total requests:              {cache_eff['total_requests']:,}\n")
                f.write(f"Total hits:                  {cache_eff['total_hits']:,}\n")
                f.write(f"Total misses:                {cache_eff['total_misses']:,}\n")
                f.write(f"Cache size:                  {cache_eff['cache_size_mb']:.1f} MB\n")
                f.write(f"Total entries:               {cache_eff['total_entries']:,}\n\n")
            
            # Recommendation
            f.write("RECOMMENDATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{analysis['recommendation']}\n\n")
            
            # Individual game results
            f.write("DETAILED GAME RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write("Cache ENABLED games:\n")
            for game in cache_enabled['games']:
                f.write(f"  Game {game['game_number']:2d}: {game['duration']:6.2f}s, {game['turn_count']:2d} turns\n")
            
            f.write("\nCache DISABLED games:\n")
            for game in cache_disabled['games']:
                f.write(f"  Game {game['game_number']:2d}: {game['duration']:6.2f}s, {game['turn_count']:2d} turns\n")
        
        print(f"📝 Comparison report saved to: {report_file}")
        
        # Create visualization
        self._create_performance_plots(cache_enabled, cache_disabled, analysis, timestamp)
        
        return report_file
    
    def _create_performance_plots(self, cache_enabled: Dict, cache_disabled: Dict, 
                                 analysis: Dict, timestamp: str):
        """Create performance comparison plots."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cache Performance Comparison', fontsize=16)
        
        # Extract data
        cache_enabled_times = [game['duration'] for game in cache_enabled['games']]
        cache_disabled_times = [game['duration'] for game in cache_disabled['games']]
        cache_enabled_turns = [game['turn_count'] for game in cache_enabled['games']]
        cache_disabled_turns = [game['turn_count'] for game in cache_disabled['games']]
        
        game_numbers = list(range(1, len(cache_enabled_times) + 1))
        
        # Plot 1: Execution time comparison
        ax1.plot(game_numbers, cache_enabled_times, 'g-o', label='Cache Enabled', alpha=0.7)
        ax1.plot(game_numbers, cache_disabled_times, 'r-s', label='Cache Disabled', alpha=0.7)
        ax1.set_title('Execution Time per Game')
        ax1.set_xlabel('Game Number')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        ax2.boxplot([cache_enabled_times, cache_disabled_times], 
                   labels=['Cache Enabled', 'Cache Disabled'])
        ax2.set_title('Execution Time Distribution')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Turn count comparison
        ax3.plot(game_numbers, cache_enabled_turns, 'g-o', label='Cache Enabled', alpha=0.7)
        ax3.plot(game_numbers, cache_disabled_turns, 'r-s', label='Cache Disabled', alpha=0.7)
        ax3.set_title('Game Length (Turns)')
        ax3.set_xlabel('Game Number')
        ax3.set_ylabel('Number of Turns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        speedup = analysis['timing_analysis']['speedup_factor']
        time_diff_pct = analysis['timing_analysis']['time_difference_percent']
        
        categories = ['Speedup Factor', 'Time Difference (%)']
        values = [speedup, time_diff_pct]
        colors = ['green' if v > 0 else 'red' for v in [speedup - 1, time_diff_pct]]
        
        bars = ax4.bar(categories, [speedup, abs(time_diff_pct)], color=colors, alpha=0.7)
        ax4.set_title('Performance Summary')
        ax4.set_ylabel('Factor / Percentage')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
        
        # Add value labels on bars
        for bar, value in zip(bars, [speedup, time_diff_pct]):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"cache_performance_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to: {plot_file}")
        
        return plot_file


def main():
    """Main execution function."""
    print("🚀 Cache Performance Comparison Tool")
    print("=" * 50)
    
    # Configuration
    num_games = 20  # Number of games per test
    map_size = "test"
    mr_x_agent = AgentType.OPTIMIZED_MCTS
    detective_agent = AgentType.OPTIMIZED_MCTS
    max_turns = 24
    
    print(f"Configuration:")
    print(f"  Games per test: {num_games}")
    print(f"  Map size: {map_size}")
    print(f"  Mr. X agent: {mr_x_agent.value}")
    print(f"  Detective agent: {detective_agent.value}")
    print(f"  Max turns: {max_turns}")
    
    # Run comparison
    comparison = CachePerformanceComparison()
    
    try:
        results = comparison.run_performance_test(
            num_games=num_games,
            map_size=map_size,
            mr_x_agent=mr_x_agent,
            detective_agent=detective_agent,
            max_turns=max_turns
        )
        
        print(f"\n🎉 Performance comparison completed!")
        print(f"📊 Results: {results['analysis']['recommendation']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
