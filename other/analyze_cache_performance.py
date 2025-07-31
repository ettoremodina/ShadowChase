#!/usr/bin/env python3
"""
Cache Performance Analysis Script

This script analyzes how game length and performance change as the persistent cache
gets populated with data from thousands of games.

Process:
1. Play initial MCTS vs MCTS game (baseline)
2. Play 1000 random vs random games to populate cache
3. Play final MCTS vs MCTS game (post-cache)
4. Create plots showing game length changes
5. Save comparison data
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

from game_controls.game_utils import play_single_game, play_multiple_games
from game_controls.display_utils import VerbosityLevel
from ScotlandYard.services.cache_system import get_global_cache, CacheNamespace
from ScotlandYard.services.game_loader import GameLoader
from ScotlandYard.services.game_service import GameService

from agents import AgentType

class CachePerformanceAnalyzer:
    """Analyzer for cache performance impact on game length and speed."""
    
    def __init__(self, save_dir: str = "cache_analysis"):
        self.save_dir = save_dir
        self.results_dir = Path(save_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize cache and clear it for clean testing
        self.cache = get_global_cache()
        self.initial_cache_stats = self.cache.get_global_stats()
        
        # Data storage
        self.game_length_data = []
        self.cache_size_data = []
        self.execution_time_data = []
        self.cache_hit_rates = []  # Track hit rates over time
        self.mcts_comparison_data = {}
        
    def clear_cache_for_testing(self):
        """Clear the cache to start with a clean slate."""
        print("🧹 Clearing cache for clean testing...")
        self.cache.clear_all()
        print(f"Cache cleared. Current stats: {self.cache.get_global_stats()}")
    
    def play_mcts_baseline_game(self, game_label: str = "baseline") -> Dict[str, Any]:
        """Play an MCTS vs MCTS game and record performance metrics."""
        print(f"\n🤖 Playing MCTS vs MCTS game ({game_label})...")
        
        start_time = time.time()
        cache_stats_before = self.cache.get_global_stats()
        
        # Play single MCTS vs MCTS game with high verbosity for detailed info
        game_id, turn_count, completed = play_single_game(
            map_size="extended",
            play_mode="ai_vs_ai",
            num_detectives=5,
            verbosity=VerbosityLevel.BASIC,
            auto_save=True,
            max_turns=24,
            mr_x_agent_type=AgentType.OPTIMIZED_MCTS,
            detective_agent_type=AgentType.OPTIMIZED_MCTS,
            save_dir=self.save_dir
        )
        
        end_time = time.time()
        cache_stats_after = self.cache.get_global_stats()
        execution_time = end_time - start_time
        
        # Calculate cache performance metrics from global stats
        hits_before = cache_stats_before.get('global_stats', {}).get('hits', 0)
        hits_after = cache_stats_after.get('global_stats', {}).get('hits', 0)
        misses_before = cache_stats_before.get('global_stats', {}).get('misses', 0)
        misses_after = cache_stats_after.get('global_stats', {}).get('misses', 0)
        
        total_hits_delta = hits_after - hits_before
        total_misses_delta = misses_after - misses_before
        total_operations = total_hits_delta + total_misses_delta
        hit_rate = (total_hits_delta / total_operations * 100) if total_operations > 0 else 0
        
        # For namespace breakdown, we'll use entry counts as proxy since hits/misses aren't tracked per namespace
        cache_hits_delta = {}
        cache_misses_delta = {}
        for namespace in CacheNamespace:
            ns_name = namespace.value
            ns_before = cache_stats_before.get('namespace_stats', {}).get(ns_name, {})
            ns_after = cache_stats_after.get('namespace_stats', {}).get(ns_name, {})
            
            entries_before = ns_before.get('total_entries', 0)
            entries_after = ns_after.get('total_entries', 0)
            
            # Use entry delta as approximation since namespace hits/misses aren't tracked
            cache_hits_delta[ns_name] = max(0, entries_after - entries_before)
            cache_misses_delta[ns_name] = 0  # Can't determine from available data
        
        result = {
            'game_id': game_id,
            'turn_count': turn_count,
            'completed': completed,
            'execution_time': execution_time,
            'cache_stats_before': cache_stats_before,
            'cache_stats_after': cache_stats_after,
            'cache_hits_delta': cache_hits_delta,
            'cache_misses_delta': cache_misses_delta,
            'total_cache_entries_before': cache_stats_before.get('global_stats', {}).get('total_entries', 0),
            'total_cache_entries_after': cache_stats_after.get('global_stats', {}).get('total_entries', 0),
            'total_hits_delta': total_hits_delta,
            'total_misses_delta': total_misses_delta,
            'game_hit_rate': hit_rate
        }
        
        print(f"✅ MCTS game ({game_label}) completed:")
        print(f"   Turns: {turn_count}")
        print(f"   Time: {execution_time:.2f}s")
        print(f"   Cache entries before: {result['total_cache_entries_before']}")
        print(f"   Cache entries after: {result['total_cache_entries_after']}")
        
        return result
    
    def play_cache_population_games(self, num_games: int = 1000) -> Dict[str, Any]:
        """Play many random vs random games to populate the cache."""
        print(f"\n🎲 Playing {num_games} random vs random games to populate cache...")
        
        # Record cache stats every N games to track growth
        checkpoint_interval = max(1, num_games // 20)  # 20 checkpoints
        checkpoints = []
        
        start_time = time.time()
        
        # Use play_multiple_games but track cache growth during execution
        for batch_start in tqdm(range(0, num_games, checkpoint_interval), desc="Game batches"):
            batch_size = min(checkpoint_interval, num_games - batch_start)
            
            batch_start_time = time.time()
            cache_stats_before = self.cache.get_global_stats()
            
            # Play batch of games
            batch_results = play_multiple_games(
                n_games=batch_size,
                map_size="extended",
                play_mode="ai_vs_ai",
                num_detectives=5,
                verbosity=VerbosityLevel.SILENT,  # Silent for batch processing
                max_turns=24,
                mr_x_agent_type=AgentType.RANDOM,
                detective_agent_type=AgentType.RANDOM,
                save_dir=self.save_dir
            )
            
            batch_end_time = time.time()
            cache_stats_after = self.cache.get_global_stats()
            
            # Record checkpoint data
            checkpoint = {
                'games_completed': batch_start + batch_size,
                'batch_execution_time': batch_end_time - batch_start_time,
                'total_cache_entries': cache_stats_after.get('global_stats', {}).get('total_entries', 0),
                'cache_stats': cache_stats_after,
                'batch_results': batch_results
            }
            checkpoints.append(checkpoint)
            
            # Store data for plotting
            self.cache_size_data.append(checkpoint['total_cache_entries'])
            current_hit_rate = cache_stats_after.get('global_stats', {}).get('hit_rate', 0) * 100
            self.cache_hit_rates.append(current_hit_rate)
            
            if batch_results.get('completed_games', 0) > 0:
                avg_turns = batch_results['total_turns'] / batch_results['completed_games']
                self.game_length_data.append(avg_turns)
                self.execution_time_data.append(checkpoint['batch_execution_time'] / batch_size)
        
        total_time = time.time() - start_time
        
        final_stats = {
            'total_games': num_games,
            'total_time': total_time,
            'checkpoints': checkpoints,
            'final_cache_size': checkpoints[-1]['total_cache_entries'] if checkpoints else 0
        }
        
        print(f"✅ Cache population completed:")
        print(f"   Total games: {num_games}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Final cache size: {final_stats['final_cache_size']} entries")
        
        return final_stats
    
    def create_performance_plots(self):
        """Create plots showing how performance changes with cache population."""
        print("\n📊 Creating performance analysis plots...")
        
        # Prepare data for plotting
        games_played = [i * len(self.game_length_data) // len(self.cache_size_data) for i in range(len(self.cache_size_data))]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cache Performance Analysis: Impact on Game Characteristics', fontsize=16)
        
        # Plot 1: Game length vs number of games played
        if self.game_length_data:
            ax1.plot(games_played[:len(self.game_length_data)], self.game_length_data, 'b-', alpha=0.7, linewidth=2)
            ax1.set_title('Average Game Length vs Games Played')
            ax1.set_xlabel('Games Played')
            ax1.set_ylabel('Average Turns per Game')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            if len(self.game_length_data) > 1:
                z = np.polyfit(games_played[:len(self.game_length_data)], self.game_length_data, 1)
                p = np.poly1d(z)
                ax1.plot(games_played[:len(self.game_length_data)], p(games_played[:len(self.game_length_data)]), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
                ax1.legend()
        
        # Plot 2: Cache size growth
        if self.cache_size_data:
            ax2.plot(games_played, self.cache_size_data, 'g-', linewidth=2)
            ax2.set_title('Cache Size Growth')
            ax2.set_xlabel('Games Played')
            ax2.set_ylabel('Total Cache Entries')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Execution time per game
        if self.execution_time_data:
            ax3.plot(games_played[:len(self.execution_time_data)], self.execution_time_data, 'r-', alpha=0.7, linewidth=2)
            ax3.set_title('Average Execution Time per Game')
            ax3.set_xlabel('Games Played')
            ax3.set_ylabel('Time per Game (seconds)')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(self.execution_time_data) > 1:
                z = np.polyfit(games_played[:len(self.execution_time_data)], self.execution_time_data, 1)
                p = np.poly1d(z)
                ax3.plot(games_played[:len(self.execution_time_data)], p(games_played[:len(self.execution_time_data)]), "orange", linestyle="--", alpha=0.8, label=f'Trend: {z[0]:.6f}x + {z[1]:.4f}')
                ax3.legend()
        
        # Plot 4: Cache hit rate over time
        if self.cache_hit_rates:
            ax4.plot(games_played[:len(self.cache_hit_rates)], self.cache_hit_rates, 'm-', linewidth=2)
            ax4.set_title('Cache Hit Rate Over Time')
            ax4.set_xlabel('Games Played')
            ax4.set_ylabel('Cache Hit Rate (%)')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)  # Set y-axis limit for percentage
        else:
            ax4.text(0.5, 0.5, 'Cache Hit Rate Data\nNot Available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Cache Hit Rate Over Time')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"cache_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to: {plot_file}")
        
        return plot_file
    
    def save_mcts_comparison_report(self):
        """Save detailed comparison of MCTS games before and after cache population."""
        if not self.mcts_comparison_data:
            print("⚠️ No MCTS comparison data available")
            return
        
        report_file = self.results_dir / f"mcts_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("MCTS PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            baseline = self.mcts_comparison_data.get('baseline', {})
            post_cache = self.mcts_comparison_data.get('post_cache', {})
            
            if baseline and post_cache:
                f.write("GAME LENGTH COMPARISON:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Baseline MCTS game turns:    {baseline.get('turn_count', 'N/A')}\n")
                f.write(f"Post-cache MCTS game turns:  {post_cache.get('turn_count', 'N/A')}\n")
                
                if baseline.get('turn_count') and post_cache.get('turn_count'):
                    turn_diff = post_cache['turn_count'] - baseline['turn_count']
                    turn_pct = (turn_diff / baseline['turn_count']) * 100
                    f.write(f"Turn difference:             {turn_diff:+d} ({turn_pct:+.1f}%)\n")
                
                f.write("\nEXECUTION TIME COMPARISON:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Baseline execution time:     {baseline.get('execution_time', 0):.3f}s\n")
                f.write(f"Post-cache execution time:   {post_cache.get('execution_time', 0):.3f}s\n")
                
                if baseline.get('execution_time') and post_cache.get('execution_time'):
                    time_diff = post_cache['execution_time'] - baseline['execution_time']
                    time_pct = (time_diff / baseline['execution_time']) * 100
                    f.write(f"Time difference:             {time_diff:+.3f}s ({time_pct:+.1f}%)\n")
                
                f.write("\nCACHE UTILIZATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Cache entries before baseline:  {baseline.get('total_cache_entries_before', 0)}\n")
                f.write(f"Cache entries after baseline:   {baseline.get('total_cache_entries_after', 0)}\n")
                f.write(f"Cache entries before post-test: {post_cache.get('total_cache_entries_before', 0)}\n")
                f.write(f"Cache entries after post-test:  {post_cache.get('total_cache_entries_after', 0)}\n")
                
                # Cache hit analysis
                f.write("\nCACHE HIT ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                for test_name, test_data in [("Baseline", baseline), ("Post-cache", post_cache)]:
                    f.write(f"\n{test_name} Game Cache Activity:\n")
                    
                    total_hits = test_data.get('total_hits_delta', 0)
                    total_misses = test_data.get('total_misses_delta', 0)
                    game_hit_rate = test_data.get('game_hit_rate', 0)
                    
                    f.write(f"  Total cache hits during game:   {total_hits:,}\n")
                    f.write(f"  Total cache misses during game: {total_misses:,}\n")
                    f.write(f"  Game cache hit rate:            {game_hit_rate:.1f}%\n")
                    
                    f.write(f"\n  Cache entries added by namespace:\n")
                    cache_entries_delta = test_data.get('cache_hits_delta', {})
                    for namespace in CacheNamespace:
                        ns_name = namespace.value
                        entries_added = cache_entries_delta.get(ns_name, 0)
                        f.write(f"    {ns_name:20s}: {entries_added:,} new entries\n")
            
            else:
                f.write("Incomplete comparison data available.\n")
                if baseline:
                    f.write(f"Baseline game: {baseline.get('turn_count', 'N/A')} turns, {baseline.get('execution_time', 0):.3f}s\n")
                if post_cache:
                    f.write(f"Post-cache game: {post_cache.get('turn_count', 'N/A')} turns, {post_cache.get('execution_time', 0):.3f}s\n")
        
        print(f"📝 MCTS comparison report saved to: {report_file}")
        return report_file
    
    def run_complete_analysis(self, num_cache_games: int = 1000):
        """Run the complete cache performance analysis."""
        print("🔬 CACHE PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"Analyzing cache impact using {num_cache_games} random games")
        
        # Step 1: Clear cache and play baseline MCTS game
        self.clear_cache_for_testing()
        baseline_result = self.play_mcts_baseline_game("baseline")
        self.mcts_comparison_data['baseline'] = baseline_result
        
        # Step 2: Populate cache with random games
        cache_population_result = self.play_cache_population_games(num_cache_games)
        
        # Step 3: Play post-cache MCTS game
        post_cache_result = self.play_mcts_baseline_game("post_cache")
        self.mcts_comparison_data['post_cache'] = post_cache_result
        
        # Step 4: Create plots
        plot_file = self.create_performance_plots()
        
        # Step 5: Save comparison report
        report_file = self.save_mcts_comparison_report()
        
        # Step 6: Save complete analysis data
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'num_cache_games': num_cache_games,
                'map_size': 'test',
                'max_turns': 50
            },
            'mcts_comparison': self.mcts_comparison_data,
            'cache_population': cache_population_result,
            'performance_data': {
                'game_length_data': self.game_length_data,
                'cache_size_data': self.cache_size_data,
                'execution_time_data': self.execution_time_data
            }
        }
        
        data_file = self.results_dir / f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(data_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        # Print summary
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Performance plots: {plot_file}")
        print(f"📝 MCTS comparison: {report_file}")
        print(f"💾 Complete data: {data_file}")
        
        return {
            'plot_file': plot_file,
            'report_file': report_file,
            'data_file': data_file,
            'analysis_data': analysis_data
        }


def main():
    """Main execution function."""
    print("🔬 Cache Performance Analysis Tool")
    print("=" * 50)
    
    # Configure analysis parameters
    num_cache_games = 20  # Number of random games to populate cache
    save_dir = "cache_analysis"
    
    # Run analysis
    analyzer = CachePerformanceAnalyzer(save_dir=save_dir)
    
    try:
        results = analyzer.run_complete_analysis(num_cache_games=num_cache_games)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"Results saved in: {save_dir}/")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
