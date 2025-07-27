#!/usr/bin/env python3
"""
Random Agent Cache Performance Test

This script compares 100 random AI games with cache enabled vs 100 games with cache disabled
to measure the pure overhead/benefit of the caching system without complex AI computations.

Using random agents allows us to focus on:
- Cache system overhead
- Memory usage impact
- Basic game method caching effectiveness
- Overall system performance

Results will show if the cache helps or hurts performance for basic gameplay.
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
import psutil

# Add project root to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from simple_play.game_utils import play_multiple_games
from simple_play.display_utils import VerbosityLevel
from agents import AgentType
from cache_system import enable_cache, disable_cache, is_cache_enabled, get_global_cache


class RandomAgentCacheTest:
    """Test cache performance with random agents."""
    
    def __init__(self, save_dir: str = "random_cache_test"):
        self.save_dir = save_dir
        self.results_dir = Path(save_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.num_games = 100
        self.map_size = "test"
        self.max_turns = 50
        self.num_detectives = 2
        
    def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete cache performance test."""
        print("🎲 RANDOM AGENT CACHE PERFORMANCE TEST")
        print("=" * 60)
        print(f"Testing {self.num_games} games with each cache setting")
        print(f"Map: {self.map_size}, Detectives: {self.num_detectives}, Max turns: {self.max_turns}")
        print("Using random agents to isolate cache performance impact")
        
        # Get initial system stats
        initial_memory = self._get_memory_usage()
        
        # Phase 1: Cache enabled test
        print(f"\n🟢 PHASE 1: Testing {self.num_games} games with cache ENABLED")
        enable_cache()
        cache_enabled_results = self._run_random_game_batch("cache_enabled")
        
        # Get cache stats after enabled test
        cache_stats_after_enabled = get_global_cache().get_global_stats()
        memory_after_enabled = self._get_memory_usage()
        
        # Clear cache and reset for fair comparison
        print("🧹 Clearing cache for clean comparison...")
        get_global_cache().clear_all()
        
        # Phase 2: Cache disabled test
        print(f"\n🔴 PHASE 2: Testing {self.num_games} games with cache DISABLED")
        disable_cache()
        cache_disabled_results = self._run_random_game_batch("cache_disabled")
        
        # Get final memory stats
        memory_after_disabled = self._get_memory_usage()
        
        # Re-enable cache for normal operation
        enable_cache()
        
        # Analyze results
        analysis_results = self._analyze_performance(
            cache_enabled_results, 
            cache_disabled_results,
            cache_stats_after_enabled,
            initial_memory,
            memory_after_enabled,
            memory_after_disabled
        )
        
        # Save comprehensive report
        self._save_detailed_report(
            cache_enabled_results, 
            cache_disabled_results, 
            analysis_results,
            cache_stats_after_enabled
        )
        
        return {
            'cache_enabled_results': cache_enabled_results,
            'cache_disabled_results': cache_disabled_results,
            'analysis': analysis_results,
            'cache_stats': cache_stats_after_enabled
        }
    
    def _run_random_game_batch(self, test_name: str) -> Dict[str, Any]:
        """Run a batch of random games and collect detailed metrics."""
        print(f"   Running {self.num_games} random vs random games...")
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Run the games using play_multiple_games
        batch_results = play_multiple_games(
            n_games=self.num_games,
            map_size=self.map_size,
            play_mode="ai_vs_ai",
            num_detectives=self.num_detectives,
            verbosity=VerbosityLevel.SILENT,  # Silent for clean output
            max_turns=self.max_turns,
            mr_x_agent_type=AgentType.RANDOM,
            detective_agent_type=AgentType.RANDOM,
            save_dir=self.save_dir
        )
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        total_duration = end_time - start_time
        
        # Extract detailed statistics
        completed_games = batch_results.get('completed_games', 0)
        total_turns = batch_results.get('total_turns', 0)
        avg_turns = total_turns / completed_games if completed_games > 0 else 0
        
        results = {
            'test_name': test_name,
            'num_games_requested': self.num_games,
            'completed_games': completed_games,
            'incomplete_games': self.num_games - completed_games,
            'total_duration': total_duration,
            'avg_duration_per_game': total_duration / completed_games if completed_games > 0 else 0,
            'total_turns': total_turns,
            'avg_turns_per_game': avg_turns,
            'games_per_second': completed_games / total_duration if total_duration > 0 else 0,
            'turns_per_second': total_turns / total_duration if total_duration > 0 else 0,
            'cache_enabled': is_cache_enabled(),
            'memory_usage': {
                'start_mb': start_memory,
                'end_mb': end_memory,
                'peak_delta_mb': end_memory - start_memory
            },
            'batch_results': batch_results
        }
        
        print(f"   ✅ Completed: {completed_games}/{self.num_games} games")
        print(f"   ⏱️  Total time: {total_duration:.2f}s ({results['games_per_second']:.2f} games/sec)")
        print(f"   🎮 Average game length: {avg_turns:.1f} turns")
        print(f"   💾 Memory usage: {start_memory:.1f} → {end_memory:.1f} MB (Δ{end_memory-start_memory:+.1f})")
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _analyze_performance(self, cache_enabled: Dict, cache_disabled: Dict, 
                           cache_stats: Dict, initial_memory: float,
                           memory_after_enabled: float, memory_after_disabled: float) -> Dict[str, Any]:
        """Analyze performance differences between cache enabled and disabled."""
        
        # Performance metrics
        enabled_time = cache_enabled['avg_duration_per_game']
        disabled_time = cache_disabled['avg_duration_per_game']
        
        enabled_games_per_sec = cache_enabled['games_per_second']
        disabled_games_per_sec = cache_disabled['games_per_second']
        
        enabled_turns_per_sec = cache_enabled['turns_per_second']
        disabled_turns_per_sec = cache_disabled['turns_per_second']
        
        # Calculate performance differences
        time_speedup = disabled_time / enabled_time if enabled_time > 0 else 1.0
        throughput_improvement = (enabled_games_per_sec / disabled_games_per_sec - 1) * 100 if disabled_games_per_sec > 0 else 0
        turns_processing_improvement = (enabled_turns_per_sec / disabled_turns_per_sec - 1) * 100 if disabled_turns_per_sec > 0 else 0
        
        # Memory analysis
        cache_memory_overhead = memory_after_enabled - memory_after_disabled
        
        # Game length consistency
        enabled_avg_turns = cache_enabled['avg_turns_per_game']
        disabled_avg_turns = cache_disabled['avg_turns_per_game']
        game_length_difference = enabled_avg_turns - disabled_avg_turns
        
        # Cache effectiveness (if cache was used)
        cache_effectiveness = {}
        if cache_stats and cache_stats.get('global_stats'):
            global_stats = cache_stats['global_stats']
            cache_effectiveness = {
                'hit_rate_percent': global_stats.get('hit_rate', 0) * 100,
                'total_requests': global_stats.get('total_requests', 0),
                'total_hits': global_stats.get('hits', 0),
                'total_misses': global_stats.get('misses', 0),
                'cache_size_mb': global_stats.get('cache_size_mb', 0),
                'total_entries': global_stats.get('total_entries', 0),
                'writes': global_stats.get('writes', 0),
                'evictions': global_stats.get('evictions', 0)
            }
        
        # Generate performance assessment
        if time_speedup > 1.1:  # >10% faster
            performance_verdict = f"🟢 CACHE IMPROVES PERFORMANCE: {time_speedup:.2f}x speedup"
        elif time_speedup < 0.9:  # >10% slower
            performance_verdict = f"🔴 CACHE DEGRADES PERFORMANCE: {1/time_speedup:.2f}x slowdown"
        else:
            performance_verdict = f"🟡 CACHE NEUTRAL: {((time_speedup-1)*100):+.1f}% performance change"
        
        return {
            'performance_summary': {
                'cache_enabled_avg_time': enabled_time,
                'cache_disabled_avg_time': disabled_time,
                'time_speedup_factor': time_speedup,
                'throughput_improvement_percent': throughput_improvement,
                'turns_processing_improvement_percent': turns_processing_improvement,
                'verdict': performance_verdict
            },
            'memory_analysis': {
                'initial_memory_mb': initial_memory,
                'memory_after_cache_enabled_mb': memory_after_enabled,
                'memory_after_cache_disabled_mb': memory_after_disabled,
                'cache_memory_overhead_mb': cache_memory_overhead,
                'memory_overhead_verdict': f"Cache adds {cache_memory_overhead:+.1f} MB memory overhead"
            },
            'game_consistency': {
                'cache_enabled_avg_turns': enabled_avg_turns,
                'cache_disabled_avg_turns': disabled_avg_turns,
                'game_length_difference': game_length_difference,
                'consistency_verdict': "Games consistent length" if abs(game_length_difference) < 1 else f"Cache changes game length by {game_length_difference:+.1f} turns"
            },
            'cache_effectiveness': cache_effectiveness,
            'raw_metrics': {
                'cache_enabled': cache_enabled,
                'cache_disabled': cache_disabled
            }
        }
    
    def _save_detailed_report(self, cache_enabled: Dict, cache_disabled: Dict, 
                            analysis: Dict, cache_stats: Dict):
        """Save a detailed performance report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"random_agent_cache_test_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RANDOM AGENT CACHE PERFORMANCE TEST REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test configuration
            f.write("TEST CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Games per test:           {self.num_games}\n")
            f.write(f"Map size:                 {self.map_size}\n")
            f.write(f"Number of detectives:     {self.num_detectives}\n")
            f.write(f"Maximum turns per game:   {self.max_turns}\n")
            f.write(f"Agent types:              Random vs Random\n\n")
            
            # Performance summary
            perf = analysis['performance_summary']
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cache ENABLED avg time per game:    {perf['cache_enabled_avg_time']:.4f}s\n")
            f.write(f"Cache DISABLED avg time per game:   {perf['cache_disabled_avg_time']:.4f}s\n")
            f.write(f"Speedup factor:                     {perf['time_speedup_factor']:.3f}x\n")
            f.write(f"Throughput improvement:             {perf['throughput_improvement_percent']:+.2f}%\n")
            f.write(f"Turn processing improvement:        {perf['turns_processing_improvement_percent']:+.2f}%\n")
            f.write(f"Verdict:                            {perf['verdict']}\n\n")
            
            # Memory analysis
            mem = analysis['memory_analysis']
            f.write("MEMORY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Initial memory usage:               {mem['initial_memory_mb']:.1f} MB\n")
            f.write(f"Memory after cache enabled test:    {mem['memory_after_cache_enabled_mb']:.1f} MB\n")
            f.write(f"Memory after cache disabled test:   {mem['memory_after_cache_disabled_mb']:.1f} MB\n")
            f.write(f"Cache memory overhead:              {mem['cache_memory_overhead_mb']:+.1f} MB\n")
            f.write(f"Memory verdict:                     {mem['memory_overhead_verdict']}\n\n")
            
            # Game consistency
            game = analysis['game_consistency']
            f.write("GAME CONSISTENCY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cache enabled avg turns:            {game['cache_enabled_avg_turns']:.2f}\n")
            f.write(f"Cache disabled avg turns:           {game['cache_disabled_avg_turns']:.2f}\n")
            f.write(f"Game length difference:             {game['game_length_difference']:+.2f} turns\n")
            f.write(f"Consistency verdict:                {game['consistency_verdict']}\n\n")
            
            # Cache effectiveness
            if analysis['cache_effectiveness']:
                cache_eff = analysis['cache_effectiveness']
                f.write("CACHE EFFECTIVENESS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Hit rate:                           {cache_eff['hit_rate_percent']:.2f}%\n")
                f.write(f"Total cache requests:               {cache_eff['total_requests']:,}\n")
                f.write(f"Cache hits:                         {cache_eff['total_hits']:,}\n")
                f.write(f"Cache misses:                       {cache_eff['total_misses']:,}\n")
                f.write(f"Cache entries created:              {cache_eff['writes']:,}\n")
                f.write(f"Cache evictions:                    {cache_eff['evictions']:,}\n")
                f.write(f"Final cache size:                   {cache_eff['cache_size_mb']:.1f} MB\n")
                f.write(f"Total cache entries:                {cache_eff['total_entries']:,}\n\n")
            
            # Detailed results
            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write("Cache ENABLED test:\n")
            f.write(f"  Completed games:      {cache_enabled['completed_games']}/{cache_enabled['num_games_requested']}\n")
            f.write(f"  Total duration:       {cache_enabled['total_duration']:.2f}s\n")
            f.write(f"  Games per second:     {cache_enabled['games_per_second']:.2f}\n")
            f.write(f"  Turns per second:     {cache_enabled['turns_per_second']:.1f}\n")
            f.write(f"  Memory delta:         {cache_enabled['memory_usage']['peak_delta_mb']:+.1f} MB\n\n")
            
            f.write("Cache DISABLED test:\n")
            f.write(f"  Completed games:      {cache_disabled['completed_games']}/{cache_disabled['num_games_requested']}\n")
            f.write(f"  Total duration:       {cache_disabled['total_duration']:.2f}s\n")
            f.write(f"  Games per second:     {cache_disabled['games_per_second']:.2f}\n")
            f.write(f"  Turns per second:     {cache_disabled['turns_per_second']:.1f}\n")
            f.write(f"  Memory delta:         {cache_disabled['memory_usage']['peak_delta_mb']:+.1f} MB\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            speedup = perf['time_speedup_factor']
            if speedup > 1.05:
                f.write("✅ ENABLE CACHE - Provides measurable performance improvement\n")
            elif speedup < 0.95:
                f.write("❌ DISABLE CACHE - Causes performance degradation\n")
            else:
                f.write("🤷 CACHE NEUTRAL - No significant performance impact\n")
            
            if mem['cache_memory_overhead_mb'] > 50:
                f.write("⚠️  HIGH MEMORY OVERHEAD - Monitor memory usage in production\n")
            elif mem['cache_memory_overhead_mb'] < 5:
                f.write("✅ LOW MEMORY OVERHEAD - Acceptable memory cost\n")
        
        print(f"📝 Detailed report saved to: {report_file}")
        
        # Create visualization
        self._create_comparison_plots(cache_enabled, cache_disabled, analysis, timestamp)
        
        # Save JSON data for further analysis
        json_file = self.results_dir / f"random_agent_cache_test_data_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'cache_enabled': cache_enabled,
                'cache_disabled': cache_disabled,
                'analysis': analysis,
                'cache_stats': cache_stats
            }, f, indent=2, default=str)
        
        print(f"💾 Raw data saved to: {json_file}")
        
        return report_file
    
    def _create_comparison_plots(self, cache_enabled: Dict, cache_disabled: Dict, 
                               analysis: Dict, timestamp: str):
        """Create visualization plots for the performance comparison."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random Agent Cache Performance Comparison\n(100 games each)', fontsize=16)
        
        # Plot 1: Performance metrics comparison
        metrics = ['Games/sec', 'Turns/sec', 'Avg Time/Game']
        cache_enabled_vals = [
            cache_enabled['games_per_second'],
            cache_enabled['turns_per_second'],
            cache_enabled['avg_duration_per_game']
        ]
        cache_disabled_vals = [
            cache_disabled['games_per_second'],
            cache_disabled['turns_per_second'],
            cache_disabled['avg_duration_per_game']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cache_enabled_vals, width, label='Cache Enabled', color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, cache_disabled_vals, width, label='Cache Disabled', color='red', alpha=0.7)
        
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 2: Memory usage comparison
        memory_categories = ['Cache Enabled Peak', 'Cache Disabled Peak', 'Cache Overhead']
        memory_values = [
            cache_enabled['memory_usage']['end_mb'],
            cache_disabled['memory_usage']['end_mb'],
            analysis['memory_analysis']['cache_memory_overhead_mb']
        ]
        colors = ['green', 'red', 'orange']
        
        bars = ax2.bar(memory_categories, memory_values, color=colors, alpha=0.7)
        ax2.set_title('Memory Usage Comparison')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, memory_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 3: Game completion comparison
        completion_data = [
            ['Cache Enabled', cache_enabled['completed_games'], cache_enabled['incomplete_games']],
            ['Cache Disabled', cache_disabled['completed_games'], cache_disabled['incomplete_games']]
        ]
        
        categories = [data[0] for data in completion_data]
        completed = [data[1] for data in completion_data]
        incomplete = [data[2] for data in completion_data]
        
        ax3.bar(categories, completed, label='Completed', color='green', alpha=0.7)
        ax3.bar(categories, incomplete, bottom=completed, label='Incomplete', color='red', alpha=0.7)
        ax3.set_title('Game Completion Rates')
        ax3.set_ylabel('Number of Games')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        speedup = analysis['performance_summary']['time_speedup_factor']
        throughput_improvement = analysis['performance_summary']['throughput_improvement_percent']
        
        summary_metrics = ['Speedup Factor', 'Throughput\nImprovement (%)']
        summary_values = [speedup, throughput_improvement]
        colors = ['green' if v > 0 else 'red' for v in [speedup - 1, throughput_improvement]]
        
        bars = ax4.bar(summary_metrics, summary_values, color=colors, alpha=0.7)
        ax4.set_title('Performance Summary')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, summary_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"random_agent_cache_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to: {plot_file}")
        
        return plot_file


def main():
    """Main execution function."""
    print("🎲 Random Agent Cache Performance Test")
    print("=" * 60)
    print("This test compares 100 random vs random games with cache enabled vs disabled")
    print("to measure pure cache system performance impact.")
    
    # Create and run test
    test = RandomAgentCacheTest()
    
    try:
        results = test.run_complete_test()
        
        # Print summary
        analysis = results['analysis']
        perf = analysis['performance_summary']
        
        print(f"\n🎉 TEST COMPLETED!")
        print(f"📊 {perf['verdict']}")
        print(f"⚡ Speedup: {perf['time_speedup_factor']:.3f}x")
        print(f"📈 Throughput: {perf['throughput_improvement_percent']:+.2f}%")
        print(f"💾 Memory overhead: {analysis['memory_analysis']['cache_memory_overhead_mb']:+.1f} MB")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
