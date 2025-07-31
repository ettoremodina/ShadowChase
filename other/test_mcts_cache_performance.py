#!/usr/bin/env python3
"""
MCTS Cache Performance Test

This script tests the specific impact of MCTS caching on agent performance.
It compares MCTS agents with MCTS cache enabled vs disabled while keeping
other cache systems (like game methods cache) active.

Features:
- Isolates MCTS cache performance impact
- Compares decision quality and thinking time
- Analyzes cache effectiveness for tree search
- Tests different game scenarios and complexity levels
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

from game_controls.game_utils import play_multiple_games
from game_controls.display_utils import VerbosityLevel
from agents import AgentType
from ScotlandYard.services.cache_system import (
    enable_cache, disable_cache, is_cache_enabled, get_global_cache,
    enable_namespace_cache, disable_namespace_cache, is_namespace_cache_enabled,
    reset_namespace_cache_settings, get_cache_status, CacheNamespace
)


class MCTSCachePerformanceTest:
    """Test MCTS cache performance impact specifically."""
    
    def __init__(self, save_dir: str = "mcts_cache_test2"):
        self.save_dir = save_dir
        self.results_dir = Path(save_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configuration - using MCTS agents
        self.num_games = 3  # Fewer games since MCTS is slower
        self.map_size = "extended"
        self.max_turns = 30  # Shorter games to focus on cache impact
        self.num_detectives = 5
        
    def run_mcts_cache_test(self) -> Dict[str, Any]:
        """Run the complete MCTS cache performance test."""
        print("🌳 MCTS CACHE PERFORMANCE TEST")
        print("=" * 60)
        print(f"Testing {self.num_games} games with MCTS cache enabled vs disabled")
        print(f"Map: {self.map_size}, Detectives: {self.num_detectives}, Max turns: {self.max_turns}")
        print("Using MCTS agents to test tree search cache effectiveness")
        
        # Ensure global cache is enabled and reset namespace settings
        enable_cache()
        reset_namespace_cache_settings()
        
        disable_namespace_cache(CacheNamespace.GAME_METHODS)
        # Get initial system stats
        initial_memory = self._get_memory_usage()
        
        # Phase 1: MCTS cache enabled test
        print(f"\n🟢 PHASE 1: Testing {self.num_games} games with MCTS cache ENABLED")
        print("   (Game methods cache remains enabled)")
        enable_namespace_cache(CacheNamespace.MCTS_NODES)
        enable_namespace_cache(CacheNamespace.AGENT_DECISIONS)
        
        # Show cache status
        self._print_cache_status()
        
        mcts_cache_enabled_results = self._run_mcts_game_batch("mcts_cache_enabled")
        
        # Get cache stats after enabled test
        cache_stats_after_enabled = get_global_cache().get_global_stats()
        memory_after_enabled = self._get_memory_usage()
        
        # Clear only MCTS cache for fair comparison
        print("🧹 Clearing MCTS cache entries for clean comparison...")
        get_global_cache().clear_namespace(CacheNamespace.MCTS_NODES)
        get_global_cache().clear_namespace(CacheNamespace.AGENT_DECISIONS)
        
        # Phase 2: MCTS cache disabled test
        print(f"\n🔴 PHASE 2: Testing {self.num_games} games with MCTS cache DISABLED")
        print("   (Game methods cache remains enabled)")
        disable_namespace_cache(CacheNamespace.MCTS_NODES)
        disable_namespace_cache(CacheNamespace.AGENT_DECISIONS)
        
        # Show cache status
        self._print_cache_status()
        
        mcts_cache_disabled_results = self._run_mcts_game_batch("mcts_cache_disabled")
        
        # Get final memory stats
        memory_after_disabled = self._get_memory_usage()
        
        # Reset cache settings to default
        reset_namespace_cache_settings()
        
        # Analyze results
        analysis_results = self._analyze_mcts_performance(
            mcts_cache_enabled_results, 
            mcts_cache_disabled_results,
            cache_stats_after_enabled,
            initial_memory,
            memory_after_enabled,
            memory_after_disabled
        )
        
        # Save comprehensive report
        self._save_mcts_detailed_report(
            mcts_cache_enabled_results, 
            mcts_cache_disabled_results, 
            analysis_results,
            cache_stats_after_enabled
        )
        
        return {
            'mcts_cache_enabled_results': mcts_cache_enabled_results,
            'mcts_cache_disabled_results': mcts_cache_disabled_results,
            'analysis': analysis_results,
            'cache_stats': cache_stats_after_enabled
        }
    
    def _print_cache_status(self):
        """Print current cache status for all namespaces."""
        status = get_cache_status()
        print(f"   Cache Status - Global: {'✅' if status['global_enabled'] else '❌'}")
        for namespace, ns_status in status['namespaces'].items():
            enabled_icon = '✅' if ns_status['enabled'] else '❌'
            explicit_note = ' (explicit)' if ns_status['explicitly_set'] else ' (default)'
            print(f"     {namespace}: {enabled_icon}{explicit_note}")
    
    def _run_mcts_game_batch(self, test_name: str) -> Dict[str, Any]:
        """Run a batch of MCTS games and collect detailed metrics."""
        print(f"   Running {self.num_games} MCTS vs MCTS games...")
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Run the games using play_multiple_games with MCTS agents
        batch_results = play_multiple_games(
            n_games=self.num_games,
            map_size=self.map_size,
            play_mode="ai_vs_ai",
            num_detectives=self.num_detectives,
            verbosity=VerbosityLevel.SILENT,  # Silent for clean output
            max_turns=self.max_turns,
            mr_x_agent_type=AgentType.OPTIMIZED_MCTS,  # Use MCTS agents
            detective_agent_type=AgentType.OPTIMIZED_MCTS,
            save_dir=self.save_dir
        )
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        total_duration = end_time - start_time
        
        # Extract detailed statistics
        completed_games = batch_results.get('completed_games', 0)
        total_turns = batch_results.get('total_turns', 0)
        avg_turns = total_turns / completed_games if completed_games > 0 else 0
        
        # MCTS-specific metrics
        avg_thinking_time = total_duration / total_turns if total_turns > 0 else 0
        
        results = {
            'test_name': test_name,
            'num_games_requested': self.num_games,
            'completed_games': completed_games,
            'incomplete_games': self.num_games - completed_games,
            'total_duration': total_duration,
            'avg_duration_per_game': total_duration / completed_games if completed_games > 0 else 0,
            'total_turns': total_turns,
            'avg_turns_per_game': avg_turns,
            'avg_thinking_time_per_turn': avg_thinking_time,
            'games_per_second': completed_games / total_duration if total_duration > 0 else 0,
            'turns_per_second': total_turns / total_duration if total_duration > 0 else 0,
            'mcts_cache_enabled': is_namespace_cache_enabled(CacheNamespace.MCTS_NODES),
            'agent_cache_enabled': is_namespace_cache_enabled(CacheNamespace.AGENT_DECISIONS),
            'game_cache_enabled': is_namespace_cache_enabled(CacheNamespace.GAME_METHODS),
            'memory_usage': {
                'start_mb': start_memory,
                'end_mb': end_memory,
                'peak_delta_mb': end_memory - start_memory
            },
            'batch_results': batch_results
        }
        
        print(f"   ✅ Completed: {completed_games}/{self.num_games} games")
        print(f"   ⏱️  Total time: {total_duration:.1f}s ({results['games_per_second']:.2f} games/sec)")
        print(f"   🎮 Average game length: {avg_turns:.1f} turns")
        print(f"   🧠 Average thinking time: {avg_thinking_time:.3f}s/turn")
        print(f"   💾 Memory usage: {start_memory:.1f} → {end_memory:.1f} MB (Δ{end_memory-start_memory:+.1f})")
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _analyze_mcts_performance(self, mcts_enabled: Dict, mcts_disabled: Dict, 
                                cache_stats: Dict, initial_memory: float,
                                memory_after_enabled: float, memory_after_disabled: float) -> Dict[str, Any]:
        """Analyze MCTS cache performance differences."""
        
        # Performance metrics
        enabled_thinking_time = mcts_enabled['avg_thinking_time_per_turn']
        disabled_thinking_time = mcts_disabled['avg_thinking_time_per_turn']
        
        enabled_games_per_sec = mcts_enabled['games_per_second']
        disabled_games_per_sec = mcts_disabled['games_per_second']
        
        enabled_game_time = mcts_enabled['avg_duration_per_game']
        disabled_game_time = mcts_disabled['avg_duration_per_game']
        
        # Calculate performance differences
        thinking_time_speedup = disabled_thinking_time / enabled_thinking_time if enabled_thinking_time > 0 else 1.0
        game_time_speedup = disabled_game_time / enabled_game_time if enabled_game_time > 0 else 1.0
        throughput_improvement = (enabled_games_per_sec / disabled_games_per_sec - 1) * 100 if disabled_games_per_sec > 0 else 0
        
        # Memory analysis
        mcts_cache_memory_overhead = memory_after_enabled - memory_after_disabled
        
        # Game quality analysis (game length consistency)
        enabled_avg_turns = mcts_enabled['avg_turns_per_game']
        disabled_avg_turns = mcts_disabled['avg_turns_per_game']
        game_length_difference = enabled_avg_turns - disabled_avg_turns
        
        # MCTS cache effectiveness (focus on MCTS and agent decision namespaces)
        mcts_cache_effectiveness = {}
        if cache_stats and cache_stats.get('namespace_stats'):
            ns_stats = cache_stats['namespace_stats']
            
            # MCTS nodes cache stats
            if 'mcts_nodes' in ns_stats:
                mcts_nodes_stats = ns_stats['mcts_nodes']
                mcts_cache_effectiveness['mcts_nodes'] = {
                    'total_entries': mcts_nodes_stats.get('total_entries', 0),
                    'avg_access_count': mcts_nodes_stats.get('avg_access_count', 0)
                }
            
            # Agent decisions cache stats
            if 'agent_decisions' in ns_stats:
                agent_decisions_stats = ns_stats['agent_decisions']
                mcts_cache_effectiveness['agent_decisions'] = {
                    'total_entries': agent_decisions_stats.get('total_entries', 0),
                    'avg_access_count': agent_decisions_stats.get('avg_access_count', 0)
                }
            
            # Global cache stats
            if cache_stats.get('global_stats'):
                global_stats = cache_stats['global_stats']
                mcts_cache_effectiveness['global'] = {
                    'hit_rate_percent': global_stats.get('hit_rate', 0) * 100,
                    'total_requests': global_stats.get('total_requests', 0),
                    'total_hits': global_stats.get('hits', 0),
                    'total_misses': global_stats.get('misses', 0),
                    'cache_size_mb': global_stats.get('cache_size_mb', 0)
                }
        
        # Generate MCTS-specific performance assessment
        if thinking_time_speedup > 1.1:  # >10% faster thinking
            performance_verdict = f"🟢 MCTS CACHE IMPROVES THINKING: {thinking_time_speedup:.2f}x faster per turn"
        elif thinking_time_speedup < 0.9:  # >10% slower thinking
            performance_verdict = f"🔴 MCTS CACHE SLOWS THINKING: {1/thinking_time_speedup:.2f}x slower per turn"
        else:
            performance_verdict = f"🟡 MCTS CACHE NEUTRAL: {((thinking_time_speedup-1)*100):+.1f}% thinking time change"
        
        return {
            'mcts_performance_summary': {
                'mcts_enabled_thinking_time': enabled_thinking_time,
                'mcts_disabled_thinking_time': disabled_thinking_time,
                'thinking_time_speedup_factor': thinking_time_speedup,
                'game_time_speedup_factor': game_time_speedup,
                'throughput_improvement_percent': throughput_improvement,
                'verdict': performance_verdict
            },
            'memory_analysis': {
                'initial_memory_mb': initial_memory,
                'memory_after_mcts_enabled_mb': memory_after_enabled,
                'memory_after_mcts_disabled_mb': memory_after_disabled,
                'mcts_cache_memory_overhead_mb': mcts_cache_memory_overhead,
                'memory_overhead_verdict': f"MCTS cache adds {mcts_cache_memory_overhead:+.1f} MB memory overhead"
            },
            'game_quality_analysis': {
                'mcts_enabled_avg_turns': enabled_avg_turns,
                'mcts_disabled_avg_turns': disabled_avg_turns,
                'game_length_difference': game_length_difference,
                'quality_verdict': "Game quality consistent" if abs(game_length_difference) < 2 else f"MCTS cache changes game length by {game_length_difference:+.1f} turns"
            },
            'mcts_cache_effectiveness': mcts_cache_effectiveness,
            'raw_metrics': {
                'mcts_enabled': mcts_enabled,
                'mcts_disabled': mcts_disabled
            }
        }
    
    def _save_mcts_detailed_report(self, mcts_enabled: Dict, mcts_disabled: Dict, 
                                 analysis: Dict, cache_stats: Dict):
        """Save a detailed MCTS cache performance report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"mcts_cache_test_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MCTS CACHE PERFORMANCE TEST REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test configuration
            f.write("TEST CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Games per test:           {self.num_games}\n")
            f.write(f"Map size:                 {self.map_size}\n")
            f.write(f"Number of detectives:     {self.num_detectives}\n")
            f.write(f"Maximum turns per game:   {self.max_turns}\n")
            f.write(f"Agent types:              MCTS vs MCTS\n")
            f.write(f"Test focus:               MCTS cache namespace isolation\n\n")
            
            # MCTS performance summary
            perf = analysis['mcts_performance_summary']
            f.write("MCTS PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"MCTS cache ENABLED thinking time/turn:  {perf['mcts_enabled_thinking_time']:.4f}s\n")
            f.write(f"MCTS cache DISABLED thinking time/turn: {perf['mcts_disabled_thinking_time']:.4f}s\n")
            f.write(f"Thinking time speedup factor:           {perf['thinking_time_speedup_factor']:.3f}x\n")
            f.write(f"Game time speedup factor:               {perf['game_time_speedup_factor']:.3f}x\n")
            f.write(f"Throughput improvement:                 {perf['throughput_improvement_percent']:+.2f}%\n")
            f.write(f"Verdict:                                {perf['verdict']}\n\n")
            
            # Memory analysis
            mem = analysis['memory_analysis']
            f.write("MEMORY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Initial memory usage:                   {mem['initial_memory_mb']:.1f} MB\n")
            f.write(f"Memory after MCTS cache enabled test:   {mem['memory_after_mcts_enabled_mb']:.1f} MB\n")
            f.write(f"Memory after MCTS cache disabled test:  {mem['memory_after_mcts_disabled_mb']:.1f} MB\n")
            f.write(f"MCTS cache memory overhead:             {mem['mcts_cache_memory_overhead_mb']:+.1f} MB\n")
            f.write(f"Memory verdict:                         {mem['memory_overhead_verdict']}\n\n")
            
            # Game quality analysis
            quality = analysis['game_quality_analysis']
            f.write("GAME QUALITY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"MCTS cache enabled avg turns:           {quality['mcts_enabled_avg_turns']:.2f}\n")
            f.write(f"MCTS cache disabled avg turns:          {quality['mcts_disabled_avg_turns']:.2f}\n")
            f.write(f"Game length difference:                 {quality['game_length_difference']:+.2f} turns\n")
            f.write(f"Quality verdict:                        {quality['quality_verdict']}\n\n")
            
            # MCTS cache effectiveness
            if analysis['mcts_cache_effectiveness']:
                cache_eff = analysis['mcts_cache_effectiveness']
                f.write("MCTS CACHE EFFECTIVENESS:\n")
                f.write("-" * 40 + "\n")
                
                if 'global' in cache_eff:
                    global_eff = cache_eff['global']
                    f.write(f"Overall hit rate:                       {global_eff['hit_rate_percent']:.2f}%\n")
                    f.write(f"Total cache requests:                   {global_eff['total_requests']:,}\n")
                    f.write(f"Cache hits:                             {global_eff['total_hits']:,}\n")
                    f.write(f"Cache misses:                           {global_eff['total_misses']:,}\n")
                    f.write(f"Total cache size:                       {global_eff['cache_size_mb']:.1f} MB\n\n")
                
                if 'mcts_nodes' in cache_eff:
                    mcts_eff = cache_eff['mcts_nodes']
                    f.write(f"MCTS nodes cache entries:               {mcts_eff['total_entries']:,}\n")
                    f.write(f"MCTS nodes avg access count:            {mcts_eff['avg_access_count']:.2f}\n")
                
                if 'agent_decisions' in cache_eff:
                    agent_eff = cache_eff['agent_decisions']
                    f.write(f"Agent decisions cache entries:          {agent_eff['total_entries']:,}\n")
                    f.write(f"Agent decisions avg access count:       {agent_eff['avg_access_count']:.2f}\n\n")
            
            # Detailed results
            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write("MCTS cache ENABLED test:\n")
            f.write(f"  Completed games:      {mcts_enabled['completed_games']}/{mcts_enabled['num_games_requested']}\n")
            f.write(f"  Total duration:       {mcts_enabled['total_duration']:.1f}s\n")
            f.write(f"  Games per second:     {mcts_enabled['games_per_second']:.2f}\n")
            f.write(f"  Thinking time/turn:   {mcts_enabled['avg_thinking_time_per_turn']:.3f}s\n")
            f.write(f"  Memory delta:         {mcts_enabled['memory_usage']['peak_delta_mb']:+.1f} MB\n\n")
            
            f.write("MCTS cache DISABLED test:\n")
            f.write(f"  Completed games:      {mcts_disabled['completed_games']}/{mcts_disabled['num_games_requested']}\n")
            f.write(f"  Total duration:       {mcts_disabled['total_duration']:.1f}s\n")
            f.write(f"  Games per second:     {mcts_disabled['games_per_second']:.2f}\n")
            f.write(f"  Thinking time/turn:   {mcts_disabled['avg_thinking_time_per_turn']:.3f}s\n")
            f.write(f"  Memory delta:         {mcts_disabled['memory_usage']['peak_delta_mb']:+.1f} MB\n\n")
            
            # Recommendations
            f.write("MCTS CACHE RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            thinking_speedup = perf['thinking_time_speedup_factor']
            if thinking_speedup > 1.05:
                f.write("✅ ENABLE MCTS CACHE - Improves thinking speed significantly\n")
            elif thinking_speedup < 0.95:
                f.write("❌ DISABLE MCTS CACHE - Slows down thinking performance\n")
            else:
                f.write("🤷 MCTS CACHE NEUTRAL - No significant thinking time impact\n")
            
            if mem['mcts_cache_memory_overhead_mb'] > 20:
                f.write("⚠️  MODERATE MEMORY OVERHEAD - Consider memory limits\n")
            elif mem['mcts_cache_memory_overhead_mb'] < 5:
                f.write("✅ LOW MEMORY OVERHEAD - Acceptable memory cost\n")
        
        print(f"📝 MCTS cache report saved to: {report_file}")
        
        # Create visualization
        self._create_mcts_comparison_plots(mcts_enabled, mcts_disabled, analysis, timestamp)
        
        # Save JSON data for further analysis
        json_file = self.results_dir / f"mcts_cache_test_data_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'mcts_enabled': mcts_enabled,
                'mcts_disabled': mcts_disabled,
                'analysis': analysis,
                'cache_stats': cache_stats
            }, f, indent=2, default=str)
        
        print(f"💾 MCTS cache raw data saved to: {json_file}")
        
        return report_file
    
    def _create_mcts_comparison_plots(self, mcts_enabled: Dict, mcts_disabled: Dict, 
                                    analysis: Dict, timestamp: str):
        """Create MCTS-specific performance comparison plots."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MCTS Cache Performance Comparison\n(Focus on Tree Search Caching)', fontsize=16)
        
        # Plot 1: Thinking time comparison
        categories = ['MCTS Cache\nEnabled', 'MCTS Cache\nDisabled']
        thinking_times = [
            mcts_enabled['avg_thinking_time_per_turn'],
            mcts_disabled['avg_thinking_time_per_turn']
        ]
        colors = ['green', 'red']
        
        bars = ax1.bar(categories, thinking_times, color=colors, alpha=0.7)
        ax1.set_title('Average Thinking Time per Turn')
        ax1.set_ylabel('Thinking Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, thinking_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}s', ha='center', va='bottom')
        
        # Plot 2: Game performance metrics
        metrics = ['Games/sec', 'Avg Game Time']
        mcts_enabled_vals = [
            mcts_enabled['games_per_second'],
            mcts_enabled['avg_duration_per_game']
        ]
        mcts_disabled_vals = [
            mcts_disabled['games_per_second'],
            mcts_disabled['avg_duration_per_game']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, mcts_enabled_vals, width, label='MCTS Cache Enabled', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, mcts_disabled_vals, width, label='MCTS Cache Disabled', color='red', alpha=0.7)
        
        ax2.set_title('Game Performance Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 3: Memory usage comparison
        memory_categories = ['MCTS Cache\nEnabled Peak', 'MCTS Cache\nDisabled Peak', 'Cache\nOverhead']
        memory_values = [
            mcts_enabled['memory_usage']['end_mb'],
            mcts_disabled['memory_usage']['end_mb'],
            analysis['memory_analysis']['mcts_cache_memory_overhead_mb']
        ]
        colors = ['green', 'red', 'orange']
        
        bars = ax3.bar(memory_categories, memory_values, color=colors, alpha=0.7)
        ax3.set_title('Memory Usage Comparison')
        ax3.set_ylabel('Memory (MB)')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, memory_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 4: Performance speedup summary
        thinking_speedup = analysis['mcts_performance_summary']['thinking_time_speedup_factor']
        game_speedup = analysis['mcts_performance_summary']['game_time_speedup_factor']
        throughput_improvement = analysis['mcts_performance_summary']['throughput_improvement_percent']
        
        summary_metrics = ['Thinking Time\nSpeedup', 'Game Time\nSpeedup', 'Throughput\nImprovement (%)']
        summary_values = [thinking_speedup, game_speedup, throughput_improvement]
        colors = ['green' if v > 1 else 'red' for v in [thinking_speedup, game_speedup]] + ['green' if throughput_improvement > 0 else 'red']
        
        bars = ax4.bar(summary_metrics, summary_values, color=colors, alpha=0.7)
        ax4.set_title('MCTS Cache Performance Summary')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, summary_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"mcts_cache_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 MCTS cache plots saved to: {plot_file}")
        
        return plot_file


def main():
    """Main execution function."""
    print("🌳 MCTS Cache Performance Test")
    print("=" * 60)
    print("This test isolates MCTS cache performance by enabling/disabling only")
    print("MCTS-related cache namespaces while keeping game method cache active.")
    
    # Create and run test
    test = MCTSCachePerformanceTest()
    
    try:
        results = test.run_mcts_cache_test()
        
        # Print summary
        analysis = results['analysis']
        perf = analysis['mcts_performance_summary']
        
        print(f"\n🎉 MCTS CACHE TEST COMPLETED!")
        print(f"📊 {perf['verdict']}")
        print(f"⚡ Thinking time speedup: {perf['thinking_time_speedup_factor']:.3f}x")
        print(f"🎮 Game time speedup: {perf['game_time_speedup_factor']:.3f}x")
        print(f"📈 Throughput: {perf['throughput_improvement_percent']:+.2f}%")
        print(f"💾 Memory overhead: {analysis['memory_analysis']['mcts_cache_memory_overhead_mb']:+.1f} MB")
        
        return results
        
    except Exception as e:
        print(f"❌ MCTS cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
