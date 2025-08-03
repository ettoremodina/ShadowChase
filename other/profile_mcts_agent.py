#!/usr/bin/env python3
"""
MCTS Agent Performance Profiler

This script profiles the MCTS agent to identify computational bottlenecks
and determine where optimizations like caching and lookup tables would be most beneficial.
"""

import cProfile
import pstats
import io
import time
import json
import tracemalloc
import gc
from functools import wraps
from collections import defaultdict, Counter
from typing import Dict, List, Any, Callable
import sys
import os
from pathlib import Path


# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ShadowChase.core.game import Player
# from agents.mcts_agent import MCTSMrXAgent, MCTSMultiDetectiveAgent
# from agents.mcts_agent import MCTSAgent, MCTSNode


from agents.optimized_mcts_agent import OptimizedMCTSAgent as MCTSAgent
from agents.optimized_mcts_agent import OptimizedMCTSNode as MCTSNode
from agents.optimized_mcts_agent import OptimizedMCTSMrXAgent as MCTSMrXAgent
from agents.optimized_mcts_agent import OptimizedMCTSMultiDetectiveAgent as MCTSMultiDetectiveAgent


from ShadowChase.examples.example_games import create_extracted_board_game

class FunctionProfiler:
    """Custom profiler to track function calls, execution times, and memory usage."""
    
    def __init__(self):
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        self.memory_usage = defaultdict(list)
        self.call_stacks = []
        self.active_calls = {}
        
    def profile_function(self, func_name: str):
        """Decorator to profile a specific function."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start timing and memory tracking
                start_time = time.perf_counter()
                start_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
                
                self.call_counts[func_name] += 1
                self.active_calls[func_name] = start_time
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # End timing and memory tracking
                    end_time = time.perf_counter()
                    end_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self.total_times[func_name] += execution_time
                    self.memory_usage[func_name].append(memory_delta)
                    
                    if func_name in self.active_calls:
                        del self.active_calls[func_name]
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        stats = {}
        for func_name in self.call_counts:
            avg_time = self.total_times[func_name] / self.call_counts[func_name]
            avg_memory = sum(self.memory_usage[func_name]) / len(self.memory_usage[func_name]) if self.memory_usage[func_name] else 0
            max_memory = max(self.memory_usage[func_name]) if self.memory_usage[func_name] else 0
            
            stats[func_name] = {
                'call_count': self.call_counts[func_name],
                'total_time': self.total_times[func_name],
                'avg_time': avg_time,
                'avg_memory_delta': avg_memory,
                'max_memory_delta': max_memory,
                'total_memory': sum(self.memory_usage[func_name])
            }
        
        return stats


class GameStateAnalyzer:
    """Analyzer to track game state operations and identify caching opportunities."""
    
    def __init__(self):
        self.state_hashes = Counter()
        self.move_generations = Counter()
        self.copy_operations = 0
        self.simulation_depths = []
        self.repeated_states = set()
        
    def track_state_hash(self, state_hash: str):
        """Track how often we see the same game state."""
        self.state_hashes[state_hash] += 1
        if self.state_hashes[state_hash] > 1:
            self.repeated_states.add(state_hash)
    
    def track_move_generation(self, player: Player, position: int):
        """Track move generation patterns."""
        key = f"{player}_{position}"
        self.move_generations[key] += 1
    
    def track_copy_operation(self):
        """Track deep copy operations."""
        self.copy_operations += 1
    
    def track_simulation_depth(self, depth: int):
        """Track simulation depths."""
        self.simulation_depths.append(depth)
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get caching opportunity analysis."""
        return {
            'unique_states': len(self.state_hashes),
            'repeated_states': len(self.repeated_states),
            'state_repetition_rate': len(self.repeated_states) / len(self.state_hashes) if self.state_hashes else 0,
            'most_common_states': dict(self.state_hashes.most_common(10)),
            'unique_move_generations': len(self.move_generations),
            'most_common_move_generations': dict(self.move_generations.most_common(10)),
            'total_copy_operations': self.copy_operations,
            'avg_simulation_depth': sum(self.simulation_depths) / len(self.simulation_depths) if self.simulation_depths else 0,
            'max_simulation_depth': max(self.simulation_depths) if self.simulation_depths else 0,
            'simulation_count': len(self.simulation_depths)
        }


# Global profiler instances
function_profiler = FunctionProfiler()
game_analyzer = GameStateAnalyzer()


# Monkey patch MCTS classes to add profiling
def patch_mcts_for_profiling():
    """Add profiling decorators to MCTS methods."""

   
    
    # Patch MCTSNode methods
    original_init = MCTSNode.__init__
    original_simulate = MCTSNode.simulate
    original_expand = MCTSNode.expand
    original_select_child = MCTSNode.select_child
    original_backpropagate = MCTSNode.backpropagate
    original_initialize_moves = MCTSNode._initialize_untried_moves
    
    @function_profiler.profile_function("MCTSNode.__init__")
    def profiled_init(self, *args, **kwargs):
        game_analyzer.track_copy_operation()  # Deep copy happens here
        return original_init(self, *args, **kwargs)
    
    @function_profiler.profile_function("MCTSNode.simulate")
    def profiled_simulate(self, *args, **kwargs):
        result = original_simulate(self, *args, **kwargs)
        # Track simulation depth by counting game state operations
        return result
    
    @function_profiler.profile_function("MCTSNode.expand")
    def profiled_expand(self, *args, **kwargs):
        game_analyzer.track_copy_operation()  # Another deep copy
        return original_expand(self, *args, **kwargs)
    
    @function_profiler.profile_function("MCTSNode.select_child")
    def profiled_select_child(self, *args, **kwargs):
        return original_select_child(self, *args, **kwargs)
    
    @function_profiler.profile_function("MCTSNode.backpropagate")
    def profiled_backpropagate(self, *args, **kwargs):
        return original_backpropagate(self, *args, **kwargs)
    
    @function_profiler.profile_function("MCTSNode._initialize_untried_moves")
    def profiled_initialize_moves(self, *args, **kwargs):
        return original_initialize_moves(self, *args, **kwargs)
    
    # Apply patches
    MCTSNode.__init__ = profiled_init
    MCTSNode.simulate = profiled_simulate
    MCTSNode.expand = profiled_expand
    MCTSNode.select_child = profiled_select_child
    MCTSNode.backpropagate = profiled_backpropagate
    MCTSNode._initialize_untried_moves = profiled_initialize_moves
    
    # Patch MCTSAgent methods
    original_mcts_search = MCTSAgent.mcts_search
    
    @function_profiler.profile_function("MCTSAgent.mcts_search")
    def profiled_mcts_search(self, *args, **kwargs):
        return original_mcts_search(self, *args, **kwargs)
    
    MCTSAgent.mcts_search = profiled_mcts_search
    
    # Patch game state operations
    from ShadowChase.core.game import ShadowChaseGame
    original_get_valid_moves = ShadowChaseGame.get_valid_moves
    original_make_move = ShadowChaseGame.make_move
    original_is_game_over = ShadowChaseGame.is_game_over
    
    @function_profiler.profile_function("ShadowChaseGame.get_valid_moves")
    def profiled_get_valid_moves(self, player, position=None, *args, **kwargs):
        if position is not None:
            game_analyzer.track_move_generation(player, position)
        return original_get_valid_moves(self, player, position, *args, **kwargs)
    
    @function_profiler.profile_function("ShadowChaseGame.make_move")
    def profiled_make_move(self, *args, **kwargs):
        return original_make_move(self, *args, **kwargs)
    
    @function_profiler.profile_function("ShadowChaseGame.is_game_over")
    def profiled_is_game_over(self, *args, **kwargs):
        return original_is_game_over(self, *args, **kwargs)
    
    ShadowChaseGame.get_valid_moves = profiled_get_valid_moves
    ShadowChaseGame.make_move = profiled_make_move
    ShadowChaseGame.is_game_over = profiled_is_game_over


def run_mcts_profiling_session(num_moves: int = 20, map_size: str = "extended") -> Dict[str, Any]:
    """Run a profiling session with MCTS agents."""
    print(f"Starting MCTS profiling session with {num_moves} moves...")
    
    # Start memory tracking
    tracemalloc.start()
    game = create_extracted_board_game(5)
    
    # Create agents with reduced parameters for profiling
    mr_x_agent = MCTSMrXAgent()
    
    detective_agent = MCTSMultiDetectiveAgent(
        num_detectives=5
    )
    detective_positions = [13,26,29,34,50]
    # Profile the game session
    move_count = 0
    start_time = time.perf_counter()
    game.initialize_shadow_chase_game(detective_positions, 91)
    
    while not game.is_game_over() and move_count < num_moves:
        current_player = game.game_state.turn
        
        # try:
        if current_player == Player.MRX:
            # Mr. X move
            move = mr_x_agent.choose_move(game)
            if move:
                dest, transport, use_double = move
                game.make_move(mr_x_moves=[(dest, transport)], use_double_move=use_double)
        else:
            # Detective moves
            moves = detective_agent.choose_all_moves(game)
            if moves:
                game.make_move(detective_moves=moves)
        
        # except Exception as e:
        #     print(f"Error during move {move_count}: {e}")
        #     break
        
        move_count += 1
        
        # Track game state for caching analysis
        state_key = f"{game.game_state.turn_count}_{game.game_state.MrX_position}_" + \
                   "_".join(map(str, game.game_state.detective_positions))
        game_analyzer.track_state_hash(state_key)
    
    end_time = time.perf_counter()
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    total_time = end_time - start_time
    
    return {
        'session_stats': {
            'total_moves': move_count,
            'total_time': total_time,
            'avg_time_per_move': total_time / move_count if move_count > 0 else 0,
            'peak_memory_mb': peak / 1024 / 1024,
            'current_memory_mb': current / 1024 / 1024
        },
        'function_stats': function_profiler.get_stats(),
        'caching_analysis': game_analyzer.get_analysis(),
        'agent_stats': {
            'mr_x_stats': mr_x_agent.get_statistics(),
            'detective_stats': detective_agent.get_statistics()
        }
    }


def run_cprofile_analysis(num_moves: int = 5) -> str:
    """Run cProfile analysis and return formatted results."""
    print("Running cProfile analysis...")
    
    # Create a profiler
    pr = cProfile.Profile()
    
    # Profile the session
    pr.enable()
    try:
        result = run_mcts_profiling_session(num_moves)
    finally:
        pr.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    return s.getvalue()


def analyze_optimization_opportunities(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the profiling results to identify optimization opportunities."""
    function_stats = stats.get('function_stats', {})
    caching_analysis = stats.get('caching_analysis', {})
    
    # Identify bottlenecks
    bottlenecks = []
    if function_stats:
        sorted_funcs = sorted(function_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        bottlenecks = sorted_funcs[:10]  # Top 10 time consumers
    
    # Identify caching opportunities
    caching_opportunities = {
        'state_repetition_high': caching_analysis.get('state_repetition_rate', 0) > 0.3,
        'many_copy_operations': caching_analysis.get('total_copy_operations', 0) > 100,
        'repeated_move_generations': len(caching_analysis.get('most_common_move_generations', {})) > 0,
        'deep_simulations': caching_analysis.get('avg_simulation_depth', 0) > 20
    }
    
    # Generate optimization recommendations
    recommendations = []
    
    # Check for expensive functions
    if function_stats:
        for func_name, func_stats in function_stats.items():
            if func_stats['total_time'] > 0.1:  # Functions taking > 100ms total
                if 'copy' in func_name.lower() or '__init__' in func_name:
                    recommendations.append(f"Optimize {func_name}: High memory allocation/copying detected")
                elif 'get_valid_moves' in func_name:
                    recommendations.append(f"Cache valid moves: {func_name} called {func_stats['call_count']} times")
                elif 'simulate' in func_name:
                    recommendations.append(f"Optimize simulations: {func_name} taking {func_stats['avg_time']:.4f}s on average")
    
    # Check caching opportunities
    if caching_opportunities['state_repetition_high']:
        recommendations.append("Implement transposition table: High state repetition detected")
    
    if caching_opportunities['many_copy_operations']:
        recommendations.append("Reduce deep copying: Many copy operations detected")
    
    if caching_opportunities['repeated_move_generations']:
        recommendations.append("Cache move generation: Repeated move calculations detected")
    
    return {
        'bottlenecks': bottlenecks,
        'caching_opportunities': caching_opportunities,
        'recommendations': recommendations,
        'performance_metrics': {
            'moves_per_second': 1.0 / stats.get('session_stats', {}).get('avg_time_per_move', 1.0),
            'memory_efficiency': stats.get('session_stats', {}).get('peak_memory_mb', 0),
            'cache_hit_potential': caching_analysis.get('state_repetition_rate', 0)
        }
    }


def main():
    """Main profiling function."""
    print("MCTS Agent Performance Profiler")
    print("=" * 50)
    
    # Apply profiling patches
    patch_mcts_for_profiling()
    
    # Run profiling sessions
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
        'profiling_sessions': {}
    }
    
    # Test different scenarios
    scenarios = [
        {'name': 'quick_test', 'moves': 10, 'map_size': 'extended'},
        # {'name': 'medium_test', 'moves': 10, 'map_size': 'extended'},
        {'name': 'long_test', 'moves': 50, 'map_size': 'extracted'},  # Uncomment for longer test
    ]
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        try:
            session_results = run_mcts_profiling_session(
                num_moves=scenario['moves'],
                map_size=scenario['map_size']
            )
            
            # Add cProfile analysis for the first scenario
            if scenario['name'] == 'quick_test':
                session_results['cprofile_analysis'] = run_cprofile_analysis(10)
            
            # Analyze optimization opportunities
            session_results['optimization_analysis'] = analyze_optimization_opportunities(session_results)
            
            results['profiling_sessions'][scenario['name']] = session_results
            
            # Print quick summary
            session_stats = session_results.get('session_stats', {})
            print(f"  Completed {session_stats.get('total_moves', 0)} moves in {session_stats.get('total_time', 0):.2f}s")
            print(f"  Peak memory: {session_stats.get('peak_memory_mb', 0):.1f}MB")
            
        except Exception as e:
            print(f"  Error in scenario {scenario['name']}: {e}")
            results['profiling_sessions'][scenario['name']] = {'error': str(e)}
    
    # Save results
    # output_file = "mcts_profiling_results.json"
    output_file = "mcts_profiling_results_2.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nProfiling complete! Results saved to {output_file}")
    
    # Print summary
    print("\nSUMMARY:")
    print("=" * 50)
    
    for scenario_name, scenario_results in results['profiling_sessions'].items():
        if 'error' in scenario_results:
            print(f"{scenario_name}: ERROR - {scenario_results['error']}")
            continue
            
        optimization_analysis = scenario_results.get('optimization_analysis', {})
        recommendations = optimization_analysis.get('recommendations', [])
        
        print(f"\n{scenario_name.upper()}:")
        if recommendations:
            for rec in recommendations[:10]:  # Top 5 recommendations
                print(f"  • {rec}")
        else:
            print("  • No specific optimization recommendations")
    
    print(f"\nDetailed results available in: {output_file}")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
        print(f"\nYou can now share the file '{output_file}' for optimization strategy analysis.")
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()
