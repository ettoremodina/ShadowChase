# Shadow Chase - Function Documentation

This document provides comprehensive documentation for all functions in the Shadow Chase codebase, organized by module and file.

## Table of Contents
- [Main Entry Point](#main-entry-point)
- [Game Core](#game-core)
- [Game Controls](#game-controls)
- [UI Components](#ui-components)
- [Services](#services)
- [Agents](#agents)
- [Training](#training)
- [Utilities](#utilities)

---

## Main Entry Point

### `main.py`

#### `print_game_state(game)`
**Purpose:** Print the current state of the game including turn count, player positions, and game status.
**Parameters:** 
- `game`: The game instance to display state for
**Returns:** None

#### `show_valid_moves(game, player, position=None)`
**Purpose:** Display all valid moves available for a specific player at their current position.
**Parameters:**
- `game`: The game instance
- `player`: The player (DETECTIVES or MRX)
- `position`: Optional position for detective moves
**Returns:** List of valid moves

#### `test_basic_game()`
**Purpose:** Test basic game mechanics with a small grid game setup.
**Parameters:** None
**Returns:** None

#### `test_game_until_end()`
**Purpose:** Run a complete game simulation until the game ends, testing all mechanics.
**Parameters:** None
**Returns:** None

#### `test_shadow_chase_game()`
**Purpose:** Test Shadow Chase specific game mechanics including tickets and transport types.
**Parameters:** None
**Returns:** None

#### `demo_path_game()`
**Purpose:** Demonstrate the game on a simple path graph configuration.
**Parameters:** None
**Returns:** None

#### `demo_cycle_game()`
**Purpose:** Demonstrate the game on a cycle graph configuration.
**Parameters:** None
**Returns:** None

#### `demo_grid_game()`
**Purpose:** Demonstrate the game on a grid graph configuration.
**Parameters:** None
**Returns:** None

#### `demo_shadow_chase_game()`
**Purpose:** Launch a full Shadow Chase game demonstration with GUI.
**Parameters:** None
**Returns:** None

#### `demo_simple_shadow_chase()`
**Purpose:** Run a simple Shadow Chase game without GUI visualization.
**Parameters:** None
**Returns:** None

#### `demo_shadow_chase_visualizer()`
**Purpose:** Launch the Shadow Chase game with the visual interface.
**Parameters:** None
**Returns:** None

#### `demo_test_shadow_chase()`
**Purpose:** Run a test version of Shadow Chase on a smaller map.
**Parameters:** None
**Returns:** None

#### `demo_simple_test_shadow_chase()`
**Purpose:** Run a simplified test version without visual interface.
**Parameters:** None
**Returns:** None

#### `demo_extracted_board_game(num_detectives=3, auto_init=True)`
**Purpose:** Run Shadow Chase on the extracted board data with configurable detective count.
**Parameters:**
- `num_detectives`: Number of detectives (default: 3)
- `auto_init`: Whether to auto-initialize positions (default: True)
**Returns:** None

---

## Game Core

### `ShadowChase/core/game.py`

#### Class: `Player(Enum)`
**Purpose:** Enumeration defining the two types of players in the game.
**Values:** DETECTIVES, MRX

#### Class: `TransportType(Enum)`
**Purpose:** Enumeration defining available transport methods.
**Values:** TAXI, BUS, UNDERGROUND, BLACK, FERRY

#### Class: `TicketType(Enum)`
**Purpose:** Enumeration defining ticket types used in Shadow Chase.
**Values:** TAXI, BUS, UNDERGROUND, BLACK, DOUBLE_MOVE

#### Class: `GameState`
**Purpose:** Represents the current state of a game including positions, turn info, and tickets.

#### Class: `MovementRule(ABC)`
**Purpose:** Abstract base class defining movement rules for different game variants.

#### Class: `StandardMovement(MovementRule)`
**Purpose:** Standard movement rule allowing moves to adjacent nodes.

#### Class: `DistanceKMovement(MovementRule)`
**Purpose:** Movement rule allowing moves up to distance K from current position.

#### Class: `WinCondition(ABC)`
**Purpose:** Abstract base class defining different win conditions.

#### Class: `CaptureWinCondition(WinCondition)`
**Purpose:** Win condition where detectives win by capturing MrX.

#### Class: `DistanceKWinCondition(WinCondition)`
**Purpose:** Win condition based on maintaining distance between players.

#### Class: `ShadowChaseMovement(MovementRule)`
**Purpose:** Movement rules specific to Shadow Chase game including transport constraints.

#### Class: `ShadowChaseWinCondition(WinCondition)`
**Purpose:** Win conditions specific to Shadow Chase game.

#### Class: `Game`
**Purpose:** Main game engine handling game logic, moves, and state management.

#### Class: `ShadowChaseGame(Game)`
**Purpose:** Extended game class with Shadow Chase specific features like tickets and transport.

---

## Game Controls

### `game_controls/simple_game.py`

#### `main()`
**Purpose:** Main entry point for the simple terminal-based Shadow Chase game.
**Parameters:** None
**Returns:** None

### `game_controls/game_utils.py`

#### `parse_arguments()`
**Purpose:** Parse command line arguments for batch mode and other options.
**Parameters:** None
**Returns:** Parsed argument namespace

#### `get_game_configuration()`
**Purpose:** Get interactive game configuration from user input.
**Parameters:** None
**Returns:** Tuple of (play_mode, map_size, num_detectives, verbosity, MrX_agent, detective_agent)

#### `create_and_initialize_game(map_size, num_detectives)`
**Purpose:** Create and initialize a new game with specified parameters.
**Parameters:**
- `map_size`: Size of the game map ("test" or "extended")
- `num_detectives`: Number of detective players
**Returns:** Initialized ShadowChaseGame instance

#### `save_game_session(game, play_mode, map_size, MrX_agent_type, detective_agent_type, num_detectives)`
**Purpose:** Save a completed game session with metadata.
**Parameters:**
- `game`: The game instance to save
- `play_mode`: Mode of play (human vs ai, etc.)
- `map_size`: Size of the map used
- `MrX_agent_type`: Type of MrX agent
- `detective_agent_type`: Type of detective agent
- `num_detectives`: Number of detectives
**Returns:** Game ID string

#### `save_game_automatically(game, play_mode, map_size, MrX_agent_type, detective_agent_type, num_detectives)`
**Purpose:** Automatically save game without user prompt.
**Parameters:** Same as save_game_session
**Returns:** Game ID string

#### `offer_save_option(game, play_mode, map_size, MrX_agent_type, detective_agent_type, num_detectives)`
**Purpose:** Prompt user whether to save the completed game.
**Parameters:** Same as save_game_session
**Returns:** Game ID string or None

#### `execute_single_turn(controller, game, display, max_turns)`
**Purpose:** Execute a single turn of the game including move validation and state updates.
**Parameters:**
- `controller`: GameController instance
- `game`: Game instance
- `display`: Display manager
- `max_turns`: Maximum allowed turns
**Returns:** Tuple of (continue_game, turn_count)

#### `play_single_game(map_size, play_mode, num_detectives, verbosity, MrX_agent_type, detective_agent_type, max_turns, save_method)`
**Purpose:** Play a complete single game with specified configuration.
**Parameters:**
- `map_size`: Game map size
- `play_mode`: Play mode string
- `num_detectives`: Number of detectives
- `verbosity`: Verbosity level
- `MrX_agent_type`: MrX agent type
- `detective_agent_type`: Detective agent type
- `max_turns`: Maximum turns allowed
- `save_method`: How to save the game
**Returns:** GameResult object

#### `play_multiple_games(n_games, map_size, play_mode, num_detectives, verbosity, MrX_agent_type, detective_agent_type, max_turns, save_method, save_dir)`
**Purpose:** Play multiple games in sequence for testing or analysis.
**Parameters:**
- `n_games`: Number of games to play
- Additional parameters same as play_single_game
- `save_dir`: Directory to save results
**Returns:** List of GameResult objects

### `game_controls/display_utils.py`

#### Class: `VerbosityLevel`
**Purpose:** Constants defining different levels of output verbosity.
**Constants:** SILENT, BASIC

#### Class: `GameDisplay`
**Purpose:** Manages display formatting and output for terminal-based gameplay.

#### `format_transport_input(user_input)`
**Purpose:** Parse user input for moves including destination and transport type.
**Parameters:**
- `user_input`: Raw user input string
**Returns:** Tuple of (destination, transport_type, use_black_ticket, double_move)

#### `get_user_choice(prompt, valid_choices)`
**Purpose:** Get validated user choice from a list of options.
**Parameters:**
- `prompt`: Question to display to user
- `valid_choices`: List of valid choice strings
**Returns:** Selected choice string

#### `get_user_move(display)`
**Purpose:** Get and validate a move input from the user.
**Parameters:**
- `display`: GameDisplay instance
**Returns:** Move input string

#### `display_game_start_info(display, play_mode, map_size)`
**Purpose:** Display information about the game setup at start.
**Parameters:**
- `display`: GameDisplay instance
- `play_mode`: Current play mode
- `map_size`: Current map size
**Returns:** None

#### `display_game_over(game, display, turn_count, max_turns)`
**Purpose:** Display game over information including winner and statistics.
**Parameters:**
- `game`: Game instance
- `display`: GameDisplay instance
- `turn_count`: Final turn count
- `max_turns`: Maximum turns allowed
**Returns:** None

### `game_controls/game_logic.py`

#### Class: `GameController`
**Purpose:** Handles game flow, move processing, and AI agent integration.

#### Class: `GameSetup`
**Purpose:** Manages game initialization and configuration.

#### `get_game_mode()`
**Purpose:** Interactive selection of game mode (human vs human, human vs AI, etc.).
**Parameters:** None
**Returns:** Tuple of (play_mode, map_size, num_detectives, MrX_agent, detective_agent)

#### `get_agent_configuration()`
**Purpose:** Interactive selection of AI agent types for both players.
**Parameters:** None
**Returns:** Tuple of (MrX_agent_type, detective_agent_type)

#### `get_verbosity_level()`
**Purpose:** Interactive selection of output verbosity level.
**Parameters:** None
**Returns:** Verbosity level integer

---

## UI Components

### `ShadowChase/ui/game_visualizer.py`

#### Class: `GameVisualizer(BaseVisualizer)`
**Purpose:** Main GUI application for interactive Shadow Chase gameplay with visualization.

### `ShadowChase/ui/base_visualizer.py`

#### Class: `BaseVisualizer`
**Purpose:** Base class providing common visualization utilities and graph rendering.

### `ShadowChase/ui/game_controls.py`

#### Class: `GameControls`
**Purpose:** Manages game control UI elements and user interactions in the GUI.

### `ShadowChase/ui/setup_controls.py`

#### Class: `SetupControls`
**Purpose:** Handles game setup UI including mode selection and agent configuration.

### `ShadowChase/ui/game_replay.py`

#### Class: `GameReplayWindow(BaseVisualizer)`
**Purpose:** Window for replaying saved games step by step with visual controls.

### `ShadowChase/ui/ui_components.py`

#### Class: `ScrollableFrame(ttk.Frame)`
**Purpose:** Custom scrollable frame widget for UI layouts.

#### Class: `StyledButton(ttk.Button)`
**Purpose:** Enhanced button widget with custom styling options.

#### Class: `InfoDisplay(ttk.Frame)`
**Purpose:** Text display widget for showing game information and logs.

### `ShadowChase/ui/enhanced_components.py`

#### Class: `VisualTicketDisplay(ttk.Frame)`
**Purpose:** Enhanced visual display for ticket information with table layout.

#### Class: `EnhancedTurnDisplay(ttk.Frame)`
**Purpose:** Enhanced turn display with progress indicators and status badges.

#### Class: `EnhancedMovesDisplay(ttk.Frame)`
**Purpose:** Enhanced display for showing available moves and move history.

### `ShadowChase/ui/video_exporter.py`

#### Class: `GameVideoExporter(BaseVisualizer)`
**Purpose:** Exports game replays as MP4 videos with configurable settings.

#### `show_video_export_dialog(parent, loader)`
**Purpose:** Display dialog for video export options and settings.
**Parameters:**
- `parent`: Parent window
- `loader`: GameLoader instance
**Returns:** None

#### `export_video_from_command_line(game_file, output_file, frame_duration, end_delay_seconds)`
**Purpose:** Export video from command line with specified parameters.
**Parameters:**
- `game_file`: Path to saved game file
- `output_file`: Output video file path
- `frame_duration`: Duration per frame in seconds
- `end_delay_seconds`: Delay at end of video
**Returns:** Output file path

---

## Services

### `ShadowChase/services/game_service.py`

#### Class: `GameService`
**Purpose:** Centralized service for game operations including save/load with metadata.

### `ShadowChase/services/game_loader.py`

#### Class: `GameRecord`
**Purpose:** Represents a saved game record with metadata and history.

#### Class: `GameLoader`
**Purpose:** Handles saving and loading games with organized folder structure.

### `ShadowChase/services/board_loader.py`

#### `load_board_graph_from_csv(nodes_file, edges_file)`
**Purpose:** Load game board graph from optimized CSV files.
**Parameters:**
- `nodes_file`: Path to nodes CSV file
- `edges_file`: Path to edges CSV file
**Returns:** Tuple of (graph, positions)

#### `create_extracted_board_game(num_detectives, nodes_file, edges_file)`
**Purpose:** Create Shadow Chase game using extracted board data.
**Parameters:**
- `num_detectives`: Number of detective players
- `nodes_file`: Nodes data file
- `edges_file`: Edges data file
**Returns:** ShadowChaseGame instance

#### `load_board_metadata(metadata_file)`
**Purpose:** Load board configuration metadata from JSON file.
**Parameters:**
- `metadata_file`: Path to metadata file
**Returns:** Metadata dictionary

#### `load_transport_colors(colors_file)`
**Purpose:** Load transport type color mappings for visualization.
**Parameters:**
- `colors_file`: Path to colors configuration file
**Returns:** Color mapping dictionary

### `ShadowChase/services/cache_system.py`

#### Class: `CacheNamespace(Enum)`
**Purpose:** Enumeration of different cache namespaces for organizing cached data.

#### Class: `CacheEntry`
**Purpose:** Data structure representing a cached entry with metadata.

#### Class: `PersistentCache`
**Purpose:** High-performance persistent caching system with LRU eviction and compression.

#### `get_global_cache()`
**Purpose:** Get the global cache instance, creating it if necessary.
**Parameters:** None
**Returns:** PersistentCache instance

#### `init_cache(cache_dir, **kwargs)`
**Purpose:** Initialize the global cache with specified directory and options.
**Parameters:**
- `cache_dir`: Directory for cache files
- `**kwargs`: Additional cache configuration
**Returns:** PersistentCache instance

#### `enable_cache()`
**Purpose:** Enable the global caching system.
**Parameters:** None
**Returns:** None

#### `disable_cache()`
**Purpose:** Disable the global caching system.
**Parameters:** None
**Returns:** None

#### `is_cache_enabled()`
**Purpose:** Check if the global cache is currently enabled.
**Parameters:** None
**Returns:** Boolean cache status

#### `enable_namespace_cache(namespace)`
**Purpose:** Enable caching for a specific namespace.
**Parameters:**
- `namespace`: CacheNamespace to enable
**Returns:** None

#### `disable_namespace_cache(namespace)`
**Purpose:** Disable caching for a specific namespace.
**Parameters:**
- `namespace`: CacheNamespace to disable
**Returns:** None

#### `is_namespace_cache_enabled(namespace)`
**Purpose:** Check if caching is enabled for a specific namespace.
**Parameters:**
- `namespace`: CacheNamespace to check
**Returns:** Boolean status

#### `reset_namespace_cache_settings()`
**Purpose:** Reset all namespace cache settings to default.
**Parameters:** None
**Returns:** None

#### `get_cache_status()`
**Purpose:** Get comprehensive status of the cache system.
**Parameters:** None
**Returns:** Dictionary with cache status information

#### `cache_game_method(namespace, ttl_hours)`
**Purpose:** Decorator for caching game method results.
**Parameters:**
- `namespace`: Cache namespace to use
- `ttl_hours`: Time-to-live in hours
**Returns:** Decorator function

#### `cache_agent_decision(agent_type, agent_id, namespace, ttl_hours)`
**Purpose:** Decorator for caching agent decision results.
**Parameters:**
- `agent_type`: Type of agent
- `agent_id`: Agent identifier
- `namespace`: Cache namespace
- `ttl_hours`: Time-to-live in hours
**Returns:** Decorator function

### `ShadowChase/services/analyze_games.py`

#### `get_display_name(agent_name)`
**Purpose:** Convert internal agent names to user-friendly display names.
**Parameters:**
- `agent_name`: Internal agent name
**Returns:** Display name string

#### `calculate_proportion_confidence_interval(successes, total, confidence)`
**Purpose:** Calculate confidence interval for win rate proportions.
**Parameters:**
- `successes`: Number of successes
- `total`: Total attempts
- `confidence`: Confidence level (default: 0.95)
**Returns:** Tuple of (lower_bound, upper_bound)

#### `calculate_mean_confidence_interval(values, confidence)`
**Purpose:** Calculate confidence interval for mean of sample values.
**Parameters:**
- `values`: List of values
- `confidence`: Confidence level
**Returns:** Tuple of (mean, lower_bound, upper_bound)

#### Class: `GameStatistics`
**Purpose:** Container for comprehensive game statistics and analysis.

#### Class: `GameAnalyzer`
**Purpose:** Analyzes game results and generates comprehensive visualizations.

---

## Agents

### `agents/base_agent.py`

#### Class: `Agent(ABC)`
**Purpose:** Abstract base class for all AI agents in the game.

#### Class: `DetectiveAgent(Agent)`
**Purpose:** Base class for agents controlling detective players.

#### Class: `MrXAgent(Agent)`
**Purpose:** Base class for agents controlling Mr. X player.

#### Class: `MultiDetectiveAgent(Agent)`
**Purpose:** Base class for agents controlling multiple detective players.

### `agents/random_agent.py`

#### Class: `RandomMrXAgent(MrXAgent)`
**Purpose:** Mr. X agent that makes completely random valid moves.

#### Class: `RandomMultiDetectiveAgent(MultiDetectiveAgent)`
**Purpose:** Agent controlling all detectives with random move selection.

### `agents/heuristic_agent.py`

#### Class: `HeuristicMrXAgent(MrXAgent)`
**Purpose:** Mr. X agent using heuristic-based decision making.

#### Class: `HeuristicMultiDetectiveAgent(MultiDetectiveAgent)`
**Purpose:** Detective agent using heuristic strategies.

### `agents/mcts_agent.py`

#### `load_mcts_config(config_path)`
**Purpose:** Load MCTS configuration from JSON file.
**Parameters:**
- `config_path`: Path to configuration file
**Returns:** Configuration dictionary

#### Class: `MCTSNode`
**Purpose:** Node in Monte Carlo Tree Search representing a game state.

#### Class: `MCTSAgent`
**Purpose:** Base agent using Monte Carlo Tree Search algorithm.

#### Class: `MCTSMrXAgent(MrXAgent, MCTSAgent)`
**Purpose:** Mr. X agent using MCTS for decision making.

#### Class: `MCTSDetectiveAgent(DetectiveAgent, MCTSAgent)`
**Purpose:** Detective agent using MCTS for decision making.

#### Class: `MCTSMultiDetectiveAgent(MultiDetectiveAgent)`
**Purpose:** Multi-detective agent using MCTS.

### `agents/optimized_mcts_agent.py`

#### Class: `GameStateHash(NamedTuple)`
**Purpose:** Hashable representation of game state for caching.

#### Class: `CachedNodeResult(NamedTuple)`
**Purpose:** Cached result from MCTS node evaluation.

#### Class: `GameStateCache`
**Purpose:** Cache system for MCTS game state evaluations.

#### Class: `OptimizedMCTSNode`
**Purpose:** Optimized MCTS node with caching and performance improvements.

#### Class: `OptimizedMCTSAgent`
**Purpose:** Optimized MCTS agent with enhanced performance features.

#### Class: `OptimizedMCTSMrXAgent(MrXAgent, OptimizedMCTSAgent)`
**Purpose:** Optimized Mr. X agent using enhanced MCTS.

#### Class: `OptimizedMCTSDetectiveAgent(DetectiveAgent, OptimizedMCTSAgent)`
**Purpose:** Optimized detective agent using enhanced MCTS.

### `agents/epsilon_greedy_mcts_agent.py`

#### Class: `GameStateHash(NamedTuple)`
**Purpose:** Hashable game state representation for epsilon-greedy MCTS.

#### Class: `CachedNodeResult(NamedTuple)`
**Purpose:** Cached node evaluation result.

#### Class: `GameStateCache`
**Purpose:** Specialized cache for epsilon-greedy MCTS.

#### Class: `EpsilonGreedyMCTSNode`
**Purpose:** MCTS node with epsilon-greedy exploration strategy.

#### Class: `EpsilonGreedyMCTSAgent`
**Purpose:** MCTS agent using epsilon-greedy exploration.

#### Class: `EpsilonGreedyMCTSMrXAgent(MrXAgent, EpsilonGreedyMCTSAgent)`
**Purpose:** Mr. X agent with epsilon-greedy MCTS.

#### Class: `EpsilonGreedyMCTSDetectiveAgent(DetectiveAgent, EpsilonGreedyMCTSAgent)`
**Purpose:** Detective agent with epsilon-greedy MCTS.

#### Class: `EpsilonGreedyMCTSMultiDetectiveAgent(MultiDetectiveAgent)`
**Purpose:** Multi-detective agent with epsilon-greedy MCTS.

### `agents/dqn_agent.py`

#### Class: `DQNMrXAgent(MrXAgent)`
**Purpose:** Mr. X agent using Deep Q-Network for decision making.

#### Class: `DQNMultiDetectiveAgent(MultiDetectiveAgent)`
**Purpose:** Multi-detective agent using Deep Q-Network.

### `agents/heuristics.py`

#### Class: `GameHeuristics`
**Purpose:** Provides heuristic calculations for game position evaluation.

---

## Training

### `training/base_trainer.py`

#### Class: `TrainingResult`
**Purpose:** Data structure containing results from a training session.

#### Class: `EvaluationResult`
**Purpose:** Data structure containing evaluation metrics.

#### Class: `BaseTrainer(ABC)`
**Purpose:** Abstract base class for all training algorithms.

### `training/training_environment.py`

#### Class: `GameResult`
**Purpose:** Container for single game result data.

#### Class: `TrainingEnvironment`
**Purpose:** Standardized environment for training AI agents.

### `training/feature_extractor_simple.py`

#### Class: `FeatureConfig`
**Purpose:** Configuration for feature extraction parameters.

#### Class: `GameFeatureExtractor`
**Purpose:** Converts game states to numerical feature vectors for ML algorithms.

### `training/deep_q/dqn_model.py`

#### Class: `DQNModel(nn.Module)`
**Purpose:** Deep Q-Network model for reinforcement learning.

#### `create_dqn_model(config)`
**Purpose:** Factory function to create DQN model with specified configuration.
**Parameters:**
- `config`: Model configuration dictionary
**Returns:** DQNModel instance

#### `test_action_querying()`
**Purpose:** Test function for action querying functionality.
**Parameters:** None
**Returns:** None

#### `test_action_encoding_performance()`
**Purpose:** Test function for action encoding performance.
**Parameters:** None
**Returns:** None

### `training/deep_q/dqn_trainer.py`

#### Class: `DQNTrainer(BaseTrainer)`
**Purpose:** Trainer for Deep Q-Network agents using the training infrastructure.

### `training/deep_q/replay_buffer.py`

#### Class: `ReplayBuffer`
**Purpose:** Experience replay buffer for DQN training.

#### Class: `PrioritizedReplayBuffer(ReplayBuffer)`
**Purpose:** Prioritized experience replay buffer with importance sampling.

#### `create_replay_buffer(config)`
**Purpose:** Factory function to create replay buffer with specified configuration.
**Parameters:**
- `config`: Buffer configuration dictionary
**Returns:** ReplayBuffer instance

---

## Utilities

### `train_dqn.py`

#### `plot_training_metrics(trainer, save_path, plotting_every)`
**Purpose:** Plot training metrics including rewards, losses, and performance.
**Parameters:**
- `trainer`: DQN trainer instance
- `save_path`: Path to save plot
- `plotting_every`: Frequency of plotting
**Returns:** None

#### `train_with_monitoring(player_role, num_episodes, plotting_every, device)`
**Purpose:** Train DQN agent with real-time monitoring and plotting.
**Parameters:**
- `player_role`: Role to train ("MrX" or "Detective")
- `num_episodes`: Number of training episodes
- `plotting_every`: Plot update frequency
- `device`: Training device (CPU/GPU)
**Returns:** None

#### `evaluate_trained_agent(model_path, player_role, num_games, device)`
**Purpose:** Evaluate a trained DQN agent's performance.
**Parameters:**
- `model_path`: Path to saved model
- `player_role`: Role of the agent
- `num_games`: Number of evaluation games
- `device`: Evaluation device
**Returns:** None

#### `main()`
**Purpose:** Main entry point for DQN training script.
**Parameters:** None
**Returns:** None

### `test_agents.py`

#### `play_combination(test_name, MrX_agent, detective_agent, games_per_combo, map_size, num_detectives, max_turns)`
**Purpose:** Run a specific combination of agents for testing.
**Parameters:**
- `test_name`: Name of the test session
- `MrX_agent`: MrX agent type
- `detective_agent`: Detective agent type
- `games_per_combo`: Number of games to play
- `map_size`: Size of game map
- `num_detectives`: Number of detectives
- `max_turns`: Maximum turns per game
**Returns:** None

#### `analyze_games(test_name)`
**Purpose:** Analyze results from agent testing sessions.
**Parameters:**
- `test_name`: Name of test session to analyze
**Returns:** None

#### `main()`
**Purpose:** Main entry point for agent testing script.
**Parameters:** None
**Returns:** None

### `other/visualize_board.py`

#### `load_board_data(progress_file)`
**Purpose:** Load board data from JSON file for visualization.
**Parameters:**
- `progress_file`: Path to board progress file
**Returns:** Tuple of (nodes, edges)

#### `visualize_board(save_path)`
**Purpose:** Create and display board visualization with matplotlib.
**Parameters:**
- `save_path`: Optional path to save visualization
**Returns:** None

### `other/create_board_data.py`

#### `create_optimized_board_data(progress_file, output_dir)`
**Purpose:** Convert board_progress.json to optimized CSV files for fast loading.
**Parameters:**
- `progress_file`: Input JSON file
- `output_dir`: Output directory for CSV files
**Returns:** None

### Performance Analysis Tools

#### `other/analyze_cache_performance.py`

#### Class: `CachePerformanceAnalyzer`
**Purpose:** Analyzes how cache performance changes as it gets populated with data.

#### `other/compare_cache_performance.py`

#### Class: `CachePerformanceComparison`
**Purpose:** Compares game performance with and without cache enabled.

#### `other/test_mcts_cache_performance.py`

#### Class: `MCTSCachePerformanceTest`
**Purpose:** Tests the specific impact of MCTS caching on agent performance.

#### `other/test_random_cache_performance.py`

#### Class: `RandomAgentCacheTest`
**Purpose:** Tests cache overhead with random agents to isolate caching impact.

#### `other/profile_mcts_agent.py`

#### Class: `FunctionProfiler`
**Purpose:** Profiles function calls to identify performance bottlenecks.

#### Class: `GameStateAnalyzer`
**Purpose:** Analyzes game state patterns and frequencies.

#### `patch_mcts_for_profiling()`
**Purpose:** Dynamically patch MCTS methods with profiling decorators.
**Parameters:** None
**Returns:** None

#### `run_mcts_profiling_session(num_moves, map_size)`
**Purpose:** Run a profiling session for MCTS agent performance.
**Parameters:**
- `num_moves`: Number of moves to profile
- `map_size`: Size of game map
**Returns:** Profiling statistics dictionary

#### `run_cprofile_analysis(num_moves)`
**Purpose:** Run cProfile analysis on MCTS agent.
**Parameters:**
- `num_moves`: Number of moves to analyze
**Returns:** Profile output string

#### `analyze_optimization_opportunities(stats)`
**Purpose:** Analyze profiling statistics to identify optimization opportunities.
**Parameters:**
- `stats`: Profiling statistics dictionary
**Returns:** Analysis results dictionary

### Board Creation Tools

#### `other/createBoard.py`

#### Class: `Phase(Enum)`
**Purpose:** Enumeration of board creation phases.

#### Class: `Mode(Enum)`
**Purpose:** Enumeration of interaction modes during board creation.

#### Class: `Node`
**Purpose:** Data structure representing a board node.

#### Class: `Edge`
**Purpose:** Data structure representing a board edge.

#### Class: `ShadowChaseBoardCreator`
**Purpose:** Interactive tool for creating Shadow Chase board from image.

#### `other/calibrate_board_overlay.py`

#### Class: `BoardCalibrator`
**Purpose:** Tool for calibrating board image overlay with graph coordinates.

---

## Examples

### `ShadowChase/examples/example_games.py`

#### `create_path_graph_game(n, num_detectives)`
**Purpose:** Create a game on a path graph with n nodes.
**Parameters:**
- `n`: Number of nodes in path
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_cycle_graph_game(n, num_detectives)`
**Purpose:** Create a game on a cycle graph with n nodes.
**Parameters:**
- `n`: Number of nodes in cycle
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_complete_graph_game(n, num_detectives)`
**Purpose:** Create a game on a complete graph with n nodes.
**Parameters:**
- `n`: Number of nodes
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_grid_graph_game(m, n, num_detectives)`
**Purpose:** Create a game on an m×n grid graph.
**Parameters:**
- `m`: Grid height
- `n`: Grid width
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_petersen_graph_game(num_detectives)`
**Purpose:** Create a game on the Petersen graph.
**Parameters:**
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_distance_k_game(graph, k, num_detectives)`
**Purpose:** Create a game with distance-k movement rules.
**Parameters:**
- `graph`: NetworkX graph
- `k`: Maximum movement distance
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_distance_k_win_game(graph, k, num_detectives)`
**Purpose:** Create a game with distance-k win conditions.
**Parameters:**
- `graph`: NetworkX graph
- `k`: Win distance threshold
- `num_detectives`: Number of detective players
**Returns:** Game instance

#### `create_shadowChase_game(num_detectives)`
**Purpose:** Create a standard Shadow Chase game.
**Parameters:**
- `num_detectives`: Number of detective players
**Returns:** ShadowChaseGame instance

#### `create_simple_shadow_chase_game(num_detectives, start_positions)`
**Purpose:** Create a simple Shadow Chase game for testing.
**Parameters:**
- `num_detectives`: Number of detective players
- `start_positions`: Starting positions dictionary
**Returns:** ShadowChaseGame instance

#### `create_test_shadow_chase_game(num_detectives)`
**Purpose:** Create a test Shadow Chase game on smaller map.
**Parameters:**
- `num_detectives`: Number of detective players
**Returns:** ShadowChaseGame instance

#### `create_simple_test_shadow_chase_game(num_detectives, start_positions)`
**Purpose:** Create a simple test Shadow Chase game with custom positions.
**Parameters:**
- `num_detectives`: Number of detective players
- `start_positions`: Starting positions dictionary
**Returns:** ShadowChaseGame instance

#### `create_extracted_board_game(num_detectives)`
**Purpose:** Create a Shadow Chase game using extracted board data.
**Parameters:**
- `num_detectives`: Number of detective players
**Returns:** ShadowChaseGame instance

---

This documentation covers all major functions and classes in the Shadow Chase codebase. Each function includes its purpose, parameters, and return values to help developers understand and use the codebase effectively.
