"""
Game logic utilities for simple terminal-based Scotland Yard gameplay.
Handles move validation, AI moves, and game flow.
"""
import random
from typing import List, Tuple, Optional
from ShadowChase.core.game import ScotlandYardGame, Player, TicketType, TransportType
from .display_utils import GameDisplay, format_transport_input
from ShadowChase.services.board_loader import load_board_graph_from_csv
from agents import AgentType, AgentSelector, get_agent_registry


class GameController:
    """Handles game logic and flow"""
    
    def __init__(self, game: ScotlandYardGame, display: GameDisplay, 
                 mr_x_agent_type: AgentType = AgentType.RANDOM, 
                 detective_agent_type: AgentType = AgentType.RANDOM):
        self.game = game
        self.display = display
        self.current_detective = 0
        self.double_move_first = False
        
        # Initialize agents using the registry
        registry = get_agent_registry()
        num_detectives = len(game.game_state.detective_positions)
        self.agent_mrx = registry.create_mr_x_agent(mr_x_agent_type)
        self.agent_detectives = registry.create_multi_detective_agent(detective_agent_type, num_detectives)
    
    def make_all_detective_moves(self) -> bool:
        """Handle all detective moves simultaneously"""
        if self.game.game_state.turn != Player.DETECTIVES:
            self.display.print_error("Not detectives' turn")
            return False
        
        detective_moves = []
        
        # Collect moves from all detectives
        for i, detective_pos in enumerate(self.game.game_state.detective_positions):
            print(f"\n--- Detective {i+1} (Position {detective_pos}) ---")
            
            # Check if detective can move with pending moves consideration
            if isinstance(self.game, ScotlandYardGame):
                available_moves = self.game.get_valid_moves(Player.DETECTIVES, detective_pos, pending_moves=detective_moves)
            else:
                available_moves = self.game.get_valid_moves(Player.DETECTIVES, detective_pos)
            
            if not available_moves:
                self.display.print_info(f"Detective {i+1} has no valid moves (staying in place)")
                detective_moves.append((detective_pos, None))  # Stay in place
                continue
            
            # Get move from user
            move = self._get_detective_move_input(i + 1, detective_pos, available_moves, detective_moves)
            if move is None:
                return False  # User quit
            
            detective_moves.append(move)
        
        # Make the combined move
        success = self.game.make_move(detective_moves=detective_moves)
        
        if success:
            self.display.print_move_result(True, "All detectives moved")
            return True
        else:
            self.display.print_error("Detective moves failed")
            return False
    
    def _get_detective_move_input(self, detective_num: int, position: int, available_moves, pending_moves: List[Tuple[int, TransportType]]) -> Optional[Tuple[int, TransportType]]:
        """Get move input for a single detective"""
        # Show available moves (already filtered by pending moves)
        print(f"\n🎯 AVAILABLE MOVES for Detective {detective_num} at position {position}:")
        
        if not available_moves:
            print("  ❌ No valid moves available!")
            return (position, None)  # Stay in place
        
        # Group moves by destination
        move_dict = {}
        for dest, transport in available_moves:
            if dest not in move_dict:
                move_dict[dest] = []
            move_dict[dest].append(transport)
        
        for dest in sorted(move_dict.keys()):
            transports = move_dict[dest]
            transport_icons = [self.display.symbols.get(t.name.lower(), '🎫') for t in transports]
            print(f"  → Position {dest}: {' '.join(transport_icons)}")
            
            if self.display.verbosity >= 3:  # Detailed level
                transport_names = [t.name.capitalize() for t in transports]
                print(f"    ({', '.join(transport_names)})")
        
        # Show which positions are already taken by pending moves
        taken_positions = [move[0] for move in pending_moves if move[1] is not None]
        if taken_positions:
            print(f"  ⚠️  Positions already taken by other detectives: {taken_positions}")
        
        while True:
            move_input = input(f"\n🎮 Detective {detective_num} move: ").strip()
            
            if move_input.lower() in ['help', 'h']:
                self.display.print_input_help()
                continue
            
            if move_input.lower() in ['quit', 'exit', 'q']:
                return None
            
            # Parse input
            dest, transport, use_black, is_double = format_transport_input(move_input)
            
            if dest is None:
                self.display.print_error("Invalid input format")
                continue
            
            # Find valid transport for this destination
            valid_transports = []
            for move_dest, move_transport in available_moves:
                if move_dest == dest:
                    valid_transports.append(move_transport)
            
            if not valid_transports:
                self.display.print_error(f"Cannot move to position {dest}")
                continue
            
            # Choose transport
            if transport is None and len(valid_transports) == 1:
                transport = valid_transports[0]
            elif transport is None and len(valid_transports) > 1:
                print(f"Multiple transport options for position {dest}:")
                for i, t in enumerate(valid_transports):
                    icon = self.display.symbols.get(t.name.lower(), '🎫')
                    print(f"  {i+1}. {icon} {t.name.capitalize()}")
                
                while True:
                    try:
                        choice = int(input("Choose transport (number): ")) - 1
                        if 0 <= choice < len(valid_transports):
                            transport = valid_transports[choice]
                            break
                        else:
                            print("Invalid choice")
                    except ValueError:
                        print("Please enter a number")
            elif transport not in valid_transports:
                available_names = [t.name.capitalize() for t in valid_transports]
                self.display.print_error(f"Cannot use {transport.name} transport. Available: {', '.join(available_names)}")
                continue
            
            return (dest, transport)
    
    def make_mr_x_move(self) -> bool:
        """Handle Mr. X move"""
        available_moves = self.display.print_available_moves(self.game, Player.MRX)
        if not available_moves:
            return False
        
        # Check for double move
        double_move_available = self.game.can_use_double_move()
        
        while True:
            if double_move_available and not self.double_move_first:
                print(f"\n{self.display.symbols['double_move']} Double move available! Type 'DD' to use it.")
            
            move_input = input(f"\n🎮 Mr. X move: ").strip()
            
            if move_input.lower() in ['help', 'h']:
                self.display.print_input_help()
                continue
            
            if move_input.lower() in ['quit', 'exit', 'q']:
                return False
            
            # Parse input
            dest, transport, use_black, is_double = format_transport_input(move_input)
            
            # Handle double move
            if is_double and double_move_available and not self.double_move_first:
                self.display.print_info("Double move activated! Enter first move:")
                self.double_move_first = True
                continue
            
            if dest is None:
                self.display.print_error("Invalid input format")
                continue
            
            # Find valid transport for this destination
            valid_transports = []
            for move_dest, move_transport in available_moves:
                if move_dest == dest:
                    valid_transports.append(move_transport)
            
            if not valid_transports:
                self.display.print_error(f"Cannot move to position {dest}")
                continue
            
            # Choose transport
            chosen_transport = None
            
            if use_black:
                # Check if black ticket is available
                mr_x_tickets = self.game.get_mr_x_tickets()
                if mr_x_tickets.get(TicketType.BLACK, 0) > 0:
                    chosen_transport = TransportType.BLACK
                else:
                    self.display.print_error("No black tickets available")
                    continue
            elif transport is not None:
                if transport in valid_transports:
                    chosen_transport = transport
                else:
                    available_names = [t.name.capitalize() for t in valid_transports]
                    self.display.print_error(f"Cannot use {transport.name} transport. Available: {', '.join(available_names)}")
                    continue
            else:
                # Auto-choose transport if only one option
                if len(valid_transports) == 1:
                    chosen_transport = valid_transports[0]
                else:
                    print(f"Multiple transport options for position {dest}:")
                    for i, t in enumerate(valid_transports):
                        icon = self.display.symbols.get(t.name.lower(), '🎫')
                        print(f"  {i+1}. {icon} {t.name.capitalize()}")
                    
                    while True:
                        try:
                            choice = int(input("Choose transport (number): ")) - 1
                            if 0 <= choice < len(valid_transports):
                                chosen_transport = valid_transports[choice]
                                break
                            else:
                                print("Invalid choice")
                        except ValueError:
                            print("Please enter a number")
            
            # Make move
            use_double_move = self.double_move_first
            success = self.game.make_move(mr_x_moves=[(dest, chosen_transport)], use_double_move=use_double_move)
            
            if success:
                move_desc = f"Mr. X moved to position {dest}"
                if use_double_move:
                    if self.game.game_state.double_move_active:
                        move_desc += " (first move of double move)"
                        self.display.print_move_result(True, move_desc)
                        # Continue for second move
                        self.double_move_first = False
                        available_moves = self.display.print_available_moves(self.game, Player.MRX)
                        continue
                    else:
                        move_desc += " (second move of double move)"
                        self.double_move_first = False
                
                self.display.print_move_result(True, move_desc)
                return True
            else:
                self.display.print_error("Move failed")
                continue
    
    def make_ai_move(self, player: Player) -> bool:
        """Make an AI move using the agent system"""
        if player == Player.DETECTIVES:
            detective_moves = self.agent_detectives.choose_all_moves(self.game)
            if detective_moves:
                success = self.game.make_move(detective_moves=detective_moves)
                if success:
                    self.display.print_move_result(True, "AI Detectives moved")
                    return True
                else:
                    self.display.print_error("AI Detective moves failed")
                    return False
            else:
                self.display.print_error("AI Detectives could not find valid moves")
                return False
        else:  # Player.MRX
            dest, transport, use_double  = self.agent_mrx.choose_move(self.game)
            if dest is not None and transport is not None:
                success = self.game.make_move(mr_x_moves=[(dest, transport)], use_double_move=use_double)
                if success:
                    self.display.print_move_result(True, "AI Mr. X moved")
                    return True
                else:
                    self.display.print_error("AI Mr. X move failed")
                    return False
            else:
                self.display.print_info("AI Mr. X has no valid moves")
                self.game.game_state.turn = Player.DETECTIVES  # Skip to next detective turn
                # AI could not move, game over
                return True
    
    def reset_turn_state(self):
        """Reset turn-specific state"""
        self.current_detective = 0
        self.double_move_first = False


class GameSetup:
    """Handles game initialization with predefined scenarios"""
    
    @staticmethod
    def create_test_game(num_detectives: int = 2) -> ScotlandYardGame:
        """Create a test Scotland Yard game (small map)"""
        from ShadowChase.examples.example_games import create_test_scotland_yard_game
        return create_test_scotland_yard_game(num_detectives)
    
    @staticmethod
    def create_full_game(num_detectives: int = 3) -> ScotlandYardGame:
        """Create a full Scotland Yard game (full map)"""
        # Try to use extracted board first, fall back to CSV
        try:
            from ShadowChase.examples.example_games import create_extracted_board_game
            return create_extracted_board_game(num_detectives)
        except:
            # Fall back to original CSV-based game
            from ShadowChase.examples.example_games import create_scotlandYard_game
            return create_scotlandYard_game(num_detectives)
    
    @staticmethod
    def create_extracted_board_game(num_detectives: int = 3) -> ScotlandYardGame:
        """Create a Scotland Yard game using extracted board data"""
        from ShadowChase.examples.example_games import create_extracted_board_game
        return create_extracted_board_game(num_detectives)
    
    @staticmethod
    def initialize_test_positions(game: ScotlandYardGame) -> None:
        """Initialize with predefined test positions"""
        detective_positions = [1, 2]
        mr_x_position = 9
        game.initialize_scotland_yard_game(detective_positions, mr_x_position)
    
    
    @staticmethod
    def initialize_extracted_board_positions(game: ScotlandYardGame, num_detectives: int) -> None:
        """Initialize with positions suitable for the extracted board"""
        # Get available nodes from the game graph
        available_nodes = list(game.graph.nodes())
        
        if len(available_nodes) < num_detectives + 1:
            raise ValueError(f"Not enough nodes ({len(available_nodes)}) for {num_detectives} detectives + Mr. X")
        
        # Use predefined positions if they exist on the extracted board
        starting_cards = [13,26,29,34,50,53,91,103,112,132,138,141,155,174,197,94, 117, 198]
        sample = random.sample(starting_cards, num_detectives+1)

        detective_positions = sample[1:num_detectives+1]        
        mr_x_position = sample[0]  # First position is Mr. X
        
        game.initialize_scotland_yard_game(detective_positions, mr_x_position)


def get_game_mode() -> Tuple[str, str, int, AgentType, AgentType]:
    """Get game configuration from user, including agent types for AI players"""
    print("\n🎮 GAME SETUP")
    print("=" * 50)
    
    # Map size
    print("\n📍 Choose map size:")
    print("1. Test map (10 nodes) - Good for learning")
    print("2. Extracted board (199 nodes) - Your custom extracted board")
    
    while True:
        map_choice = input("Map size (1-2): ").strip()
        if map_choice == "1":
            map_size = "test"
            num_detectives = 2
            break
        elif map_choice == "2":
            map_size = "extracted"
            # Check if extracted board is available
            try:
                
                graph, positions = load_board_graph_from_csv()
                print(f"✓ Found extracted board with {len(graph.nodes())} nodes")
                num_detectives = 5
                break
            except FileNotFoundError:
                print("❌ board_progress.json not found. Please extract board data first using createBoard.py")
                print("Falling back to other options...")
                continue
            except ImportError:
                print("❌ Board loader not available. Please check your installation.")
                print("Falling back to other options...")
                continue
        else:
            print("Please enter 1, 2, or 3")
    
    # Play mode
    print(f"\n🎭 Choose play mode:")
    print("1. Human vs Human - You control both sides")
    print("2. Human Detectives vs AI Mr. X")
    print("3. AI Detectives vs Human Mr. X")
    print("4. AI vs AI - Watch the game play")
    
    while True:
        mode_choice = input("Play mode (1-4): ").strip()
        if mode_choice == "1":
            play_mode = "human_vs_human"
            break
        elif mode_choice == "2":
            play_mode = "human_det_vs_ai_mrx"
            break
        elif mode_choice == "3":
            play_mode = "ai_det_vs_human_mrx"
            break
        elif mode_choice == "4":
            play_mode = "ai_vs_ai"
            break
        else:
            print("❌ Please enter 1, 2, 3, or 4")
    
    # Get agent configurations for AI players
    mr_x_agent_type = AgentType.RANDOM
    detective_agent_type = AgentType.RANDOM
    
    if play_mode in ["human_det_vs_ai_mrx", "ai_vs_ai"]:
        print(f"\n🤖 Select AI Agent for Mr. X:")
        mr_x_agent_type = AgentSelector.get_user_agent_choice("Choose Mr. X AI agent type")
    
    if play_mode in ["ai_det_vs_human_mrx", "ai_vs_ai"]:
        print(f"\n🕵️ Select AI Agent for Detectives:")
        detective_agent_type = AgentSelector.get_user_agent_choice("Choose Detective AI agent type")
    
    return map_size, play_mode, num_detectives, mr_x_agent_type, detective_agent_type


def get_agent_configuration() -> Tuple[AgentType, AgentType]:
    """Get agent configuration separately for use in batch mode"""
    print(f"\n🤖 AI AGENT CONFIGURATION")
    print("=" * 50)
    
    print(f"\n🤖 Select AI Agent for Mr. X:")
    mr_x_agent_type = AgentSelector.get_user_agent_choice("Choose Mr. X AI agent type")
    
    print(f"\n🕵️ Select AI Agent for Detectives:")
    detective_agent_type = AgentSelector.get_user_agent_choice("Choose Detective AI agent type")
    
    return mr_x_agent_type, detective_agent_type


def get_verbosity_level() -> int:
    """Get display verbosity level from user"""
    print(f"\n📊 Choose display verbosity:")
    print("1. Basic - Just positions and turn")
    print("2. Standard - + Available moves and tickets")
    print("3. Detailed - + Move history and transport details")
    print("4. Debug - + All internal game state")
    print("5. Heuristics - + Possible Mr. X positions analysis")
    
    while True:
        verb_choice = input("Verbosity level (1-5): ").strip()
        if verb_choice in ["1", "2", "3", "4", "5"]:
            return int(verb_choice)
        print("❌ Please enter 1, 2, 3, 4, or 5")
