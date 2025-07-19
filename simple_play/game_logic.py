"""
Game logic utilities for simple terminal-based Scotland Yard gameplay.
Handles move validation, AI moves, and game flow.
"""
import random
from typing import List, Tuple, Optional
from cops_and_robbers.core.game import ScotlandYardGame, Player, TicketType, TransportType
from .display_utils import GameDisplay, format_transport_input


class GameController:
    """Handles game logic and flow"""
    
    def __init__(self, game: ScotlandYardGame, display: GameDisplay):
        self.game = game
        self.display = display
        self.current_detective = 0
        self.double_move_first = False
    
    def make_human_move(self, player: Player, position: Optional[int] = None) -> bool:
        """Handle human player move input"""
        if player == Player.COPS:
            return self._make_detective_move(position)
        else:
            return self._make_mr_x_move()
    
    def make_all_detective_moves(self) -> bool:
        """Handle all detective moves simultaneously"""
        if self.game.game_state.turn != Player.COPS:
            self.display.print_error("Not detectives' turn")
            return False
        
        detective_moves = []
        
        # Collect moves from all detectives
        for i, detective_pos in enumerate(self.game.game_state.cop_positions):
            print(f"\n--- Detective {i+1} (Position {detective_pos}) ---")
            
            # Check if detective can move with pending moves consideration
            if isinstance(self.game, ScotlandYardGame):
                available_moves = self.game.get_valid_moves(Player.COPS, detective_pos, pending_moves=detective_moves)
            else:
                available_moves = self.game.get_valid_moves(Player.COPS, detective_pos)
            
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
    
    def _make_detective_move(self, position: int) -> bool:
        """Handle detective move"""
        # Find detective ID
        try:
            detective_id = self.game.game_state.cop_positions.index(position)
        except ValueError:
            self.display.print_error(f"No detective at position {position}")
            return False
        
        # Show available moves
        available_moves = self.display.print_available_moves(self.game, Player.COPS, position)
        if not available_moves:
            return False
        
        while True:
            move_input = input(f"\n🎮 Detective {detective_id + 1} move: ").strip()
            
            if move_input.lower() in ['help', 'h']:
                self.display.print_input_help()
                continue
            
            if move_input.lower() in ['quit', 'exit', 'q']:
                return False
            
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
            
            # Create move list for all detectives (others stay in place)
            detective_moves = []
            for i in range(self.game.num_cops):
                if i == detective_id:
                    detective_moves.append((dest, transport))
                else:
                    current_pos = self.game.game_state.cop_positions[i]
                    detective_moves.append((current_pos, None))  # Stay in place
            
            # Make move
            success = self.game.make_move(detective_moves=detective_moves)
            
            if success:
                self.display.print_move_result(True, f"Detective {detective_id + 1} moved to position {dest}")
                return True
            else:
                self.display.print_error("Move failed")
                continue
    
    def _make_mr_x_move(self) -> bool:
        """Handle Mr. X move"""
        available_moves = self.display.print_available_moves(self.game, Player.ROBBER)
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
                        available_moves = self.display.print_available_moves(self.game, Player.ROBBER)
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
        from .agent import make_random_move
        
        success = make_random_move(self.game)
        
        if success:
            if player == Player.COPS:
                self.display.print_move_result(True, "AI Detectives moved")
            else:
                self.display.print_move_result(True, "AI Mr. X moved")
        else:
            self.display.print_error("AI move failed")
        
        return success
    
    def reset_turn_state(self):
        """Reset turn-specific state"""
        self.current_detective = 0
        self.double_move_first = False


class GameSetup:
    """Handles game initialization with predefined scenarios"""
    
    @staticmethod
    def create_test_game(num_detectives: int = 2) -> ScotlandYardGame:
        """Create a test Scotland Yard game (small map)"""
        from cops_and_robbers.examples.example_games import create_test_scotland_yard_game
        return create_test_scotland_yard_game(num_detectives)
    
    @staticmethod
    def create_full_game(num_detectives: int = 3) -> ScotlandYardGame:
        """Create a full Scotland Yard game (full map)"""
        from cops_and_robbers.examples.example_games import create_scotlandYard_game
        return create_scotlandYard_game(num_detectives)
    
    @staticmethod
    def initialize_test_positions(game: ScotlandYardGame) -> None:
        """Initialize with predefined test positions"""
        detective_positions = [1, 2]
        mr_x_position = 9
        game.initialize_scotland_yard_game(detective_positions, mr_x_position)
    
    @staticmethod
    def initialize_full_positions(game: ScotlandYardGame, num_detectives: int) -> None:
        """Initialize with predefined full game positions"""
        # Predefined starting positions for full Scotland Yard
        detective_start_options = [1, 13, 26, 34, 50, 53, 91, 94, 103, 112, 138, 141, 155, 174, 197]
        mr_x_start_options = [35, 45, 51, 71, 78, 104, 106, 127, 132, 166, 170, 172]
        
        detective_positions = detective_start_options[:num_detectives]
        mr_x_position = random.choice(mr_x_start_options)
        
        game.initialize_scotland_yard_game(detective_positions, mr_x_position)


def get_game_mode() -> Tuple[str, str, int]:
    """Get game configuration from user"""
    print("\n🎮 GAME SETUP")
    print("=" * 50)
    
    # Map size
    print("\n📍 Choose map size:")
    print("1. Test map (10 nodes) - Good for learning")
    print("2. Full map (199 nodes) - Complete Scotland Yard")
    
    while True:
        map_choice = input("Map size (1-2): ").strip()
        if map_choice == "1":
            map_size = "test"
            num_detectives = 2
            break
        elif map_choice == "2":
            map_size = "full"
            # Number of detectives for full game
            print("\n👮 Number of detectives:")
            print("2. Two detectives (easier)")
            print("3. Three detectives (standard)")
            print("4. Four detectives (harder)")
            
            while True:
                det_choice = input("Number of detectives (2-4): ").strip()
                if det_choice in ["2", "3", "4"]:
                    num_detectives = int(det_choice)
                    break
                print("❌ Please enter 2, 3, or 4")
            break
        else:
            print("❌ Please enter 1 or 2")
    
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
    
    return map_size, play_mode, num_detectives


def get_verbosity_level() -> int:
    """Get display verbosity level from user"""
    print(f"\n📊 Choose display verbosity:")
    print("1. Basic - Just positions and turn")
    print("2. Standard - + Available moves and tickets")
    print("3. Detailed - + Move history and transport details")
    print("4. Debug - + All internal game state")
    
    while True:
        verb_choice = input("Verbosity level (1-4): ").strip()
        if verb_choice in ["1", "2", "3", "4"]:
            return int(verb_choice)
        print("❌ Please enter 1, 2, 3, or 4")
