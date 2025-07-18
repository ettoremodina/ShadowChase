"""
Display utilities for simple terminal-based Scotland Yard gameplay.
Provides clean, customizable output formatting without graph visualization.
"""
from typing import Dict, List, Tuple, Optional, Set
from cops_and_robbers.core.game import ScotlandYardGame, Player, TicketType, TransportType


class VerbosityLevel:
    """Verbosity level constants"""
    BASIC = 1      # Basic game state (positions, turn)
    MOVES = 2      # + Available moves and tickets  
    DETAILED = 3   # + Move history and detailed transport info
    DEBUG = 4      # + All internal game mechanics


class GameDisplay:
    """Handles all game display formatting with configurable verbosity"""
    
    def __init__(self, verbosity: int = VerbosityLevel.MOVES):
        self.verbosity = verbosity
        self.move_history = []
        
        # Display symbols
        self.symbols = {
            'detective': '🕵️',
            'mr_x': '🕵️‍♂️',
            'taxi': '🚕',
            'bus': '🚌', 
            'underground': '🚇',
            'black': '⚫',
            'double_move': '⚡',
            'visible': '👁️',
            'hidden': '❓'
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_separator(self, char='=', length=60):
        """Print a separator line"""
        print(char * length)
    
    def print_title(self, title: str):
        """Print a formatted title"""
        self.print_separator()
        print(f"  {title.upper()}")
        self.print_separator()
    
    def print_game_state(self, game: ScotlandYardGame):
        """Print current game state based on verbosity level"""
        if not game.game_state:
            print("❌ Game not initialized")
            return
        
        state = game.get_state_representation()
        
        # Basic level - always show
        print(f"\n🎯 TURN {state['turn_count']} - {state['turn'].upper()}'S TURN")
        
        # Show positions
        self._print_positions(game)
        
        # Show tickets if verbosity >= MOVES
        if self.verbosity >= VerbosityLevel.MOVES:
            self._print_tickets(game)
        
        # Show game status
        if state['game_over']:
            winner = state.get('winner', 'Unknown')
            print(f"\n🏆 GAME OVER! Winner: {winner.upper()}")
        
        # Show move history if verbosity >= DETAILED
        if self.verbosity >= VerbosityLevel.DETAILED:
            self._print_move_history(game)
        
        print()
    
    def _print_positions(self, game: ScotlandYardGame):
        """Print player positions"""
        state = game.game_state
        
        # Detective positions
        print(f"\n{self.symbols['detective']} DETECTIVES:")
        for i, pos in enumerate(state.cop_positions):
            print(f"  Detective {i+1}: Position {pos}")
        
        # Mr. X position
        print(f"\n{self.symbols['mr_x']} MR. X:")
        if state.mr_x_visible:
            print(f"  Position: {state.robber_position} {self.symbols['visible']}")
        else:
            print(f"  Position: Hidden {self.symbols['hidden']}")
    
    def _print_tickets(self, game: ScotlandYardGame):
        """Print ticket information"""
        print(f"\n🎫 TICKETS:")
        
        # Detective tickets
        for i in range(game.num_cops):
            tickets = game.get_detective_tickets(i)
            print(f"\n  Detective {i+1}:")
            for ticket_type, count in tickets.items():
                icon = self.symbols.get(ticket_type.value, '🎫')
                print(f"    {icon} {ticket_type.value.capitalize()}: {count}")
        
        # Mr. X tickets
        mr_x_tickets = game.get_mr_x_tickets()
        print(f"\n  Mr. X:")
        for ticket_type, count in mr_x_tickets.items():
            icon = self.symbols.get(ticket_type.value, '🎫')
            print(f"    {icon} {ticket_type.value.capitalize()}: {count}")
    
    def _print_move_history(self, game: ScotlandYardGame):
        """Print recent move history"""
        if not hasattr(game.game_state, 'mr_x_moves_log'):
            return
            
        moves_log = game.game_state.mr_x_moves_log
        if not moves_log:
            return
        
        print(f"\n📜 RECENT MR. X MOVES:")
        # Show last 5 moves
        recent_moves = moves_log[-5:]
        for i, (pos, transport) in enumerate(recent_moves, 1):
            icon = self.symbols.get(transport.name.lower(), '🎫')
            print(f"  Move {len(moves_log) - len(recent_moves) + i}: {icon} → Position {pos}")
    
    def print_available_moves(self, game: ScotlandYardGame, player: Player, position: Optional[int] = None):
        """Print available moves for a player"""
        if player == Player.COPS and position is not None:
            if isinstance(game, ScotlandYardGame):
                moves = game.get_valid_moves(Player.COPS, position)
            else:
                moves = game.get_valid_moves(Player.COPS, position)
            print(f"\n🎯 AVAILABLE MOVES for Detective at position {position}:")
            
            if not moves:
                print("  ❌ No valid moves available!")
                return moves
            
            # Handle both Scotland Yard (tuples) and regular games (integers)
            if isinstance(game, ScotlandYardGame):
                # Group moves by destination
                move_dict = {}
                for dest, transport in moves:
                    if dest not in move_dict:
                        move_dict[dest] = []
                    move_dict[dest].append(transport)
                
                for dest in sorted(move_dict.keys()):
                    transports = move_dict[dest]
                    transport_icons = [self.symbols.get(t.name.lower(), '🎫') for t in transports]
                    print(f"  → Position {dest}: {' '.join(transport_icons)}")
                    
                    if self.verbosity >= VerbosityLevel.DETAILED:
                        transport_names = [t.name.capitalize() for t in transports]
                        print(f"    ({', '.join(transport_names)})")
            else:
                # Regular game - just show destinations
                for dest in sorted(moves):
                    print(f"  → Position {dest}")
        
        elif player == Player.ROBBER:
            if isinstance(game, ScotlandYardGame):
                moves = game.get_valid_moves(Player.ROBBER)
            else:
                moves = game.get_valid_moves(Player.ROBBER)
            print(f"\n🎯 AVAILABLE MOVES for Mr. X:")
            
            if not moves:
                print("  ❌ No valid moves available!")
                return moves
            
            # Handle both Scotland Yard (tuples) and regular games (integers)
            if isinstance(game, ScotlandYardGame):
                # Group moves by destination
                move_dict = {}
                for dest, transport in moves:
                    if dest not in move_dict:
                        move_dict[dest] = []
                    move_dict[dest].append(transport)
                
                for dest in sorted(move_dict.keys()):
                    transports = move_dict[dest]
                    transport_icons = [self.symbols.get(t.name.lower(), '🎫') for t in transports]
                    print(f"  → Position {dest}: {' '.join(transport_icons)}")
                    
                    if self.verbosity >= VerbosityLevel.DETAILED:
                        transport_names = [t.name.capitalize() for t in transports]
                        print(f"    ({', '.join(transport_names)})")
                
                # Show special moves
                if game.can_use_double_move():
                    print(f"  {self.symbols['double_move']} Double move available!")
            else:
                # Regular game - just show destinations
                for dest in sorted(moves):
                    print(f"  → Position {dest}")
        
        return moves
    
    def print_move_result(self, success: bool, move_description: str):
        """Print result of a move attempt"""
        if success:
            print(f"✅ {move_description}")
        else:
            print(f"❌ Failed: {move_description}")
    
    def print_error(self, message: str):
        """Print an error message"""
        print(f"❌ ERROR: {message}")
    
    def print_info(self, message: str):
        """Print an info message"""
        print(f"ℹ️  {message}")
    
    def print_input_help(self):
        """Print help for input format"""
        print("\n📋 INPUT HELP:")
        print("  • Enter destination number (e.g., '45')")
        print("  • Specify transport: '45 taxi', '45 bus', '45 underground'")
        print("  • Use black ticket: 'B45' or '45 black'")
        print("  • Double move: 'DD' then enter two moves")
        print("  • Type 'help' for this message")
        print("  • Type 'quit' to exit")
    
    def print_debug_info(self, game: ScotlandYardGame):
        """Print debug information (verbosity level 4)"""
        if self.verbosity < VerbosityLevel.DEBUG:
            return
        
        print("\n🔧 DEBUG INFO:")
        state = game.game_state
        print(f"  Double move active: {state.double_move_active}")
        print(f"  Reveal turns: {game.reveal_turns}")
        print(f"  Next reveal: {min([t for t in game.reveal_turns if t > state.turn_count], default='None')}")
        print(f"  Game history length: {len(game.game_history)}")


def format_transport_input(user_input: str) -> Tuple[Optional[int], Optional[TransportType], bool, bool]:
    """
    Parse user input for moves.
    
    Returns:
        (destination, transport_type, use_black_ticket, is_double_move)
    """
    user_input = user_input.strip().lower()
    
    # Handle special commands
    if user_input in ['help', 'h']:
        return None, None, False, False
    
    if user_input in ['quit', 'exit', 'q']:
        return None, None, False, False
    
    if user_input in ['dd', 'double']:
        return None, None, False, True
    
    # Handle black ticket format (B45)
    if user_input.startswith('b') and user_input[1:].isdigit():
        dest = int(user_input[1:])
        return dest, None, True, False
    
    # Split input
    parts = user_input.split()
    
    if not parts:
        return None, None, False, False
    
    # Get destination
    try:
        dest = int(parts[0])
    except ValueError:
        return None, None, False, False
    
    # Get transport type if specified
    transport = None
    use_black = False
    
    if len(parts) > 1:
        transport_str = parts[1].lower()
        if transport_str in ['black', 'b']:
            use_black = True
        elif transport_str == 'taxi':
            transport = TransportType.TAXI
        elif transport_str == 'bus':
            transport = TransportType.BUS
        elif transport_str in ['underground', 'tube', 'metro']:
            transport = TransportType.UNDERGROUND
    
    return dest, transport, use_black, False


def get_user_choice(prompt: str, valid_choices: List[str]) -> str:
    """Get user choice with validation"""
    while True:
        choice = input(f"{prompt} ({'/'.join(valid_choices)}): ").strip().lower()
        if choice in valid_choices:
            return choice
        print(f"❌ Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def get_user_move(display: GameDisplay) -> str:
    """Get move input from user with help"""
    while True:
        move_input = input("\n🎮 Enter your move (or 'help'): ").strip()
        if move_input:
            return move_input
        print("❌ Please enter a move")
