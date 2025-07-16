"""
Gameplay Controller Component

Handles the active gameplay phase with move selection and validation.
"""

import tkinter as tk
from tkinter import messagebox
from typing import Set, List, Optional, Callable, Tuple, Dict
from ..core.game import Game, Player, TransportType, TicketType, ScotlandYardGame


class GameplayController:
    """Controller for active gameplay phase"""
    
    def __init__(self, game: Game):
        """Initialize gameplay controller"""
        self.game = game
        self.is_scotland_yard = isinstance(game, ScotlandYardGame)
        
        # Move selection state
        self.pending_cop_moves: List[int] = []
        self.pending_robber_move: Optional[int] = None
        self.current_cop_index = 0
        self.move_in_progress = False
        
        # Scotland Yard specific state
        self.use_black_ticket = False
        self.double_move_active = False
        self.double_move_requested = False
        self.pending_transport_type: Optional[TransportType] = None
        
        # Callbacks
        self.on_move_completed: Optional[Callable] = None
        self.on_turn_changed: Optional[Callable] = None
        self.on_game_over: Optional[Callable] = None
        self.on_moves_updated: Optional[Callable] = None
        self.on_status_update: Optional[Callable[[str], None]] = None
        
    def handle_node_click(self, node: int) -> bool:
        """Handle node click during gameplay"""
        if not self.game.game_state or self.game.is_game_over():
            return False
        
        current_player = self.game.game_state.turn
        
        if current_player == Player.COPS:
            return self._handle_cop_move_selection(node)
        else:
            return self._handle_robber_move_selection(node)
    
    def _handle_cop_move_selection(self, node: int) -> bool:
        """Handle cop move selection"""
        if self.current_cop_index >= len(self.game.game_state.cop_positions):
            return False
        
        current_cop_pos = self.game.game_state.cop_positions[self.current_cop_index]
        valid_moves = self.game.get_valid_moves(Player.COPS, current_cop_pos)
        
        if node not in valid_moves:
            self._show_message(f"Invalid move for Detective {self.current_cop_index + 1}")
            return False
        
        # Store the move
        if len(self.pending_cop_moves) <= self.current_cop_index:
            self.pending_cop_moves.extend([None] * (self.current_cop_index + 1 - len(self.pending_cop_moves)))
        
        self.pending_cop_moves[self.current_cop_index] = node
        
        # Move to next cop or complete selection
        self.current_cop_index += 1
        
        if self.current_cop_index >= self.game.num_cops:
            # All cops have moves selected
            self._update_status("All detective moves selected. Click 'Make Move' to execute.")
            self.move_in_progress = True
        else:
            self._update_status(f"Select move for Detective {self.current_cop_index + 1}")
        
        self._notify_moves_updated()
        return True
    
    def _handle_robber_move_selection(self, node: int) -> bool:
        """Handle robber move selection"""
        if self.is_scotland_yard:
            return self._handle_scotland_yard_robber_move(node)
        else:
            return self._handle_standard_robber_move(node)
    
    def _handle_standard_robber_move(self, node: int) -> bool:
        """Handle standard robber move selection"""
        valid_moves = self.game.get_valid_moves(Player.ROBBER)
        
        if node not in valid_moves:
            self._show_message("Invalid move for Mr. X")
            return False
        
        self.pending_robber_move = node
        self.move_in_progress = True
        self._update_status("Mr. X move selected. Click 'Make Move' to execute.")
        self._notify_moves_updated()
        return True
    
    def _handle_scotland_yard_robber_move(self, node: int) -> bool:
        """Handle Scotland Yard Mr. X move selection"""
        # Get valid moves with transport types
        valid_moves = self.game.get_valid_moves(Player.ROBBER)
        # Check if the node is reachable
        reachable_nodes = {move[0] if isinstance(move, tuple) else move for move in valid_moves}
        if node not in reachable_nodes:
            self._show_message("Invalid move for Mr. X")
            return False

        # Find available transport types for this move
        current_pos = self.game.game_state.robber_position
        available_transports = []

        for move in valid_moves:
            if isinstance(move, tuple) and move[0] == node:
                available_transports.append(move[1])
            elif move == node:
                available_transports.append(TransportType.TAXI)  # Default

        if not available_transports:
            return False

        # Select transport type
        if len(available_transports) == 1:
            selected_transport = available_transports[0]
        else:
            # Show transport selection dialog
            selected_transport = self._show_transport_selection_dialog(available_transports)
            if not selected_transport:
                return False

        self.pending_robber_move = node
        self.pending_transport_type = selected_transport
        self.move_in_progress = True

        transport_name = selected_transport.name.lower()
        self._update_status(f"Mr. X move selected (using {transport_name}). Click 'Make Move' to execute.")
        self._notify_moves_updated()
        return True

    def _show_transport_selection_dialog(self, available_transports: List[TransportType]) -> Optional[TransportType]:
        """Show dialog for transport type selection"""
        # Create a simple selection dialog
        root = tk.Toplevel()
        root.title("Select Transport")
        root.geometry("300x200")
        root.resizable(False, False)
        
        selected_transport = None
        
        tk.Label(root, text="Select transport type:", font=("Arial", 12)).pack(pady=10)
        
        var = tk.StringVar()
        for transport in available_transports:
            transport_name = transport.name.lower().replace('_', ' ').title()
            tk.Radiobutton(root, text=transport_name, variable=var, 
                          value=transport.name).pack(anchor=tk.W, padx=50)
        
        if available_transports:
            var.set(available_transports[0].name)
        
        def confirm():
            nonlocal selected_transport
            try:
                selected_transport = TransportType[var.get()]
                root.destroy()
            except KeyError:
                pass
        
        def cancel():
            root.destroy()
        
        tk.Button(root, text="Confirm", command=confirm).pack(side=tk.LEFT, padx=20, pady=20)
        tk.Button(root, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=20, pady=20)
        
        root.wait_window()
        return selected_transport
    
    def execute_move(self) -> bool:
        """Execute the pending moves"""
        if not self.move_in_progress:
            self._show_message("No moves selected")
            return False
        
        try:
            current_player = self.game.game_state.turn
            
            if current_player == Player.COPS:
                success = self._execute_cop_moves()
            else:
                success = self._execute_robber_move()
            
            if success:
                self._reset_move_state()
                self._check_game_over()
                
                if self.on_move_completed:
                    self.on_move_completed()
                
                if not self.game.is_game_over() and self.on_turn_changed:
                    self.on_turn_changed()
                
                return True
            else:
                self._show_message("Move execution failed")
                return False
                
        except Exception as e:
            self._show_message(f"Error executing move: {str(e)}")
            return False
    
    def _execute_cop_moves(self) -> bool:
        """Execute cop moves"""
        if len(self.pending_cop_moves) != self.game.num_cops:
            return False
        
        # Validate all moves
        for i, move in enumerate(self.pending_cop_moves):
            if move is None:
                return False
            
            current_pos = self.game.game_state.cop_positions[i]
            valid_moves = self.game.get_valid_moves(Player.COPS, current_pos)
            
            if move not in valid_moves:
                return False
        
        # Execute the moves
        if self.is_scotland_yard:
            # Scotland Yard specific move execution
            return self._execute_scotland_yard_cop_moves()
        else:
            # Standard move execution
            return self.game.make_move(new_positions=self.pending_cop_moves)
    
    def _execute_scotland_yard_cop_moves(self) -> bool:
        """Execute Scotland Yard cop moves with ticket validation"""
        # This would need to be implemented based on the Scotland Yard game logic
        # For now, use standard move execution
        return self.game.make_move(new_positions=self.pending_cop_moves)
    
    def _execute_robber_move(self) -> bool:
        """Execute robber move"""
        if self.pending_robber_move is None:
            return False
        
        if self.is_scotland_yard and self.pending_transport_type:
            # Scotland Yard specific move execution with transport type
            return self._execute_scotland_yard_robber_move()
        else:
            # Standard move execution
            return self.game.make_move(new_robber_pos=self.pending_robber_move)
    
    def _execute_scotland_yard_robber_move(self) -> bool:
        """Execute Scotland Yard Mr. X move with transport type"""
        # This would need to interface with the Scotland Yard game logic
        # For now, use standard move execution
        return self.game.make_move(new_robber_pos=self.pending_robber_move)
    
    def skip_turn(self) -> bool:
        """Skip current turn if no moves available"""
        if not self.game.game_state or self.game.is_game_over():
            return False
        
        current_player = self.game.game_state.turn
        
        # Check if there are actually no moves available
        if current_player == Player.COPS:
            can_move = self._any_cop_can_move()
        else:
            valid_moves = self.game.get_valid_moves(Player.ROBBER)
            can_move = len(valid_moves) > 0
        
        if can_move:
            self._show_message("Cannot skip turn - moves are available")
            return False
        
        try:
            # Force turn change by making a "no-op" move
            self.game.game_state.turn = Player.ROBBER if current_player == Player.COPS else Player.COPS
            self.game.game_state.turn_count += 1
            
            if self.on_turn_changed:
                self.on_turn_changed()
            
            return True
            
        except Exception as e:
            self._show_message(f"Error skipping turn: {str(e)}")
            return False
    
    def _any_cop_can_move(self) -> bool:
        """Check if any cop can move"""
        for i, pos in enumerate(self.game.game_state.cop_positions):
            valid_moves = self.game.get_valid_moves(Player.COPS, pos)
            if len(valid_moves) > 1:  # More than just staying in place
                return True
        return False
    
    def get_available_moves(self) -> Set[int]:
        """Get available moves for current player"""
        if not self.game.game_state or self.game.is_game_over():
            return set()
        
        current_player = self.game.game_state.turn
        
        if current_player == Player.COPS:
            if self.current_cop_index < len(self.game.game_state.cop_positions):
                current_pos = self.game.game_state.cop_positions[self.current_cop_index]
                moves = self.game.get_valid_moves(Player.COPS, current_pos)
                return set(moves) if isinstance(moves, (list, set)) else {moves}
        else:
            moves = self.game.get_valid_moves(Player.ROBBER)
            if isinstance(moves, set) and moves and isinstance(next(iter(moves)), tuple):
                # Scotland Yard moves with transport types
                return {move[0] for move in moves}
            return set(moves) if isinstance(moves, (list, set)) else {moves}
        
        return set()
    
    def get_pending_moves(self) -> Dict:
        """Get currently pending moves"""
        return {
            'cop_moves': self.pending_cop_moves.copy(),
            'robber_move': self.pending_robber_move,
            'current_cop_index': self.current_cop_index,
            'transport_type': self.pending_transport_type
        }
    
    def is_move_ready(self) -> bool:
        """Check if move is ready to execute"""
        return self.move_in_progress
    
    def can_skip_turn(self) -> bool:
        """Check if current turn can be skipped"""
        if not self.game.game_state or self.game.is_game_over():
            return False
        
        current_player = self.game.game_state.turn
        
        if current_player == Player.COPS:
            return not self._any_cop_can_move()
        else:
            valid_moves = self.game.get_valid_moves(Player.ROBBER)
            return len(valid_moves) == 0
    
    def reset_move_selection(self):
        """Reset current move selection"""
        self._reset_move_state()
        self._notify_moves_updated()
        self._update_status(self._get_current_status_message())
    
    def _reset_move_state(self):
        """Reset move selection state"""
        self.pending_cop_moves.clear()
        self.pending_robber_move = None
        self.current_cop_index = 0
        self.move_in_progress = False
        self.pending_transport_type = None
    
    def _check_game_over(self):
        """Check if game is over and notify"""
        if self.game.is_game_over() and self.on_game_over:
            self.on_game_over()
    
    def _get_current_status_message(self) -> str:
        """Get current status message"""
        if not self.game.game_state:
            return "Game not started"
        
        if self.game.is_game_over():
            winner = self.game.get_winner()
            if winner:
                winner_name = "Detectives" if winner == Player.COPS else "Mr. X"
                return f"Game Over - {winner_name} Win!"
            return "Game Over - Draw"
        
        current_player = self.game.game_state.turn
        
        if current_player == Player.COPS:
            if self.move_in_progress:
                return "Detective moves selected. Click 'Make Move' to execute."
            elif self.current_cop_index == 0:
                return f"Select move for Detective 1"
            else:
                return f"Select move for Detective {self.current_cop_index + 1}"
        else:
            if self.move_in_progress:
                return "Mr. X move selected. Click 'Make Move' to execute."
            else:
                return "Select move for Mr. X"
    
    def _update_status(self, message: str):
        """Update status message"""
        if self.on_status_update:
            self.on_status_update(message)
    
    def _notify_moves_updated(self):
        """Notify that moves have been updated"""
        if self.on_moves_updated:
            self.on_moves_updated()
    
    def _show_message(self, message: str):
        """Show message to user"""
        messagebox.showwarning("Gameplay", message)
    
    def set_use_black_ticket(self, use_black: bool):
        """Set whether to use black ticket for Mr. X moves"""
        self.use_black_ticket = use_black
    
    def activate_double_move(self):
        """Activate double move for Mr. X"""
        if not self.is_scotland_yard:
            return False
        
        # This would need to be implemented based on Scotland Yard rules
        self.double_move_active = True
        self._update_status("Double move activated - Mr. X will move twice")
        return True
    
    # Callback setters
    def set_move_completed_callback(self, callback: Callable):
        """Set callback for move completion"""
        self.on_move_completed = callback
    
    def set_turn_changed_callback(self, callback: Callable):
        """Set callback for turn changes"""
        self.on_turn_changed = callback
    
    def set_game_over_callback(self, callback: Callable):
        """Set callback for game over"""
        self.on_game_over = callback
    
    def set_moves_updated_callback(self, callback: Callable):
        """Set callback for moves updates"""
        self.on_moves_updated = callback
    
    def set_status_update_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates"""
        self.on_status_update = callback
