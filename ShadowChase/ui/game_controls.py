import tkinter as tk
from tkinter import ttk, messagebox
from .ui_components import StyledButton, InfoDisplay, EnhancedTurnDisplay, EnhancedMovesDisplay
from ..core.game import Player, ShadowChaseGame, TicketType

class GameControls:
    """Handles game control UI and logic"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.controls_section = None
        self.mrx_section = None
        self.turn_display = None
        self.moves_display = None
        self.tickets_display = None
        
        # Mr. X special moves state
        self.use_black_ticket = tk.BooleanVar()
        self.double_move_requested = False
        self.MrX_selections = []
        
        # Transport selection state
        self.pending_move = None  # (node, available_transports)
        self.selected_transport = None
        
    def create_controls_section(self, parent):
        """Create the main game controls section"""
        self.controls_section = ttk.LabelFrame(parent, text="🎮 Game Controls")
        
        # Configure label style
        style = ttk.Style()
        style.configure("Controls.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.controls_section.configure(style="Controls.TLabelframe")
        
        button_frame = ttk.Frame(self.controls_section)
        button_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # Human player buttons
        self.move_button = StyledButton(button_frame, "📤 Make Move", 
                                       command=self.send_move, 
                                       style_type="primary", state=tk.DISABLED)
        self.move_button.pack(fill=tk.X, pady=3)
        
        self.skip_button = StyledButton(button_frame, "⏭️ Skip Turn (No Moves)",
                                      command=self.skip_turn, state=tk.DISABLED)
        self.skip_button.pack(fill=tk.X, pady=3)
        
        # AI player button
        self.ai_continue_button = StyledButton(button_frame, "▶️ Continue (AI Move)", 
                                             command=self.make_ai_continue, 
                                             style_type="success", state=tk.DISABLED)
        self.ai_continue_button.pack(fill=tk.X, pady=3)

        return self.controls_section
    
    def make_ai_continue(self):
        """Handle AI continue button - make AI move and update display"""
        success = self.visualizer.make_ai_move()
        
        if success:
            # Reset UI state after AI move
            self.visualizer.selected_positions = []
            self.visualizer.detective_selections = []
            self.MrX_selections = []
            self.visualizer.current_detective_index = 0
            self.visualizer.selected_nodes = []
            
            # Update heuristics if they're enabled and available
            if (hasattr(self.visualizer, 'heuristics') and self.visualizer.heuristics and 
                hasattr(self.visualizer.setup_controls, 'get_heuristics_enabled') and
                self.visualizer.setup_controls.get_heuristics_enabled()):
                self.visualizer.heuristics.update_game_state(self.visualizer.game)
            
            # Update UI
            self.visualizer.update_ui_visibility()
            self.visualizer.draw_graph()
            
            # Check for game over
            if self.visualizer.game.is_game_over():
                winner = self.visualizer.game.get_winner()
                winner_name = winner.value.title() if winner else "No one"
                
                # Auto-save the completed game
                self.visualizer.auto_save_completed_game()
                
                messagebox.showinfo("🎉 Game Over", f"{winner_name} wins!")
        else:
            messagebox.showerror("AI Error", "AI failed to make a move")
    
    def update_button_visibility(self):
        """Update which buttons are visible based on current player type"""
        if not self.visualizer.game.game_state:
            # No game state - hide all action buttons
            self.move_button.pack_forget()
            self.skip_button.pack_forget()
            self.ai_continue_button.pack_forget()
            return
        
        is_ai_turn = self.visualizer.is_current_player_ai()
        
        if is_ai_turn:
            # AI turn - show continue button, hide human buttons
            self.move_button.pack_forget()
            self.skip_button.pack_forget()
            self.ai_continue_button.pack(fill=tk.X, pady=3)
            self.ai_continue_button.config(state=tk.NORMAL)
        else:
            # Human turn - show human buttons, hide AI button
            self.ai_continue_button.pack_forget()
            self.move_button.pack(fill=tk.X, pady=3)
            self.skip_button.pack(fill=tk.X, pady=3)
            
            # Update human button states based on selections
            self.update_human_button_states()
    
    def update_human_button_states(self):
        """Update the state of human player buttons"""
        if not self.visualizer.game.game_state:
            return
        
        current_player = self.visualizer.game.game_state.turn
        
        if current_player == Player.DETECTIVES:
            # Enable move button when all detectives have made selections
            if len(self.visualizer.detective_selections) == self.visualizer.game.num_detectives:
                self.move_button.config(state=tk.NORMAL)
            else:
                self.move_button.config(state=tk.DISABLED)
        else:
            # Mr. X turn
            if self.MrX_selections or self.visualizer.selected_positions:
                self.move_button.config(state=tk.NORMAL)
            else:
                self.move_button.config(state=tk.DISABLED)
    
    def create_mrx_controls_section(self, parent):
        """Create Mr. X specific controls section"""
        self.mrx_section = ttk.LabelFrame(parent, text="🕵️ Mr. X Special Moves")
        
        # Configure label style
        style = ttk.Style()
        style.configure("MrX.TLabelframe.Label", anchor="w", font=('Arial', 11, 'bold'))
        self.mrx_section.configure(style="MrX.TLabelframe")
        
        controls_frame = ttk.Frame(self.mrx_section)
        controls_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.double_move_button = StyledButton(controls_frame, "⚡ Use Double Move",
                                             command=self.toggle_double_move, 
                                             state=tk.DISABLED)
        self.double_move_button.pack(fill=tk.X, pady=3)
        
        # # Add note about transport selection
        # info_label = ttk.Label(controls_frame, 
        #                      text="💡 Transport type is selected when clicking destination",
        #                      font=('Arial', 8, 'italic'))
        # info_label.pack(pady=5)
        
        return self.mrx_section
    
    def create_turn_display(self, parent):
        """Create the enhanced current turn information display"""
        self.turn_display = EnhancedTurnDisplay(parent, "📋 Current Turn")
        return self.turn_display
    
    def create_moves_display(self, parent):
        """Create the enhanced available moves display"""
        self.moves_display = EnhancedMovesDisplay(parent, "🎯 Available Moves")
        return self.moves_display
    
    def create_tickets_display(self, parent):
        """Create the tickets display"""
        self.tickets_display = InfoDisplay(parent, "🎫 Tickets", height=8)
        return self.tickets_display
   
    
    def update_turn_display(self):
        """Update current turn information using enhanced display"""
        if not self.turn_display:
            return
            
        if self.visualizer.setup_mode:
            # For setup mode, the enhanced component will handle this
            self.turn_display.update_display(None, self.visualizer.game)
            return
        
        if not self.visualizer.game.game_state:
            return
        
        # Update button visibility first
        self.update_button_visibility()
        
        # Use the enhanced component's update method
        self.turn_display.update_display(
            self.visualizer.game.game_state,
            self.visualizer.game,
            current_detective_index=getattr(self.visualizer, 'current_detective_index', 0),
            detective_selections=getattr(self.visualizer, 'detective_selections', []),
            is_ai_turn=self.visualizer.is_current_player_ai(),
            double_move_requested=getattr(self, 'double_move_requested', False)
        )
        
        # Update instructions
        self._update_instructions()
    
    def update_moves_display(self):
        """Update available moves display using enhanced component"""
        if not self.moves_display:
            return
            
        if self.visualizer.setup_mode or not self.visualizer.game.game_state:
            self.moves_display.update_moves("")
            return
        
        # Get moves text from the visualizer if available
        if hasattr(self.visualizer, 'available_moves_text'):
            moves_text = self.visualizer.available_moves_text
        else:
            moves_text = "Available moves will be shown here..."
        
        self.moves_display.update_moves(moves_text)
        
        # Update skip button state based on available moves
        if not self.visualizer.current_player_moves:
            # No moves available - enable skip button
            if hasattr(self, 'skip_button'):
                self.skip_button.config(state=tk.NORMAL)
        else:
            # Moves available - disable skip button
            if hasattr(self, 'skip_button'):
                self.skip_button.config(state=tk.DISABLED)
        
        # Update Mr. X controls if it's his turn
        if (self.visualizer.game.game_state and 
            self.visualizer.game.game_state.turn == Player.MRX):
            self.update_mrx_controls()
    
    
    def show_transport_selection(self, node, available_transports, can_use_black=False):
        """Show transport selection buttons for the given node and transports"""
        self.pending_move = (node, available_transports)
        self.selected_transport = None
        
        # Use the enhanced turn display to show transport buttons
        if self.turn_display:
            self.turn_display.show_transport_selection(
                available_transports, 
                node, 
                self.on_transport_selected,
                can_use_black
            )
        
        # Update instructions
        self._update_instructions()
    
    def on_transport_selected(self, transport):
        """Handle transport selection from buttons"""
        self.selected_transport = transport
        
        if self.pending_move:
            node, _ = self.pending_move
            
            # Add the move with selected transport
            current_player = self.visualizer.game.game_state.turn
            
            if current_player == Player.DETECTIVES:
                # Add detective move
                self.visualizer.detective_selections.append((node, transport))
                self.visualizer.current_detective_index += 1
                
                # Clear pending move
                self.pending_move = None
                self.selected_transport = None
                self.turn_display.hide_transport_selection()
                
                # Check if all detectives have moved
                if len(self.visualizer.detective_selections) == self.visualizer.game.num_detectives:
                    self.move_button.config(state=tk.NORMAL)
                
            else:  # Mr. X turn
                # Add Mr. X move
                self.MrX_selections.append((node, transport))
                
                # Clear pending move
                self.pending_move = None
                self.selected_transport = None
                self.turn_display.hide_transport_selection()
                
                # Enable send button
                self.move_button.config(state=tk.NORMAL)
            
            # Update display and redraw graph
            self.update_turn_display()
            self.visualizer.draw_graph()
    
    def cancel_transport_selection(self):
        """Cancel current transport selection"""
        self.pending_move = None
        self.selected_transport = None
        if self.turn_display:
            self.turn_display.hide_transport_selection()
        self._update_instructions()
    
    def _update_instructions(self):
        """Update instruction text based on current state"""
        if not self.turn_display:
            return
        
        if self.pending_move:
            node, _ = self.pending_move
            instruction_text = f"Select transport type for destination {node}"
        elif self.visualizer.game.game_state:
            current_player = self.visualizer.game.game_state.turn
            if current_player == Player.DETECTIVES:
                if len(self.visualizer.detective_selections) < self.visualizer.game.num_detectives:
                    instruction_text = "Click on destination nodes for detectives"
                else:
                    instruction_text = "All detectives ready - click Make Move"
            else:
                if not self.MrX_selections:
                    instruction_text = "Click on destination node for Mr. X"
                else:
                    instruction_text = "Move selected - click Make Move"
        else:
            instruction_text = "Select moves for current player"
        
        # Update the instruction label in turn display
        if hasattr(self.turn_display, 'instructions'):
            self.turn_display.instructions.config(text=instruction_text)
    
    def update_mrx_controls(self):
        """Updates the state of Mr. X's special move controls."""
        if (self.visualizer.game.game_state and 
            self.visualizer.game.game_state.turn == Player.MRX):
            MrX_tickets = self.visualizer.game.get_MrX_tickets()
            
            # Don't allow double move activation if already in progress
            double_move_available = (MrX_tickets.get(TicketType.DOUBLE_MOVE, 0) > 0 and 
                                   not self.visualizer.game.game_state.double_move_active)
            
            if double_move_available:
                self.double_move_button.config(state=tk.NORMAL)
            else:
                self.double_move_button.config(state=tk.DISABLED)
                self.double_move_requested = False
                self.double_move_button.configure(text="⚡ Use Double Move")
        else:
            self.double_move_button.config(state=tk.DISABLED)
            self.double_move_requested = False
            self.double_move_button.configure(text="⚡ Use Double Move")
    
    def toggle_double_move(self):
        """Request double move for the next Mr. X turn."""
        self.double_move_requested = not self.double_move_requested
        if self.double_move_requested:
            self.double_move_button.configure(text="⚡ Cancel Double Move")
        else:
            self.double_move_button.configure(text="⚡ Use Double Move")
    
    
    def send_move(self):
        """Send the selected moves to the game engine."""
        if ((self.visualizer.game.game_state.turn == Player.DETECTIVES and 
             len(self.visualizer.detective_selections) != self.visualizer.game.num_detectives) or 
            (self.visualizer.game.game_state.turn == Player.MRX and 
             not self.MrX_selections and not self.visualizer.selected_positions)):
            messagebox.showwarning("Invalid Selection", "A move must be selected for all players.")
            return
    
        try:
            is_shadow_chase = isinstance(self.visualizer.game, ShadowChaseGame)
            success = False
    
            if self.visualizer.game.game_state.turn == Player.DETECTIVES:
                if is_shadow_chase:
                    success = self.visualizer.game.make_move(detective_moves=self.visualizer.detective_selections)
                else:
                    success = self.visualizer.game.make_move(new_positions=self.visualizer.detective_selections)
            else:  # MrX's turn
                if is_shadow_chase:
                    # For Shadow Chase, check if this is a double move scenario
                    if self.double_move_requested and not self.visualizer.game.game_state.double_move_active:
                        # Starting a double move - make first move
                        success = self.visualizer.game.make_move(MrX_moves=self.MrX_selections,
                                                               use_double_move=True)
                        if success:
                            # Reset selections for second move but keep double move button state
                            self.MrX_selections = []
                            self.visualizer.selected_nodes = []
                            self.move_button.config(state=tk.DISABLED)
                            self.use_black_ticket.set(False)
                            # Don't reset double_move_requested yet - we're still in double move
                            self.visualizer.draw_graph()
                            return
                    else:
                        # Regular move or second move of double move
                        success = self.visualizer.game.make_move(MrX_moves=self.MrX_selections,
                                                               use_double_move=False)
                        if success and self.visualizer.game.game_state.double_move_active:
                            # This was the second move of a double move, reset the flag
                            self.double_move_requested = False
                            self.double_move_button.configure(text="⚡ Use Double Move")
                else:
                    success = self.visualizer.game.make_move(new_MrX_pos=self.visualizer.selected_positions[0])
    
            if not success:
                messagebox.showerror("Invalid Move", "The move was rejected by the game engine.")
    
            # Reset UI state after move attempt
            self.visualizer.selected_positions = []
            self.visualizer.detective_selections = []
            self.MrX_selections = []
            self.visualizer.current_detective_index = 0
            self.visualizer.selected_nodes = []
            self.move_button.config(state=tk.DISABLED)
            
            # Only reset double move flag if not in the middle of a double move
            if not self.visualizer.game.game_state.double_move_active:
                self.double_move_requested = False
                self.double_move_button.configure(text="⚡ Use Double Move")
            
            self.use_black_ticket.set(False)
            self.visualizer.update_ui_visibility()
            
            # Update heuristics if they're enabled and available
            if (hasattr(self.visualizer, 'heuristics') and self.visualizer.heuristics and 
                hasattr(self.visualizer.setup_controls, 'get_heuristics_enabled') and
                self.visualizer.setup_controls.get_heuristics_enabled()):
                self.visualizer.heuristics.update_game_state(self.visualizer.game)
            
            self.visualizer.draw_graph()
    
            # Check for game over
            if self.visualizer.game.is_game_over():
                winner = self.visualizer.game.get_winner()
                winner_name = winner.value.title() if winner else "No one"
                
                # Auto-save the completed game
                self.visualizer.auto_save_completed_game()
                
                messagebox.showinfo("🎉 Game Over", f"{winner_name} wins!")
    
        except Exception as e:
            messagebox.showerror("Move Error", f"An error occurred while making the move: {str(e)}")
            # Reset UI state on error
            self.visualizer.selected_positions = []
            self.visualizer.detective_selections = []
            self.MrX_selections = []
            self.visualizer.current_detective_index = 0
            self.visualizer.selected_nodes = []
            self.move_button.config(state=tk.DISABLED)
            self.double_move_requested = False
            self.use_black_ticket.set(False)
            self.visualizer.update_ui_visibility()
            self.visualizer.draw_graph()

    def skip_turn(self):
        """Handles a detective skipping their turn when they have no moves."""
        if (self.visualizer.game.game_state.turn == Player.DETECTIVES and 
            self.visualizer.current_detective_index < self.visualizer.game.num_detectives):
            detective_pos = self.visualizer.game.game_state.detective_positions[self.visualizer.current_detective_index]
            
            # Verify there are no moves
            if (detective_pos in self.visualizer.current_player_moves and 
                not self.visualizer.current_player_moves[detective_pos]):
                self.visualizer.detective_selections.append((detective_pos, None)) # Append a "stay" move
                self.visualizer.current_detective_index += 1
                
                if len(self.visualizer.detective_selections) == self.visualizer.game.num_detectives:
                    self.move_button.config(state=tk.NORMAL)
                
                self.visualizer.draw_graph()
            else:
                messagebox.showwarning("Invalid Action", "Skip turn is only for players with no available moves.")
        
        self.skip_button.config(state=tk.DISABLED)
