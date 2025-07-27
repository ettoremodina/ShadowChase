import tkinter as tk
from tkinter import ttk, messagebox
from .ui_components import StyledButton, InfoDisplay
from ..core.game import Player, ScotlandYardGame, TransportType, TicketType

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
        self.mr_x_selections = []
        
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
        self.move_button = StyledButton(button_frame, "✅ Make Move", 
                                       command=self.make_manual_move, 
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
            self.mr_x_selections = []
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
            if self.mr_x_selections or self.visualizer.selected_positions:
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
        """Create the current turn information display"""
        self.turn_display = InfoDisplay(parent, "📋 Current Turn", height=4)
        return self.turn_display
    
    def create_moves_display(self, parent):
        """Create the available moves display"""
        self.moves_display = InfoDisplay(parent, "🎯 Available Moves", height=6)
        return self.moves_display
    
    def create_tickets_display(self, parent):
        """Create the tickets display"""
        self.tickets_display = InfoDisplay(parent, "🎫 Tickets", height=8)
        return self.tickets_display
   
    
    def update_turn_display(self):
        """Update current turn information"""
        if not self.turn_display:
            return
            
        if self.visualizer.setup_mode:
            self.turn_display.set_text("Setup Phase - Click nodes to select positions")
            return
        
        if not self.visualizer.game.game_state:
            return
        
        # Update button visibility first
        self.update_button_visibility()
        
        is_scotland_yard = isinstance(self.visualizer.game, ScotlandYardGame)
        current_player = self.visualizer.game.game_state.turn
        is_ai_turn = self.visualizer.is_current_player_ai()
        
        turn_text = ""
        
        if current_player == Player.DETECTIVES:
            player_type = "🤖 AI" if is_ai_turn else "👤 Human"
            if is_scotland_yard:
                if not is_ai_turn and self.visualizer.current_detective_index < self.visualizer.game.num_detectives:
                    det_pos = self.visualizer.game.game_state.detective_positions[self.visualizer.current_detective_index]
                    turn_text = f"🕵️ DETECTIVE {self.visualizer.current_detective_index + 1}'S TURN ({player_type})\n"
                    turn_text += f"📍 Moving from position {det_pos}\n"
                    turn_text += f"📊 Progress: {len(self.visualizer.detective_selections)}/{self.visualizer.game.num_detectives}"
                else:
                    turn_text = f"🕵️ DETECTIVES' TURN ({player_type})\n"
                    if is_ai_turn:
                        turn_text += "🤖 Click 'Continue' to let AI make moves"
                    else:
                        turn_text += "✅ All detectives selected - make move"
            else:
                if not is_ai_turn and self.visualizer.current_detective_index < self.visualizer.game.num_detectives:
                    detective_pos = self.visualizer.game.game_state.detective_positions[self.visualizer.current_detective_index]
                    turn_text = f"👮 detective {self.visualizer.current_detective_index + 1}'S TURN ({player_type})\n"
                    turn_text += f"📍 Moving from position {detective_pos}\n"
                    turn_text += f"📊 Progress: {len(self.visualizer.detective_selections)}/{self.visualizer.game.num_detectives}"
                else:
                    turn_text = f"👮 detectives' TURN ({player_type})\n"
                    if is_ai_turn:
                        turn_text += "🤖 Click 'Continue' to let AI make moves"
                    else:
                        turn_text += "✅ All detectives selected - make move"
        else:
            player_type = "🤖 AI" if is_ai_turn else "👤 Human"
            if is_scotland_yard:
                double_status = ""
                if not is_ai_turn:
                    if self.double_move_requested:
                        double_status = " (DOUBLE MOVE REQUESTED)"
                    elif self.visualizer.game.game_state.double_move_active:
                        double_status = " (SECOND MOVE)"
                    
                turn_text = f"🕵️‍♂️ MR. X'S TURN ({player_type}){double_status}\n"
                if is_ai_turn:
                    turn_text += "🤖 Click 'Continue' to let AI make move"
                else:
                    turn_text += "📍 Select new position"
            else:
                turn_text = f"🏃 MrX'S TURN ({player_type})\n"
                if is_ai_turn:
                    turn_text += "🤖 Click 'Continue' to let AI make move"
                else:
                    turn_text += "📍 Select new position"
        
        self.turn_display.set_text(turn_text)
    
    def update_moves_display(self):
        """Update available moves display"""
        if not self.moves_display:
            return
            
        if self.visualizer.setup_mode or not self.visualizer.game.game_state:
            return
        
        is_scotland_yard = isinstance(self.visualizer.game, ScotlandYardGame)
        
        if not self.visualizer.current_player_moves:
            self.moves_display.set_text("❌ No available moves")
            return
        
        moves_text = ""
        
        # Show current detective's moves or MrX/Mr. X moves
        if (self.visualizer.game.game_state.turn == Player.DETECTIVES and 
            self.visualizer.current_detective_index < self.visualizer.game.num_detectives):
            
            detective_pos = self.visualizer.game.game_state.detective_positions[self.visualizer.current_detective_index]
            if detective_pos in self.visualizer.current_player_moves:
                moves = self.visualizer.current_player_moves[detective_pos]
                if not moves:
                    moves_text = "⚠️ No available moves. Click 'Skip Turn'."
                    self.skip_button.config(state=tk.NORMAL)
                else:
                    self.skip_button.config(state=tk.DISABLED)

                player_name = f"Detective {self.visualizer.current_detective_index + 1}" if is_scotland_yard else f"detective {self.visualizer.current_detective_index + 1}"
                moves_text += f"🎯 {player_name} from position {detective_pos}:\n"
                for target_pos, transports in moves.items():
                    if is_scotland_yard:
                        transport_names = []
                        for t in transports:
                            if t == 1: transport_names.append("🚕 Taxi")
                            elif t == 2: transport_names.append("🚌 Bus") 
                            elif t == 3: transport_names.append("🚇 Underground")
                            elif t == 4: transport_names.append("⚫ Black")
                        moves_text += f"  ➡️ {target_pos} ({', '.join(transport_names)})\n"
                    else:
                        moves_text += f"  ➡️ {target_pos}\n"
        else:
            self.skip_button.config(state=tk.DISABLED)
            self.update_mrx_controls()
            # MrX/Mr. X moves
            for source_pos, moves in self.visualizer.current_player_moves.items():
                player_name = "🕵️‍♂️ Mr. X" if is_scotland_yard else "🏃 MrX"
                moves_text += f"🎯 {player_name} from position {source_pos}:\n"
                for target_pos, transports in moves.items():
                    if is_scotland_yard:
                        transport_names = []
                        for t in sorted(set(transports)):
                            if t == 1: transport_names.append("🚕 Taxi")
                            elif t == 2: transport_names.append("🚌 Bus") 
                            elif t == 3: transport_names.append("🚇 Underground")
                            elif t == 4: transport_names.append("⚫ Black")
                        moves_text += f"  ➡️ {target_pos} ({', '.join(transport_names)})\n"
                    else:
                        moves_text += f"  ➡️ {target_pos}\n"
        
        self.moves_display.set_text(moves_text)
    
    
    def toggle_double_move(self):
        """Request double move for the next Mr. X turn."""
        self.double_move_requested = not self.double_move_requested
        if self.double_move_requested:
            self.double_move_button.configure(text="⚡ Cancel Double Move")
        else:
            self.double_move_button.configure(text="⚡ Use Double Move")
    
    def update_mrx_controls(self):
        """Updates the state of Mr. X's special move controls."""
        if (self.visualizer.game.game_state and 
            self.visualizer.game.game_state.turn == Player.MRX):
            mr_x_tickets = self.visualizer.game.get_mr_x_tickets()
            
            # Don't allow double move activation if already in progress
            double_move_available = (mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) > 0 and 
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
    
    def make_manual_move(self):
        """Make a manual move by sending selected moves to the game object."""
        if ((self.visualizer.game.game_state.turn == Player.DETECTIVES and 
             len(self.visualizer.detective_selections) != self.visualizer.game.num_detectives) or 
            (self.visualizer.game.game_state.turn == Player.MRX and 
             not self.mr_x_selections and not self.visualizer.selected_positions)):
            messagebox.showwarning("Invalid Selection", "A move must be selected for all players.")
            return
    
        try:
            is_scotland_yard = isinstance(self.visualizer.game, ScotlandYardGame)
            success = False
    
            if self.visualizer.game.game_state.turn == Player.DETECTIVES:
                if is_scotland_yard:
                    success = self.visualizer.game.make_move(detective_moves=self.visualizer.detective_selections)
                else:
                    success = self.visualizer.game.make_move(new_positions=self.visualizer.detective_selections)
            else:  # MrX's turn
                if is_scotland_yard:
                    # For Scotland Yard, check if this is a double move scenario
                    if self.double_move_requested and not self.visualizer.game.game_state.double_move_active:
                        # Starting a double move - make first move
                        success = self.visualizer.game.make_move(mr_x_moves=self.mr_x_selections,
                                                               use_double_move=True)
                        if success:
                            # Reset selections for second move but keep double move button state
                            self.mr_x_selections = []
                            self.visualizer.selected_nodes = []
                            self.move_button.config(state=tk.DISABLED)
                            self.use_black_ticket.set(False)
                            # Don't reset double_move_requested yet - we're still in double move
                            self.visualizer.draw_graph()
                            return
                    else:
                        # Regular move or second move of double move
                        success = self.visualizer.game.make_move(mr_x_moves=self.mr_x_selections,
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
            self.mr_x_selections = []
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
            self.mr_x_selections = []
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
