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
        
        self.move_button = StyledButton(button_frame, "✅ Make Move", 
                                       command=self.make_manual_move, 
                                       style_type="primary", state=tk.DISABLED)
        self.move_button.pack(fill=tk.X, pady=3)
        
        self.skip_button = StyledButton(button_frame, "⏭️ Skip Turn (No Moves)",
                                      command=self.skip_turn, state=tk.DISABLED)
        self.skip_button.pack(fill=tk.X, pady=3)

        return self.controls_section
    
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
        
        is_scotland_yard = isinstance(self.visualizer.game, ScotlandYardGame)
        current_player = self.visualizer.game.game_state.turn
        
        turn_text = ""
        
        if current_player == Player.COPS:
            if is_scotland_yard:
                if self.visualizer.current_cop_index < self.visualizer.game.num_cops:
                    det_pos = self.visualizer.game.game_state.cop_positions[self.visualizer.current_cop_index]
                    turn_text = f"🕵️ DETECTIVE {self.visualizer.current_cop_index + 1}'S TURN\n"
                    turn_text += f"📍 Moving from position {det_pos}\n"
                    turn_text += f"📊 Progress: {len(self.visualizer.cop_selections)}/{self.visualizer.game.num_cops}"
                else:
                    turn_text = "✅ All detectives selected - make move"
            else:
                if self.visualizer.current_cop_index < self.visualizer.game.num_cops:
                    cop_pos = self.visualizer.game.game_state.cop_positions[self.visualizer.current_cop_index]
                    turn_text = f"👮 COP {self.visualizer.current_cop_index + 1}'S TURN\n"
                    turn_text += f"📍 Moving from position {cop_pos}\n"
                    turn_text += f"📊 Progress: {len(self.visualizer.cop_selections)}/{self.visualizer.game.num_cops}"
                else:
                    turn_text = "✅ All cops selected - make move"
        else:
            if is_scotland_yard:
                double_status = ""
                if self.double_move_requested:
                    double_status = " (DOUBLE MOVE REQUESTED)"
                elif self.visualizer.game.game_state.double_move_active:
                    double_status = " (SECOND MOVE)"
                    
                turn_text = f"🕵️‍♂️ MR. X'S TURN{double_status}\n📍 Select new position"
            else:
                turn_text = "🏃 ROBBER'S TURN\n📍 Select new position"
        
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
        
        # Show current cop's moves or robber/Mr. X moves
        if (self.visualizer.game.game_state.turn == Player.COPS and 
            self.visualizer.current_cop_index < self.visualizer.game.num_cops):
            
            cop_pos = self.visualizer.game.game_state.cop_positions[self.visualizer.current_cop_index]
            if cop_pos in self.visualizer.current_player_moves:
                moves = self.visualizer.current_player_moves[cop_pos]
                if not moves:
                    moves_text = "⚠️ No available moves. Click 'Skip Turn'."
                    self.skip_button.config(state=tk.NORMAL)
                else:
                    self.skip_button.config(state=tk.DISABLED)

                player_name = f"Detective {self.visualizer.current_cop_index + 1}" if is_scotland_yard else f"Cop {self.visualizer.current_cop_index + 1}"
                moves_text += f"🎯 {player_name} from position {cop_pos}:\n"
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
            # Robber/Mr. X moves
            for source_pos, moves in self.visualizer.current_player_moves.items():
                player_name = "🕵️‍♂️ Mr. X" if is_scotland_yard else "🏃 Robber"
                moves_text += f"🎯 {player_name} from position {source_pos}:\n"
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
        
        self.moves_display.set_text(moves_text)
    
    def update_tickets_display(self):
        """Update the tickets display for Scotland Yard games"""
        if not self.tickets_display:
            return
            
        if not self.visualizer.game.game_state:
            return
        
        is_scotland_yard = isinstance(self.visualizer.game, ScotlandYardGame)
        if not is_scotland_yard:
            self.tickets_display.set_text("ℹ️ Not a Scotland Yard game")
            return
        
        tickets_text = "🕵️ DETECTIVE TICKETS:\n"
        for i in range(self.visualizer.game.num_cops):
            tickets = self.visualizer.game.get_detective_tickets(i)
            pos = self.visualizer.game.game_state.cop_positions[i]
            tickets_text += f"Det. {i+1} (pos {pos}):\n"
            for ticket_type, count in tickets.items():
                icon = {"taxi": "🚕", "bus": "🚌", "underground": "🚇"}.get(ticket_type.value, "🎫")
                tickets_text += f"  {icon} {ticket_type.value}: {count}\n"
        
        # Show Mr. X tickets
        mr_x_tickets = self.visualizer.game.get_mr_x_tickets()
        tickets_text += "\n🕵️‍♂️ MR. X TICKETS:\n"
        for ticket_type, count in mr_x_tickets.items():
            icon = {"taxi": "🚕", "bus": "🚌", "underground": "🚇", 
                   "black": "⚫", "double_move": "⚡"}.get(ticket_type.value, "🎫")
            tickets_text += f"  {icon} {ticket_type.value}: {count}\n"
        
        self.tickets_display.set_text(tickets_text)
    
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
            self.visualizer.game.game_state.turn == Player.ROBBER):
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
        if ((self.visualizer.game.game_state.turn == Player.COPS and 
             len(self.visualizer.cop_selections) != self.visualizer.game.num_cops) or 
            (self.visualizer.game.game_state.turn == Player.ROBBER and 
             not self.mr_x_selections and not self.visualizer.selected_positions)):
            messagebox.showwarning("Invalid Selection", "A move must be selected for all players.")
            return
    
        try:
            is_scotland_yard = isinstance(self.visualizer.game, ScotlandYardGame)
            success = False
    
            if self.visualizer.game.game_state.turn == Player.COPS:
                if is_scotland_yard:
                    success = self.visualizer.game.make_move(detective_moves=self.visualizer.cop_selections)
                else:
                    success = self.visualizer.game.make_move(new_positions=self.visualizer.cop_selections)
            else:  # Robber's turn
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
                    success = self.visualizer.game.make_move(new_robber_pos=self.visualizer.selected_positions[0])
    
            if not success:
                messagebox.showerror("Invalid Move", "The move was rejected by the game engine.")
    
            # Reset UI state after move attempt
            self.visualizer.selected_positions = []
            self.visualizer.cop_selections = []
            self.mr_x_selections = []
            self.visualizer.current_cop_index = 0
            self.visualizer.selected_nodes = []
            self.move_button.config(state=tk.DISABLED)
            
            # Only reset double move flag if not in the middle of a double move
            if not self.visualizer.game.game_state.double_move_active:
                self.double_move_requested = False
                self.double_move_button.configure(text="⚡ Use Double Move")
            
            self.use_black_ticket.set(False)
            self.visualizer.update_ui_visibility()
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
            self.visualizer.cop_selections = []
            self.mr_x_selections = []
            self.visualizer.current_cop_index = 0
            self.visualizer.selected_nodes = []
            self.move_button.config(state=tk.DISABLED)
            self.double_move_requested = False
            self.use_black_ticket.set(False)
            self.visualizer.update_ui_visibility()
            self.visualizer.draw_graph()

    def skip_turn(self):
        """Handles a detective skipping their turn when they have no moves."""
        if (self.visualizer.game.game_state.turn == Player.COPS and 
            self.visualizer.current_cop_index < self.visualizer.game.num_cops):
            cop_pos = self.visualizer.game.game_state.cop_positions[self.visualizer.current_cop_index]
            
            # Verify there are no moves
            if (cop_pos in self.visualizer.current_player_moves and 
                not self.visualizer.current_player_moves[cop_pos]):
                self.visualizer.cop_selections.append((cop_pos, None)) # Append a "stay" move
                self.visualizer.current_cop_index += 1
                
                if len(self.visualizer.cop_selections) == self.visualizer.game.num_cops:
                    self.move_button.config(state=tk.NORMAL)
                
                self.visualizer.draw_graph()
            else:
                messagebox.showwarning("Invalid Action", "Skip turn is only for players with no available moves.")
        
        self.skip_button.config(state=tk.DISABLED)
