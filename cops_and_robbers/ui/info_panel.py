"""
Information Panel Component

Displays game state, turn information, available moves, and tickets for Scotland Yard game.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Set
from ..core.game import Game, Player, TransportType, TicketType, ScotlandYardGame


class InfoPanel:
    """Information display panel for game state and details"""
    
    def __init__(self, parent_frame: tk.Frame, game: Game):
        """Initialize information panel"""
        self.parent_frame = parent_frame
        self.game = game
        
        # Create main container with scrollable content
        self._setup_scrollable_frame()
        
        # Create information sections
        self._create_game_info_section()
        self._create_turn_info_section()
        self._create_moves_section()
        self._create_tickets_section()
        self._create_history_section()
        
        # Initial update
        self.update_all()
    
    def _setup_scrollable_frame(self):
        """Setup scrollable frame for the information panel"""
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.parent_frame, width=360)
        self.scrollbar = ttk.Scrollbar(self.parent_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _create_game_info_section(self):
        """Create game information section"""
        self.game_info_frame = ttk.LabelFrame(self.scrollable_frame, text="Game Information", padding="10")
        self.game_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Game state labels
        self.game_type_label = ttk.Label(self.game_info_frame, text="Game Type: Standard")
        self.game_type_label.pack(anchor=tk.W)
        
        self.turn_count_label = ttk.Label(self.game_info_frame, text="Turn: 0")
        self.turn_count_label.pack(anchor=tk.W)
        
        self.game_status_label = ttk.Label(self.game_info_frame, text="Status: Setting up")
        self.game_status_label.pack(anchor=tk.W)
        
        self.winner_label = ttk.Label(self.game_info_frame, text="")
        self.winner_label.pack(anchor=tk.W)
        
        # Graph information
        ttk.Separator(self.game_info_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.nodes_label = ttk.Label(self.game_info_frame, text=f"Nodes: {self.game.graph.number_of_nodes()}")
        self.nodes_label.pack(anchor=tk.W)
        
        self.edges_label = ttk.Label(self.game_info_frame, text=f"Edges: {self.game.graph.number_of_edges()}")
        self.edges_label.pack(anchor=tk.W)
        
        self.cops_label = ttk.Label(self.game_info_frame, text=f"Detectives: {self.game.num_cops}")
        self.cops_label.pack(anchor=tk.W)
    
    def _create_turn_info_section(self):
        """Create current turn information section"""
        self.turn_info_frame = ttk.LabelFrame(self.scrollable_frame, text="Current Turn", padding="10")
        self.turn_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_player_label = ttk.Label(self.turn_info_frame, text="Player: -", font=("Arial", 10, "bold"))
        self.current_player_label.pack(anchor=tk.W)
        
        self.current_action_label = ttk.Label(self.turn_info_frame, text="Action: -")
        self.current_action_label.pack(anchor=tk.W)
        
        # Player positions
        ttk.Separator(self.turn_info_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.positions_frame = ttk.Frame(self.turn_info_frame)
        self.positions_frame.pack(fill=tk.X)
        
        self.cop_positions_label = ttk.Label(self.positions_frame, text="Detectives: -")
        self.cop_positions_label.pack(anchor=tk.W)
        
        self.robber_position_label = ttk.Label(self.positions_frame, text="Mr. X: -")
        self.robber_position_label.pack(anchor=tk.W)
    
    def _create_moves_section(self):
        """Create available moves section"""
        self.moves_frame = ttk.LabelFrame(self.scrollable_frame, text="Available Moves", padding="10")
        self.moves_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Moves display with scrollable listbox
        self.moves_listbox = tk.Listbox(self.moves_frame, height=6, font=("Courier", 9))
        moves_scrollbar = ttk.Scrollbar(self.moves_frame, orient="vertical")
        
        self.moves_listbox.config(yscrollcommand=moves_scrollbar.set)
        moves_scrollbar.config(command=self.moves_listbox.yview)
        
        self.moves_listbox.pack(side="left", fill="both", expand=True)
        moves_scrollbar.pack(side="right", fill="y")
        
        # Selected moves display
        self.selected_moves_label = ttk.Label(self.moves_frame, text="Selected: None")
        self.selected_moves_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_tickets_section(self):
        """Create tickets section for Scotland Yard games"""
        self.tickets_frame = ttk.LabelFrame(self.scrollable_frame, text="Tickets", padding="10")
        self.tickets_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Detective tickets
        self.detective_tickets_frame = ttk.Frame(self.tickets_frame)
        self.detective_tickets_frame.pack(fill=tk.X)
        
        ttk.Label(self.detective_tickets_frame, text="Detective Tickets:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        self.detective_tickets_labels = {}
        
        # Mr. X tickets
        ttk.Separator(self.tickets_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        self.mrx_tickets_frame = ttk.Frame(self.tickets_frame)
        self.mrx_tickets_frame.pack(fill=tk.X)
        
        ttk.Label(self.mrx_tickets_frame, text="Mr. X Tickets:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        self.mrx_tickets_labels = {}
        
        # Special moves
        self.special_moves_frame = ttk.Frame(self.tickets_frame)
        self.special_moves_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.double_move_label = ttk.Label(self.special_moves_frame, text="Double Moves: -")
        self.double_move_label.pack(anchor=tk.W)
        
        # Initially hide if not Scotland Yard game
        if not isinstance(self.game, ScotlandYardGame):
            self.tickets_frame.pack_forget()
    
    def _create_history_section(self):
        """Create game history section"""
        self.history_frame = ttk.LabelFrame(self.scrollable_frame, text="Move History", padding="10")
        self.history_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # History display with scrollable text
        self.history_text = tk.Text(self.history_frame, height=8, width=40, wrap=tk.WORD, font=("Courier", 8))
        history_scrollbar = ttk.Scrollbar(self.history_frame, orient="vertical")
        
        self.history_text.config(yscrollcommand=history_scrollbar.set)
        history_scrollbar.config(command=self.history_text.yview)
        
        self.history_text.pack(side="left", fill="both", expand=True)
        history_scrollbar.pack(side="right", fill="y")
        
        # Make read-only
        self.history_text.config(state=tk.DISABLED)
    
    def update_all(self):
        """Update all information sections"""
        self.update_game_info()
        self.update_turn_info()
        self.update_moves()
        self.update_tickets()
        self.update_history()
    
    def update_game_info(self):
        """Update game information section"""
        # Determine game type
        game_type = "Scotland Yard" if isinstance(self.game, ScotlandYardGame) else "Standard Cops & Robbers"
        self.game_type_label.config(text=f"Game Type: {game_type}")
        
        # Turn count
        turn_count = 0
        if hasattr(self.game, 'game_state') and self.game.game_state:
            turn_count = self.game.game_state.turn_count
        self.turn_count_label.config(text=f"Turn: {turn_count}")
        
        # Game status
        if not hasattr(self.game, 'game_state') or not self.game.game_state:
            status = "Setting up"
        elif self.game.is_game_over():
            winner = self.game.get_winner()
            if winner:
                winner_name = "Detectives" if winner == Player.COPS else "Mr. X"
                status = f"Game Over - {winner_name} Win!"
            else:
                status = "Game Over - Draw"
        else:
            status = "In Progress"
        
        self.game_status_label.config(text=f"Status: {status}")
        
        # Winner
        if self.game.is_game_over():
            winner = self.game.get_winner()
            if winner:
                winner_name = "Detectives" if winner == Player.COPS else "Mr. X"
                self.winner_label.config(text=f"Winner: {winner_name}", foreground="green")
            else:
                self.winner_label.config(text="Result: Draw", foreground="orange")
        else:
            self.winner_label.config(text="")
    
    def update_turn_info(self):
        """Update current turn information"""
        if not hasattr(self.game, 'game_state') or not self.game.game_state:
            self.current_player_label.config(text="Player: -")
            self.current_action_label.config(text="Action: Setting up game")
            self.cop_positions_label.config(text="Detectives: -")
            self.robber_position_label.config(text="Mr. X: -")
            return
        
        state = self.game.game_state
        
        # Current player
        player_name = "Detectives" if state.turn == Player.COPS else "Mr. X"
        self.current_player_label.config(text=f"Player: {player_name}")
        
        # Current action
        if self.game.is_game_over():
            action = "Game finished"
        elif state.turn == Player.COPS:
            action = "Select moves for detectives"
        else:
            action = "Select move for Mr. X"
        self.current_action_label.config(text=f"Action: {action}")
        
        # Positions
        cop_pos_text = ", ".join(map(str, sorted(state.cop_positions))) if state.cop_positions else "-"
        self.cop_positions_label.config(text=f"Detectives: {cop_pos_text}")
        
        # Mr. X position (with visibility rules)
        if isinstance(self.game, ScotlandYardGame):
            if state.mr_x_visible:
                robber_text = str(state.robber_position)
            else:
                robber_text = "Hidden"
        else:
            robber_text = str(state.robber_position) if state.robber_position is not None else "-"
        
        self.robber_position_label.config(text=f"Mr. X: {robber_text}")
    
    def update_moves(self, available_moves: Set[int] = None, selected_moves: List[int] = None):
        """Update available moves section"""
        self.moves_listbox.delete(0, tk.END)
        
        if available_moves:
            for move in sorted(available_moves):
                self.moves_listbox.insert(tk.END, f"Node {move}")
        else:
            self.moves_listbox.insert(tk.END, "No moves available")
        
        # Selected moves
        if selected_moves:
            selected_text = ", ".join(map(str, selected_moves))
            self.selected_moves_label.config(text=f"Selected: {selected_text}")
        else:
            self.selected_moves_label.config(text="Selected: None")
    
    def update_tickets(self):
        """Update tickets section for Scotland Yard games"""
        if not isinstance(self.game, ScotlandYardGame):
            return
        
        if not hasattr(self.game, 'game_state') or not self.game.game_state:
            return
        
        state = self.game.game_state
        
        # Clear existing ticket labels
        for widget in self.detective_tickets_frame.winfo_children():
            if isinstance(widget, ttk.Label) and widget != self.detective_tickets_frame.winfo_children()[0]:
                widget.destroy()
        
        for widget in self.mrx_tickets_frame.winfo_children():
            if isinstance(widget, ttk.Label) and widget != self.mrx_tickets_frame.winfo_children()[0]:
                widget.destroy()
        
        # Detective tickets
        if state.detective_tickets:
            for detective_id, tickets in state.detective_tickets.items():
                detective_label = ttk.Label(self.detective_tickets_frame, text=f"Detective {detective_id}:")
                detective_label.pack(anchor=tk.W, padx=(10, 0))
                
                for ticket_type, count in tickets.items():
                    ticket_label = ttk.Label(self.detective_tickets_frame, 
                                           text=f"  {ticket_type.value}: {count}")
                    ticket_label.pack(anchor=tk.W, padx=(20, 0))
        
        # Mr. X tickets
        if state.mr_x_tickets:
            for ticket_type, count in state.mr_x_tickets.items():
                ticket_label = ttk.Label(self.mrx_tickets_frame, 
                                       text=f"  {ticket_type.value}: {count}")
                ticket_label.pack(anchor=tk.W, padx=(10, 0))
        
        # Double moves
        double_moves = state.mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0) if state.mr_x_tickets else 0
        double_status = " (ACTIVE)" if state.double_move_active else ""
        self.double_move_label.config(text=f"Double Moves: {double_moves}{double_status}")
    
    def update_history(self):
        """Update move history"""
        if not hasattr(self.game, 'game_history') or not self.game.game_history:
            return
        
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        for i, state in enumerate(self.game.game_history):
            turn_info = f"Turn {i + 1}: "
            
            if state.turn == Player.COPS:
                positions = ", ".join(map(str, sorted(state.cop_positions)))
                turn_info += f"Detectives moved to [{positions}]\n"
            else:
                if isinstance(self.game, ScotlandYardGame) and not state.mr_x_visible:
                    turn_info += f"Mr. X moved (hidden)\n"
                else:
                    turn_info += f"Mr. X moved to {state.robber_position}\n"
            
            self.history_text.insert(tk.END, turn_info)
        
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)
    
    def set_game(self, game: Game):
        """Update the game reference"""
        self.game = game
        
        # Show/hide tickets section based on game type
        if isinstance(game, ScotlandYardGame):
            self.tickets_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.tickets_frame.pack_forget()
        
        # Update node/edge counts
        self.nodes_label.config(text=f"Nodes: {game.graph.number_of_nodes()}")
        self.edges_label.config(text=f"Edges: {game.graph.number_of_edges()}")
        self.cops_label.config(text=f"Detectives: {game.num_cops}")
        
        self.update_all()
    
    def add_message(self, message: str, level: str = "info"):
        """Add a message to the history"""
        self.history_text.config(state=tk.NORMAL)
        
        # Add timestamp and format message
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.history_text.insert(tk.END, formatted_message)
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)
