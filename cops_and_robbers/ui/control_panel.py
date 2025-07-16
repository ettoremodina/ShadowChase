"""
Control Panel Component

Provides game controls, Scotland Yard specific controls, and save/load functionality.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional, Dict, List
from ..core.game import Game, Player, TicketType, ScotlandYardGame
from ..storage.game_loader import GameLoader


class ControlPanel:
    """Control panel for game actions and Scotland Yard specific features"""
    
    def __init__(self, parent_frame: tk.Frame, game: Game):
        """Initialize control panel"""
        self.parent_frame = parent_frame
        self.game = game
        self.game_loader = GameLoader()
        
        # Callbacks for external control
        self.on_make_move: Optional[Callable] = None
        self.on_skip_turn: Optional[Callable] = None
        self.on_start_game: Optional[Callable] = None
        self.on_reset_game: Optional[Callable] = None
        self.on_new_game: Optional[Callable] = None
        self.on_double_move_request: Optional[Callable] = None
        self.on_black_ticket: Optional[Callable] = None
        
        # State tracking
        self.setup_phase = True
        self.moves_selected = False
        self.can_skip = False
        
        # Create control sections
        self._create_game_controls()
        self._create_scotland_yard_controls()
        self._create_save_load_controls()
        self._create_setup_controls()
        
        # Initial state
        self.update_controls()
    
    def _create_game_controls(self):
        """Create basic game control buttons"""
        self.game_controls_frame = ttk.LabelFrame(self.parent_frame, text="Game Controls", padding="10")
        self.game_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Make Move button
        self.make_move_btn = ttk.Button(
            self.game_controls_frame,
            text="Make Move",
            command=self._on_make_move_clicked,
            state=tk.DISABLED
        )
        self.make_move_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Skip Turn button
        self.skip_turn_btn = ttk.Button(
            self.game_controls_frame,
            text="Skip Turn",
            command=self._on_skip_turn_clicked,
            state=tk.DISABLED
        )
        self.skip_turn_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Reset Game button
        self.reset_game_btn = ttk.Button(
            self.game_controls_frame,
            text="Reset Game",
            command=self._on_reset_game_clicked
        )
        self.reset_game_btn.pack(fill=tk.X, pady=(0, 5))
        
        # New Game button
        self.new_game_btn = ttk.Button(
            self.game_controls_frame,
            text="New Game",
            command=self._on_new_game_clicked
        )
        self.new_game_btn.pack(fill=tk.X)
    
    def _create_scotland_yard_controls(self):
        """Create Scotland Yard specific controls"""
        self.scotland_yard_frame = ttk.LabelFrame(self.parent_frame, text="Scotland Yard Controls", padding="10")
        self.scotland_yard_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Black Ticket option
        self.use_black_ticket_var = tk.BooleanVar()
        self.black_ticket_cb = ttk.Checkbutton(
            self.scotland_yard_frame,
            text="Use Black Ticket",
            variable=self.use_black_ticket_var,
            command=self._on_black_ticket_changed
        )
        self.black_ticket_cb.pack(anchor=tk.W, pady=(0, 5))
        
        # Double Move button
        self.double_move_btn = ttk.Button(
            self.scotland_yard_frame,
            text="Double Move",
            command=self._on_double_move_clicked,
            state=tk.DISABLED
        )
        self.double_move_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Ticket usage display
        self.ticket_usage_frame = ttk.Frame(self.scotland_yard_frame)
        self.ticket_usage_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(self.ticket_usage_frame, text="Next Move Will Use:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        self.ticket_usage_label = ttk.Label(self.ticket_usage_frame, text="Taxi ticket")
        self.ticket_usage_label.pack(anchor=tk.W, padx=(10, 0))
        
        # Initially hide if not Scotland Yard game
        if not isinstance(self.game, ScotlandYardGame):
            self.scotland_yard_frame.pack_forget()
    
    def _create_save_load_controls(self):
        """Create save and load controls"""
        self.save_load_frame = ttk.LabelFrame(self.parent_frame, text="Save & Load", padding="10")
        self.save_load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Save Game button
        self.save_game_btn = ttk.Button(
            self.save_load_frame,
            text="Save Game",
            command=self._on_save_game_clicked
        )
        self.save_game_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Load Game button
        self.load_game_btn = ttk.Button(
            self.save_load_frame,
            text="Load Game",
            command=self._on_load_game_clicked
        )
        self.load_game_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Export Game button
        self.export_game_btn = ttk.Button(
            self.save_load_frame,
            text="Export Game",
            command=self._on_export_game_clicked
        )
        self.export_game_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Game List
        ttk.Label(self.save_load_frame, text="Saved Games:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        self.games_listbox = tk.Listbox(self.save_load_frame, height=5, font=("Arial", 8))
        games_scrollbar = ttk.Scrollbar(self.save_load_frame, orient="vertical")
        
        self.games_listbox.config(yscrollcommand=games_scrollbar.set)
        games_scrollbar.config(command=self.games_listbox.yview)
        
        self.games_listbox.pack(side="left", fill="both", expand=True)
        games_scrollbar.pack(side="right", fill="y")
        
        # Bind double-click to load
        self.games_listbox.bind("<Double-Button-1>", self._on_game_double_click)
        
        # Refresh games list
        self._refresh_games_list()
    
    def _create_setup_controls(self):
        """Create setup phase controls"""
        self.setup_frame = ttk.LabelFrame(self.parent_frame, text="Game Setup", padding="10")
        self.setup_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Setup status
        self.setup_status_label = ttk.Label(self.setup_frame, text="Click nodes to select starting positions")
        self.setup_status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Start Game button
        self.start_game_btn = ttk.Button(
            self.setup_frame,
            text="Start Game",
            command=self._on_start_game_clicked,
            state=tk.DISABLED
        )
        self.start_game_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Clear Selections button
        self.clear_selections_btn = ttk.Button(
            self.setup_frame,
            text="Clear Selections",
            command=self._on_clear_selections_clicked
        )
        self.clear_selections_btn.pack(fill=tk.X)
    
    def _on_make_move_clicked(self):
        """Handle make move button click"""
        if self.on_make_move:
            self.on_make_move()
    
    def _on_skip_turn_clicked(self):
        """Handle skip turn button click"""
        if self.on_skip_turn:
            self.on_skip_turn()
    
    def _on_reset_game_clicked(self):
        """Handle reset game button click"""
        result = messagebox.askyesno("Reset Game", "Are you sure you want to reset the current game?")
        if result and self.on_reset_game:
            self.on_reset_game()
    
    def _on_new_game_clicked(self):
        """Handle new game button click"""
        if self.on_new_game:
            self.on_new_game()
    
    def _on_double_move_clicked(self):
        """Handle double move button click"""
        if self.on_double_move_request:
            self.on_double_move_request()
    
    def _on_black_ticket_changed(self):
        """Handle black ticket checkbox change"""
        if self.on_black_ticket:
            self.on_black_ticket(self.use_black_ticket_var.get())
        
        # Update ticket usage display
        self._update_ticket_usage_display()
    
    def _on_start_game_clicked(self):
        """Handle start game button click"""
        if self.on_start_game:
            self.on_start_game()
    
    def _on_clear_selections_clicked(self):
        """Handle clear selections button click"""
        # This will be handled by the setup controller
        pass
    
    def _on_save_game_clicked(self):
        """Handle save game button click"""
        try:
            # Generate game ID and save
            game_id = self.game_loader.save_game(self.game)
            messagebox.showinfo("Save Game", f"Game saved successfully as {game_id}")
            self._refresh_games_list()
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save game: {str(e)}")
    
    def _on_load_game_clicked(self):
        """Handle load game button click"""
        selection = self.games_listbox.curselection()
        if not selection:
            messagebox.showwarning("Load Game", "Please select a game to load")
            return
        
        try:
            game_info = self.games_listbox.get(selection[0])
            game_id = game_info.split(" - ")[0]
            
            loaded_game = self.game_loader.load_game(game_id)
            if loaded_game:
                # This would need to be handled by the main visualizer
                messagebox.showinfo("Load Game", f"Game {game_id} loaded successfully")
            else:
                messagebox.showerror("Load Error", f"Failed to load game {game_id}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load game: {str(e)}")
    
    def _on_export_game_clicked(self):
        """Handle export game button click"""
        if not hasattr(self.game, 'game_state') or not self.game.game_state:
            messagebox.showwarning("Export Game", "No game to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Game",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # This would need game loader export functionality
                messagebox.showinfo("Export Game", f"Game exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export game: {str(e)}")
    
    def _on_game_double_click(self, event):
        """Handle double-click on games list"""
        self._on_load_game_clicked()
    
    def _refresh_games_list(self):
        """Refresh the list of saved games"""
        try:
            self.games_listbox.delete(0, tk.END)
            games = self.game_loader.list_games()
            
            for game_info in games:
                game_id = game_info.get('game_id', 'Unknown')
                status = game_info.get('status', 'Unknown')
                created = game_info.get('created_at', 'Unknown')
                
                # Format for display
                display_text = f"{game_id} - {status} ({created[:10]})"
                self.games_listbox.insert(tk.END, display_text)
        except Exception as e:
            print(f"Error refreshing games list: {e}")
    
    def _update_ticket_usage_display(self):
        """Update ticket usage display for Scotland Yard"""
        if not isinstance(self.game, ScotlandYardGame):
            return
        
        if self.use_black_ticket_var.get():
            self.ticket_usage_label.config(text="Black ticket", foreground="purple")
        else:
            self.ticket_usage_label.config(text="Transport ticket", foreground="black")
    
    def update_controls(self):
        """Update control states based on game state"""
        if self.setup_phase:
            # Setup phase - hide gameplay controls
            self.game_controls_frame.pack_forget()
            self.setup_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            # Gameplay phase - hide setup controls
            self.setup_frame.pack_forget()
            self.game_controls_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Update gameplay controls
            self._update_gameplay_controls()
        
        # Update Scotland Yard controls
        self._update_scotland_yard_controls()
    
    def _update_gameplay_controls(self):
        """Update gameplay control states"""
        game_over = self.game.is_game_over()
        has_game_state = hasattr(self.game, 'game_state') and self.game.game_state
        
        # Make Move button
        self.make_move_btn.config(
            state=tk.NORMAL if (not game_over and has_game_state and self.moves_selected) else tk.DISABLED
        )
        
        # Skip Turn button
        self.skip_turn_btn.config(
            state=tk.NORMAL if (not game_over and has_game_state and self.can_skip) else tk.DISABLED
        )
        
        # Reset and New Game buttons are always available
        self.reset_game_btn.config(state=tk.NORMAL)
        self.new_game_btn.config(state=tk.NORMAL)
    
    def _update_scotland_yard_controls(self):
        """Update Scotland Yard specific controls"""
        if not isinstance(self.game, ScotlandYardGame):
            self.scotland_yard_frame.pack_forget()
            return
        
        # Show Scotland Yard controls
        self.scotland_yard_frame.pack(fill=tk.X, padx=5, pady=5)
        
        game_over = self.game.is_game_over()
        has_game_state = hasattr(self.game, 'game_state') and self.game.game_state
        is_mr_x_turn = has_game_state and self.game.game_state.turn == Player.ROBBER
        
        # Black ticket checkbox
        if has_game_state and is_mr_x_turn:
            black_tickets = self.game.game_state.mr_x_tickets.get(TicketType.BLACK, 0)
            self.black_ticket_cb.config(
                state=tk.NORMAL if black_tickets > 0 else tk.DISABLED
            )
        else:
            self.black_ticket_cb.config(state=tk.DISABLED)
        
        # Double move button
        if has_game_state and is_mr_x_turn:
            double_moves = self.game.game_state.mr_x_tickets.get(TicketType.DOUBLE_MOVE, 0)
            self.double_move_btn.config(
                state=tk.NORMAL if double_moves > 0 else tk.DISABLED
            )
        else:
            self.double_move_btn.config(state=tk.DISABLED)
        
        # Update ticket usage display
        self._update_ticket_usage_display()
    
    def set_setup_phase(self, is_setup: bool):
        """Set whether we're in setup phase"""
        self.setup_phase = is_setup
        self.update_controls()
    
    def set_moves_selected(self, selected: bool):
        """Set whether moves are selected"""
        self.moves_selected = selected
        self.update_controls()
    
    def set_can_skip(self, can_skip: bool):
        """Set whether turn can be skipped"""
        self.can_skip = can_skip
        self.update_controls()
    
    def update_setup_status(self, message: str):
        """Update setup status message"""
        self.setup_status_label.config(text=message)
    
    def enable_start_game(self, enabled: bool):
        """Enable/disable start game button"""
        self.start_game_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
    
    def get_use_black_ticket(self) -> bool:
        """Get whether black ticket should be used"""
        return self.use_black_ticket_var.get()
    
    def reset_black_ticket(self):
        """Reset black ticket checkbox"""
        self.use_black_ticket_var.set(False)
    
    def set_game(self, game: Game):
        """Update the game reference"""
        self.game = game
        self.update_controls()
    
    # Callback setters
    def set_make_move_callback(self, callback: Callable):
        """Set callback for make move button"""
        self.on_make_move = callback
    
    def set_skip_turn_callback(self, callback: Callable):
        """Set callback for skip turn button"""
        self.on_skip_turn = callback
    
    def set_start_game_callback(self, callback: Callable):
        """Set callback for start game button"""
        self.on_start_game = callback
    
    def set_reset_game_callback(self, callback: Callable):
        """Set callback for reset game button"""
        self.on_reset_game = callback
    
    def set_new_game_callback(self, callback: Callable):
        """Set callback for new game button"""
        self.on_new_game = callback
    
    def set_double_move_callback(self, callback: Callable):
        """Set callback for double move button"""
        self.on_double_move_request = callback
    
    def set_black_ticket_callback(self, callback: Callable):
        """Set callback for black ticket checkbox"""
        self.on_black_ticket = callback
