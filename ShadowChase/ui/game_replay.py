import tkinter as tk
from tkinter import ttk
import networkx as nx
from ..core.game import  ShadowChaseGame
from ..services.game_loader import GameLoader
from .ui_components import StyledButton, InfoDisplay
from .base_visualizer import BaseVisualizer

NODE_SIZE = 300


class GameReplayWindow(BaseVisualizer):
    """Window for replaying saved games step by step"""
    
    def __init__(self, parent, game_id: str, game: ShadowChaseGame, loader: GameLoader):
        super().__init__(game)
        self.parent = parent
        self.game_id = game_id
        self.loader = loader
        self.window = None
        self.current_step = 0
        
        # Control widgets
        self.step_label = None
        self.prev_button = None
        self.next_button = None
        self.step_scale = None
        self.info_display = None
        self.game_info_display = None
        self.tickets_display = None
    
    def show(self):
        """Show the replay window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"🎬 Game Replay - {self.game_id}")
        self.window.geometry("1400x900")
        self.window.configure(bg="#f8f9fa")
        
        # Make window resizable
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        
        self.setup_ui()
        self.setup_graph_display()
        self.update_display()
    
    def setup_ui(self):
        """Setup the replay window UI"""
        # Left panel for controls
        left_panel = ttk.Frame(self.window, width=350)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_panel.grid_propagate(False)
        
        # Title
        title_label = ttk.Label(left_panel, text="🎬 Game Replay", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Game info
        info_text = f"Game ID: {self.game_id}\n"
        if self.game.game_history:
            info_text += f"Total Steps: {len(self.game.game_history)}\n"
            if self.game.is_game_over():
                winner = self.game.get_winner()
                info_text += f"Winner: {winner.value.title() if winner else 'None'}\n"
        
        game_info_frame = ttk.LabelFrame(left_panel, text="📋 Game Information")
        game_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(game_info_frame, text=info_text, 
                 justify=tk.LEFT).pack(padx=10, pady=5)
        
        # Step controls
        controls_frame = ttk.LabelFrame(left_panel, text="🎮 Playback Controls")
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current step display
        self.step_label = ttk.Label(controls_frame, text="Step: 0 / 0", 
                                   font=('Arial', 10, 'bold'))
        self.step_label.pack(pady=5)
        
        # Step slider
        if self.game.game_history:
            max_steps = len(self.game.game_history) - 1
            self.step_scale = tk.Scale(controls_frame, from_=0, to=max_steps, 
                                      orient=tk.HORIZONTAL, command=self.on_scale_change)
            self.step_scale.pack(fill=tk.X, padx=10, pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.prev_button = StyledButton(nav_frame, "⏮️ Previous", 
                                       command=self.prev_step, style_type="primary")
        self.prev_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.next_button = StyledButton(nav_frame, "Next ⏭️", 
                                       command=self.next_step, style_type="primary")
        self.next_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Auto-play controls
        auto_frame = ttk.Frame(controls_frame)
        auto_frame.pack(fill=tk.X, padx=10, pady=5)
        
        StyledButton(auto_frame, "⏪ First", command=self.go_to_first).pack(side=tk.LEFT, padx=(0, 2))
        StyledButton(auto_frame, "⏩ Last", command=self.go_to_last).pack(side=tk.RIGHT, padx=(2, 0))
        
        # Game History display
        self.history_display = InfoDisplay(left_panel, "📜 Game History", height=6)
        self.history_display.pack(fill=tk.X, pady=(0, 10))
        
        # Current state information
        self.info_display = InfoDisplay(left_panel, "📊 Current State", height=6)
        self.info_display.pack(fill=tk.X, pady=(0, 10))
        
        # Tickets display
        self.tickets_display = InfoDisplay(left_panel, "🎫 Ticket Information", height=10)
        self.tickets_display.pack(fill=tk.X, pady=(0, 10))
        
        # Right panel for graph
        self.graph_frame = ttk.Frame(self.window)
        self.graph_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
    
    def setup_graph_display(self):
        """Setup matplotlib graph display"""
        # Debug: Check if game has node positions
        if hasattr(self.game, 'node_positions') and self.game.node_positions:
            print(f"🎬 Replay: Using extracted board positions ({len(self.game.node_positions)} nodes)")
        else:
            print("🎬 Replay: No node positions found, using spring layout")
            # If the game doesn't have node_positions but we can detect it's an extracted board,
            # try to load them from the board_progress.json file
            try:
                from ShadowChase.services.board_loader import load_board_graph_from_csv
                _, node_positions = load_board_graph_from_csv()
                
                # Check if this game uses the same nodes as the extracted board
                game_nodes = set(self.game.graph.nodes())
                extracted_nodes = set(node_positions.keys())
                
                if game_nodes == extracted_nodes:
                    print("🎬 Replay: Loading node positions from board_progress.json")
                    self.game.node_positions = node_positions
                else:
                    print(f"🎬 Replay: Node sets don't match (game: {len(game_nodes)}, extracted: {len(extracted_nodes)})")
            except Exception as e:
                print(f"🎬 Replay: Could not load board positions: {e}")
        
        # Call parent method which handles both extracted positions and spring layout
        super().setup_graph_display(self.graph_frame)
    
    def on_scale_change(self, value):
        """Handle scale change"""
        self.current_step = int(value)
        self.update_display()
    
    def prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            if self.step_scale:
                self.step_scale.set(self.current_step)
            self.update_display()
    
    def next_step(self):
        """Go to next step"""
        max_steps = len(self.game.game_history) - 1
        if self.current_step < max_steps:
            self.current_step += 1
            if self.step_scale:
                self.step_scale.set(self.current_step)
            self.update_display()
    
    def go_to_first(self):
        """Go to first step"""
        self.current_step = 0
        if self.step_scale:
            self.step_scale.set(self.current_step)
        self.update_display()
    
    def go_to_last(self):
        """Go to last step"""
        self.current_step = len(self.game.game_history) - 1
        if self.step_scale:
            self.step_scale.set(self.current_step)
        self.update_display()
    
    def update_display(self):
        """Update the display for current step"""
        if not self.game.game_history or self.current_step >= len(self.game.game_history):
            return
        
        # Get current state
        current_state = self.game.game_history[self.current_step]
        
        # Update step label
        max_steps = len(self.game.game_history) - 1
        self.step_label.configure(text=f"Step: {self.current_step} / {max_steps}")
        
        # Update button states
        self.prev_button.configure(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL if self.current_step < max_steps else tk.DISABLED)
        
        # Update displays
        self.update_history_display(current_state)
        self.update_info_display(current_state)
        self.update_tickets_display_table(self.tickets_display, current_state)
        self.draw_graph(current_state)

    def update_history_display(self, current_state):
        """Update game history display up to current step using ticket history"""
        if not self.history_display:
            return

        history_text = f"📜 MOVES UP TO STEP {self.current_step}:\n\n"
        # Assume self.game.ticket_history is available and matches the format provided
        # ticket_history = getattr(self.game, "ticket_history", None)
        ticket_history = self.game.ticket_history 
        if not ticket_history:
            self.history_display.set_text("No ticket history available.")
            return

        for i in range(min(self.current_step + 1, len(ticket_history))):
            step = ticket_history[i]
            turn_num = step.get('turn_number', i)
            player = step.get('player', '')
            if player == 'mr_x':
                history_text += f"🕵️‍♂️ Turn {turn_num} - MR. X MOVES:\n"
                for move in step.get('mr_x_moves', []):
                    edge = move.get('edge', None)
                    ticket_used = move.get('ticket_used', 'Unknown')
                    ticket_emoji = self.get_ticket_emoji(ticket_used)
                    transport = move.get('transport_used', None)
                    double_move_part = move.get('double_move_part', None)
                    move_str = f"  🎭 Mr. X: {edge[0]} → {edge[1]} {ticket_emoji}({ticket_used})"
                    if step.get('double_move_used', False):
                        move_str += " ⚡[DOUBLE MOVE]"
                    history_text += move_str + "\n"
                history_text += "\n"
            elif player == 'detectives':
                history_text += f"👮 Turn {turn_num} - DETECTIVES MOVE:\n"
                for move in step.get('detective_moves', []):
                    det_id = move.get('detective_id', None)
                    edge = move.get('edge', None)
                    ticket_used = move.get('ticket_used', 'Unknown')
                    ticket_emoji = self.get_ticket_emoji(ticket_used)
                    stayed = move.get('stayed', False)
                    if stayed:
                        move_str = f"  🕵️ Detective {det_id+1}: Stayed at {edge[0]}"
                    else:
                        move_str = f"  🕵️ Detective {det_id+1}: {edge[0]} → {edge[1]} {ticket_emoji}({ticket_used})"
                    history_text += move_str + "\n"
                history_text += "\n"
            else:
                history_text += f"⏸️ Turn {turn_num} - NO POSITION CHANGE\n\n"

        self.history_display.set_text(history_text)
    

    def draw_graph(self, state):
        """Draw the graph using base class methods"""
        self.ax.clear()
        
        # Use base class method for drawing edges with reduced transparency for replay
        self.draw_edges_with_parallel_positioning(alpha=0.3)
        
        # Get node colors and sizes using unified system
        node_colors, node_sizes = self.get_node_colors_and_sizes(
            mode="history", 
            step=self.current_step, 
            node_size=NODE_SIZE
        )
        
        nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                              node_color=node_colors, node_size=node_sizes)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        # Get historical state for consistent title formatting
        historical_state = self.get_historical_state(self.current_step)
        turn_count = getattr(historical_state, 'turn_count', self.current_step)
        current_turn = self._normalize_turn(historical_state)
        turn_info = f"Turn {turn_count} - {current_turn.title()}'s Turn"
        self.ax.set_title(f"🎬 Game Replay - {turn_info}", fontsize=12, fontweight='bold')
        
        # Use base class method for legend
        self.draw_transport_legend()
        
        self.ax.axis('off')
        self.canvas.draw()

    def update_info_display(self, state):
        """Update current state information"""
        if not self.info_display:
            return
        
        # Get historical state for consistent data access
        historical_state = self.get_historical_state(self.current_step)
        
        # Get normalized attributes
        turn_count = getattr(historical_state, 'turn_count', self.current_step)
        detective_positions = getattr(historical_state, 'detective_positions', [])
        mr_x_position = getattr(historical_state, 'mr_x_position', None) or getattr(historical_state, 'MrX_position', 0)
        mr_x_visible = getattr(historical_state, 'mr_x_visible', True)
        current_turn = self._normalize_turn(historical_state)
        
        info_text = f"🎯 Turn: {current_turn.title()}\n"
        info_text += f"📊 Turn Count: {turn_count}\n"
        info_text += f"👮 Detective Positions: {detective_positions}\n"
        
        if not mr_x_visible:
            info_text += f"🎭 Mr. X Position: {mr_x_position} HIDDEN\n"
        else:
            info_text += f"🕵️‍♂️ Mr. X Position: {mr_x_position}\n"
        
        double_move_active = getattr(historical_state, 'double_move_active', False)
        if double_move_active:
            info_text += "⚡ Double Move: ACTIVE\n"
        
        # Add ticket usage information if available
        last_move_ticket = getattr(historical_state, 'last_move_ticket', None)
        if last_move_ticket:
            ticket_emoji = self.get_ticket_emoji(last_move_ticket)
            info_text += f"🎫 Last Move Ticket: {ticket_emoji} {last_move_ticket}\n"
        
        previous_position = getattr(historical_state, 'previous_position', None)
        if previous_position:
            current_pos = mr_x_position if current_turn == 'mr_x' else "N/A"
            info_text += f"🔄 Move: {previous_position} → {current_pos}\n"
        
        # Check if game is over at this step
        game_over = self._check_game_over_status(historical_state)
        if game_over:
            winner = self._get_winner_at_step(historical_state)
            info_text += f"\n🏆 GAME OVER!\n🎉 Winner: {winner or 'None'}"
        
        self.info_display.set_text(info_text)
    

