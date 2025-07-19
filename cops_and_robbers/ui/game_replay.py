import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
from ..core.game import Game, Player, ScotlandYardGame, TicketType, TransportType
from ..storage.game_loader import GameLoader
from .ui_components import StyledButton, InfoDisplay


class GameReplayWindow:
    """Window for replaying saved games step by step"""
    
    def __init__(self, parent, game_id: str, game: ScotlandYardGame, loader: GameLoader):
        self.parent = parent
        self.game_id = game_id
        self.game = game
        self.loader = loader
        self.window = None
        self.current_step = 0
        
        # Transport type colors and styles (same as main visualizer)
        self.transport_styles = {
            1: {'color': 'yellow', 'width': 2, 'name': 'Taxi'},
            2: {'color': 'blue', 'width': 3, 'name': 'Bus'},
            3: {'color': 'red', 'width': 4, 'name': 'Underground'},
            4: {'color': 'green', 'width': 3, 'name': 'Ferry'}
        }
        
        # UI components
        self.fig = None
        self.ax = None
        self.canvas = None
        self.pos = None
        
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
        self.tickets_display = InfoDisplay(left_panel, "🎫 Ticket Information", height=8)
        self.tickets_display.pack(fill=tk.X, pady=(0, 10))
        
        # Right panel for graph
        self.graph_frame = ttk.Frame(self.window)
        self.graph_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
    
    def setup_graph_display(self):
        """Setup matplotlib graph display"""
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Calculate graph layout once
        self.pos = nx.spring_layout(self.game.graph, seed=42, k=1, iterations=50)
    
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
        self.update_tickets_display(current_state)
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
            elif player == 'cops':
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

    def get_ticket_emoji(self, ticket_used):
        """Get emoji for ticket type"""
        if not ticket_used or ticket_used == "Unknown":
            return "🎫"
        
        ticket_emojis = {
            'taxi': '🚕',
            'bus': '🚌', 
            'underground': '🚇',
            'black': '⚫',
            'double_move': '⚡',
            'TAXI': '🚕',
            'BUS': '🚌',
            'UNDERGROUND': '🚇', 
            'BLACK': '⚫',
            'DOUBLE_MOVE': '⚡'
        }
        return ticket_emojis.get(ticket_used, '🎫')


    def update_tickets_display(self, state):
        """Update tickets display as a table format with improved spacing"""
        if not self.tickets_display:
            return
        
        # Create table format for tickets with better spacing
        tickets_text = "🎫 TICKET TABLE:\n\n"
        
        # Header row with proper spacing
        tickets_text += "Player│🚕│🚌│🚇│⚫│⚡\n"
        tickets_text += "──────┼──┼──┼──┼──┼──\n"
        
        # Detective rows
        if hasattr(state, 'detective_tickets') or hasattr(self.game, 'detective_tickets'):
            detective_tickets = getattr(state, 'detective_tickets', None) or getattr(self.game, 'detective_tickets', [])
            
            for i in range(self.game.num_cops):
                pos = state.cop_positions[i] if i < len(state.cop_positions) else "N/A"
                player_name = f"Det {i+1}"
                
                # Get tickets for this detective
                if isinstance(detective_tickets, dict) and i in detective_tickets:
                    tickets = detective_tickets[i]
                elif isinstance(detective_tickets, list) and i < len(detective_tickets):
                    tickets = detective_tickets[i]
                else:
                    tickets = {}
                
                # Display ticket counts in table format with proper alignment
                taxi_count = self._get_ticket_count(tickets, 'taxi')
                bus_count = self._get_ticket_count(tickets, 'bus')
                underground_count = self._get_ticket_count(tickets, 'underground')
                
                tickets_text += f"{player_name:<6}│{taxi_count:>2}│{bus_count:>2}│{underground_count:>2}│ -│ -\n"
        
        # Mr. X row
        if hasattr(state, 'mr_x_tickets'):
            mr_x_tickets = state.mr_x_tickets
        elif hasattr(self.game, 'mr_x_tickets'):
            mr_x_tickets = self.game.mr_x_tickets
        else:
            mr_x_tickets = {}
        
        taxi_count = self._get_ticket_count(mr_x_tickets, 'taxi')
        bus_count = self._get_ticket_count(mr_x_tickets, 'bus')
        underground_count = self._get_ticket_count(mr_x_tickets, 'underground')
        black_count = self._get_ticket_count(mr_x_tickets, 'black')
        double_count = self._get_ticket_count(mr_x_tickets, 'double_move')
        
        tickets_text += f"{'Mr. X':<6}│{taxi_count:>2}│{bus_count:>2}│{underground_count:>2}│{black_count:>2}│{double_count:>2}\n"
        
        self.tickets_display.set_text(tickets_text)
    
    def _get_ticket_count(self, tickets, ticket_name):
        """Helper to get ticket count handling different formats"""
        if not tickets:
            return 0
        
        # Try different possible key formats
        possible_keys = [
            ticket_name.lower(),
            ticket_name.upper(),
            f"TicketType.{ticket_name.upper()}",
        ]
        
        # Also try enum objects
        for key, value in tickets.items():
            if hasattr(key, 'value') and key.value.lower() == ticket_name.lower():
                return value
            elif hasattr(key, 'name') and key.name.lower() == ticket_name.lower():
                return value
            elif str(key).lower() == ticket_name.lower():
                return value
        
        # Try string keys
        for possible_key in possible_keys:
            if possible_key in tickets:
                return tickets[possible_key]
        
        return 0

    def draw_graph(self, state):
        """Draw the graph using the same style as game_visualizer"""
        self.ax.clear()
        
        # Use the same multi-edge drawing logic from game_visualizer
        self.draw_edges_with_parallel_positioning(state)
        self.draw_nodes_with_proper_colors(state)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        # Title with step info and emoji
        turn_info = f"Turn {state.turn_count} - {state.turn.value.title()}'s Turn"
        self.ax.set_title(f"🎬 Game Replay - {turn_info}", fontsize=12, fontweight='bold')
        
        # Legend
        self.draw_transport_legend()
        
        self.ax.axis('off')
        self.canvas.draw()

    def draw_edges_with_parallel_positioning(self, state):
        """Draw edges with parallel positioning for multiple transport types (copied from game_visualizer)"""
        # Build edge data structure for parallel drawing
        edge_data = {}  # (u, v) -> [transport_types]
        
        for u, v, data in self.game.graph.edges(data=True):
            edge_transports = data.get('transports', [])
            edge_type = data.get('edge_type', None)
            
            # Determine which transport types are available for this edge
            available_transports = []
            if edge_transports:
                available_transports = edge_transports
            elif edge_type:
                available_transports = [edge_type]
            
            # Ensure consistent edge direction for parallel calculation
            edge_key = (min(u, v), max(u, v))
            if edge_key not in edge_data:
                edge_data[edge_key] = []
            edge_data[edge_key].extend(available_transports)
        
        # Remove duplicates from transport types
        for edge_key in edge_data:
            edge_data[edge_key] = list(set(edge_data[edge_key]))
        
        # Draw edges with parallel positioning
        for (u, v), transport_types in edge_data.items():
            # Calculate parallel positions for this edge
            parallel_positions = self.calculate_parallel_edge_positions(u, v, transport_types)
            
            for i, transport_type in enumerate(transport_types):
                if transport_type not in self.transport_styles:
                    continue
                
                style = self.transport_styles[transport_type]
                
                # Get the parallel position for this transport type
                if i < len(parallel_positions):
                    pos_u_offset, pos_v_offset = parallel_positions[i]
                    
                    # Draw the edge with reduced transparency for replay
                    self.ax.plot([pos_u_offset[0], pos_v_offset[0]], 
                               [pos_u_offset[1], pos_v_offset[1]],
                               color=style['color'], 
                               linewidth=style['width'], 
                               alpha=0.3, 
                               solid_capstyle='round')

    def calculate_parallel_edge_positions(self, u, v, transport_types, offset_distance=0.02):
        """Calculate parallel positions for multiple edges between two nodes (copied from game_visualizer)"""
        if u not in self.pos or v not in self.pos:
            return []
        
        pos_u = np.array(self.pos[u])
        pos_v = np.array(self.pos[v])
        
        # Calculate the vector from u to v
        edge_vector = pos_v - pos_u
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length == 0:
            return [(pos_u, pos_v) for _ in transport_types]
        
        # Calculate perpendicular vector for offsetting
        perp_vector = np.array([-edge_vector[1], edge_vector[0]])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)
        
        # Calculate offsets for each transport type
        num_transports = len(transport_types)
        positions = []
        
        if num_transports == 1:
            # Single edge - no offset needed
            positions.append((pos_u, pos_v))
        else:
            # Multiple edges - distribute them symmetrically
            for i, transport in enumerate(transport_types):
                # Calculate offset from center
                offset_multiplier = (i - (num_transports - 1) / 2) * offset_distance
                offset = perp_vector * offset_multiplier
                
                pos_u_offset = pos_u + offset
                pos_v_offset = pos_v + offset
                positions.append((pos_u_offset, pos_v_offset))
        
        return positions

    def draw_nodes_with_proper_colors(self, state):
        """Draw nodes with proper colors based on current state (adapted from game_visualizer)"""
        node_colors = []
        node_sizes = []
        
        for node in self.game.graph.nodes():
            if node in state.cop_positions:
                node_colors.append('blue')
                node_sizes.append(500)
            elif node == state.robber_position:  
                node_colors.append('red')
                node_sizes.append(500)

            else:
                node_colors.append('lightgray')
                node_sizes.append(300)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                              node_color=node_colors, node_size=node_sizes)

    def draw_transport_legend(self):
        """Draw transport legend (copied from game_visualizer)"""
        legend_handles = []
        import matplotlib.lines as mlines
        for transport_val, style in self.transport_styles.items():
            legend_handles.append(mlines.Line2D([], [], color=style['color'], 
                                              linewidth=style['width'], 
                                              label=style['name']))
        
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc='upper right')

    def update_info_display(self, state):
        """Update current state information"""
        if not self.info_display:
            return
        
        info_text = f"🎯 Turn: {state.turn.value.title()}\n"
        info_text += f"📊 Turn Count: {state.turn_count}\n"
        info_text += f"👮 Cop Positions: {state.cop_positions}\n"
        
        # is_scotland_yard = isinstance(self.game, ScotlandYardGame)
        is_scotland_yard = True
        if is_scotland_yard or hasattr(state, 'mr_x_visible'):
            if hasattr(state, 'mr_x_visible') and not state.mr_x_visible:
                info_text += f"🎭 Mr. X Position: HIDDEN\n"
            else:
                info_text += f"🕵️‍♂️ Mr. X Position: {state.robber_position}\n"
            
            if hasattr(state, 'double_move_active') and state.double_move_active:
                info_text += "⚡ Double Move: ACTIVE\n"
        else:
            info_text += f"🏃 Robber Position: {state.robber_position}\n"
        
        # Add ticket usage information if available
        if hasattr(state, 'last_move_ticket') and state.last_move_ticket:
            ticket_emoji = self.get_ticket_emoji(state.last_move_ticket)
            info_text += f"🎫 Last Move Ticket: {ticket_emoji} {state.last_move_ticket}\n"
        
        if hasattr(state, 'previous_position') and state.previous_position:
            current_pos = state.robber_position if state.turn.value == 'robber' else "N/A"
            info_text += f"🔄 Move: {state.previous_position} → {current_pos}\n"
        
        # Check if game is over at this step
        temp_game_state = self.game.game_state
        self.game.game_state = state
        is_over = self.game.is_game_over()
        if is_over:
            winner = self.game.get_winner()
            info_text += f"\n🏆 GAME OVER!\n🎉 Winner: {winner.value.title() if winner else 'None'}"
        self.game.game_state = temp_game_state
        
        self.info_display.set_text(info_text)
    
    
    def _get_ticket_count(self, tickets, ticket_name):
        """Helper to get ticket count handling different formats"""
        if not tickets:
            return 0
        
        # Try different possible key formats
        possible_keys = [
            ticket_name.lower(),
            ticket_name.upper(),
            f"TicketType.{ticket_name.upper()}",
        ]
        
        # Also try enum objects
        for key, value in tickets.items():
            if hasattr(key, 'value') and key.value.lower() == ticket_name.lower():
                return value
            elif hasattr(key, 'name') and key.name.lower() == ticket_name.lower():
                return value
            elif str(key).lower() == ticket_name.lower():
                return value
        
        # Try string keys
        for possible_key in possible_keys:
            if possible_key in tickets:
                return tickets[possible_key]
        
        return 0
        
        return 0
        # Try string keys
        for possible_key in possible_keys:
            if possible_key in tickets:
                return tickets[possible_key]
        
        return 0
