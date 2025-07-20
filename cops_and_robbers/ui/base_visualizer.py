import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
from ..core.game import ScotlandYardGame


class BaseVisualizer:
    """Base class for game visualization with shared rendering and utility methods"""
    
    def __init__(self, game):
        self.game = game
        
        # Transport type colors and styles (shared between all visualizers)
        self.transport_styles = {
            1: {'color': 'yellow', 'width': 2, 'name': 'Taxi'},
            2: {'color': 'blue', 'width': 3, 'name': 'Bus'},
            3: {'color': 'red', 'width': 4, 'name': 'Underground'},
            4: {'color': 'green', 'width': 3, 'name': 'Ferry'}
        }
        
        # Graph display components (to be initialized by subclasses)
        self.fig = None
        self.ax = None
        self.canvas = None
        self.pos = None
    
    def setup_graph_display(self, parent_frame):
        """Setup matplotlib graph display"""
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Check if game has extracted node positions
        if hasattr(self.game, 'node_positions') and self.game.node_positions:
            # Use extracted board positions (need to normalize for matplotlib)
            positions = self.game.node_positions
            
            # Normalize positions to fit in standard layout bounds
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Normalize to roughly [-1, 1] range like spring layout
            x_range = x_max - x_min
            y_range = y_max - y_min
            scale = max(x_range, y_range)
            
            self.pos = {}
            for node, (x, y) in positions.items():
                normalized_x = 2 * (x - x_min) / scale - 1
                normalized_y = 2 * (y - y_min) / scale - 1
                self.pos[node] = (normalized_x, -normalized_y)  # Flip Y to match image coordinates
        else:
            # Calculate graph layout using spring layout
            self.pos = nx.spring_layout(self.game.graph, seed=42, k=1, iterations=50)
    
    def calculate_parallel_edge_positions(self, u, v, transport_types, offset_distance=0.02):
        """Calculate parallel positions for multiple edges between two nodes"""
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
    
    def draw_edges_with_parallel_positioning(self, alpha=0.6, highlighted_edges=None):
        """Draw edges with parallel positioning for multiple transport types"""
        # highlighted_edges should be a dict of {transport_type: [(u, v), ...]}
        if highlighted_edges is None:
            highlighted_edges = {}
            
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
                
                # Check if this edge should be highlighted
                is_highlighted = ((u, v) in highlighted_edges.get(transport_type, []) or 
                                (v, u) in highlighted_edges.get(transport_type, []))
                
                # Get the parallel position for this transport type
                if i < len(parallel_positions):
                    pos_u_offset, pos_v_offset = parallel_positions[i]
                    
                    # Draw the edge with appropriate highlighting
                    if is_highlighted:
                        # Highlighted edge - full color and increased thickness
                        self.ax.plot([pos_u_offset[0], pos_v_offset[0]], 
                                   [pos_u_offset[1], pos_v_offset[1]],
                                   color=style['color'], 
                                   linewidth=style['width'] + 2, 
                                   alpha=1.0, 
                                   solid_capstyle='round')
                    else:
                        # Normal edge
                        self.ax.plot([pos_u_offset[0], pos_v_offset[0]], 
                                   [pos_u_offset[1], pos_v_offset[1]],
                                   color=style['color'], 
                                   linewidth=style['width'], 
                                   alpha=alpha, 
                                   solid_capstyle='round')
    
    def draw_transport_legend(self):
        """Draw transport legend"""
        legend_handles = []
        for transport_val, style in self.transport_styles.items():
            legend_handles.append(mlines.Line2D([], [], color=style['color'], 
                                              linewidth=style['width'], 
                                              label=style['name']))
        
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc='lower right')
    
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
    
    def update_tickets_display_table(self, tickets_display, state=None):
        """Update tickets display as a table format with improved spacing"""
        if not tickets_display or not hasattr(tickets_display, 'set_text'):
            return
        
        # Use current game state if no state provided
        current_state = state or self.game.game_state
        if not current_state:
            tickets_display.set_text("No game state available")
            return
        
        # Create table format for tickets with better spacing
        tickets_text = "🎫 TICKET TABLE:\n\n"
        
        # Header row with proper spacing
        tickets_text += "Player│🚕│🚌│🚇│⚫│⚡\n"
        tickets_text += "──────┼──┼──┼──┼──┼──\n"
        
        # Detective rows
        for i in range(self.game.num_cops):
            player_name = f"Det {i+1}"
            
            # Get tickets for this detective
            if hasattr(current_state, 'detective_tickets'):
                detective_tickets = current_state.detective_tickets
                if isinstance(detective_tickets, dict) and i in detective_tickets:
                    tickets = detective_tickets[i]
                elif isinstance(detective_tickets, list) and i < len(detective_tickets):
                    tickets = detective_tickets[i]
                else:
                    tickets = {}
            else:
                # Fallback to game method if available
                tickets = getattr(self.game, 'get_detective_tickets', lambda x: {})(i)
            
            # Display ticket counts in table format with proper alignment
            taxi_count = self._get_ticket_count(tickets, 'taxi')
            bus_count = self._get_ticket_count(tickets, 'bus')
            underground_count = self._get_ticket_count(tickets, 'underground')
            
            tickets_text += f"{player_name:<6}│{taxi_count:>2}│{bus_count:>2}│{underground_count:>2}│ -│ -\n"
        
        # Mr. X row
        if hasattr(current_state, 'mr_x_tickets'):
            mr_x_tickets = current_state.mr_x_tickets
        else:
            # Fallback to game method if available
            mr_x_tickets = getattr(self.game, 'get_mr_x_tickets', lambda: {})()
        
        taxi_count = self._get_ticket_count(mr_x_tickets, 'taxi')
        bus_count = self._get_ticket_count(mr_x_tickets, 'bus')
        underground_count = self._get_ticket_count(mr_x_tickets, 'underground')
        black_count = self._get_ticket_count(mr_x_tickets, 'black')
        double_count = self._get_ticket_count(mr_x_tickets, 'double_move')
        
        tickets_text += f"{'Mr. X':<6}│{taxi_count:>2}│{bus_count:>2}│{underground_count:>2}│{black_count:>2}│{double_count:>2}\n"
        
        tickets_display.set_text(tickets_text)
    
    def draw_basic_graph_elements(self, title="Game Graph", node_colors=None, node_sizes=None):
        """Draw basic graph elements (nodes, edges, labels) without specific game state"""
        self.ax.clear()
        
        # Draw edges with parallel positioning
        self.draw_edges_with_parallel_positioning()
        
        # Use provided colors/sizes or defaults
        if node_colors is None:
            node_colors = ['lightgray'] * len(self.game.graph.nodes())
        if node_sizes is None:
            node_sizes = [300] * len(self.game.graph.nodes())
        
        nx.draw_networkx_nodes(self.game.graph, self.pos, ax=self.ax,
                              node_color=node_colors, node_size=node_sizes)
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        # Set title and legend
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.draw_transport_legend()
        
        self.ax.axis('off')
        if self.canvas:
            self.canvas.draw()
    
    def draw_common_graph_structure(self, title="Game Graph"):
        """Draw common graph structure that can be extended by subclasses"""
        self.ax.clear()
        
        # This method should be overridden by subclasses for custom drawing
        # Default implementation draws basic structure
        self.draw_edges_with_parallel_positioning()
        
        # Draw labels
        nx.draw_networkx_labels(self.game.graph, self.pos, ax=self.ax, font_size=8)
        
        # Set title and legend
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.draw_transport_legend()
        
        self.ax.axis('off')
        if self.canvas:
            self.canvas.draw()
            
