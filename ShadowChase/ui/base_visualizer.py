import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import os
from typing import Optional, List, Dict, Any


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
        
        # Board image overlay settings
        self.show_board_image = True
        self.board_image = None
        self.board_image_path = "data/board.png"
    
    def setup_graph_display(self, parent_frame):
        """Setup matplotlib graph display"""
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Load board image if available
        self.load_board_image()
        
        # Load calibration parameters
        self.load_calibration_parameters()
        
        # Check if game has extracted node positions
        if hasattr(self.game, 'node_positions') and self.game.node_positions:
            # Use extracted board positions with calibration
            self.calculate_calibrated_positions()
        else:
            # Calculate graph layout using spring layout
            self.pos = nx.spring_layout(self.game.graph, seed=42, k=1, iterations=50)
    
    def load_calibration_parameters(self):
        """Load calibration parameters from file if available"""
        self.calibration = {
            'x_offset': 0.0,
            'y_offset': 0.0, 
            'x_scale': 1.0,
            'y_scale': 1.0,
            'image_alpha': 0.8
        }
        
        try:
            import json
            calibration_file = "data/board_calibration.json"
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    saved_calibration = json.load(f)
                    self.calibration.update(saved_calibration)
                print(f"Loaded calibration parameters: {self.calibration}")
        except Exception as e:
            print(f"Could not load calibration parameters: {e}")
    
    def calculate_calibrated_positions(self):
        """Calculate calibrated positions using loaded parameters"""
        positions = self.game.node_positions
        
        # Get coordinate bounds
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Apply calibration scaling to ranges
        x_range = (x_max - x_min) * self.calibration['x_scale']
        y_range = (y_max - y_min) * self.calibration['y_scale']
        
        self.pos = {}
        for node, (x, y) in positions.items():
            # Normalize to [-1, 1] range with calibration parameters
            normalized_x = 2 * ((x - x_min) / x_range) - 1 + self.calibration['x_offset']
            normalized_y = -(2 * ((y - y_min) / y_range) - 1) + self.calibration['y_offset']  # Flip Y and apply offset
            self.pos[node] = (normalized_x, normalized_y)
    
    def load_board_image(self):
        """Load the board image for overlay"""
        try:
            if os.path.exists(self.board_image_path):
                self.board_image = mpimg.imread(self.board_image_path)
                print(f"Board image loaded: {self.board_image.shape}")
            else:
                print(f"Board image not found at: {self.board_image_path}")
                self.board_image = None
                self.show_board_image = False
        except Exception as e:
            print(f"Error loading board image: {e}")
            self.board_image = None
            self.show_board_image = False
    
    def get_live_state(self, ui_state: Optional[Dict] = None):
        """Get current game state with normalized attribute access and UI state integration"""
        current_state = getattr(self.game, 'game_state', None)
        
        if not current_state:
            # Create a simple object for setup mode
            class SetupState:
                def __init__(self):
                    self.detective_positions = []
                    self.MrX_position = 0
                    self.MrX_position = 0  # Normalized attribute
                    self.turn_count = 0
                    self.setup_mode = True
                    self.selected_positions = ui_state.get('selected_positions', []) if ui_state else []
                    self.active_player_positions = ui_state.get('active_player_positions', []) if ui_state else []
                    self.detective_selections = ui_state.get('detective_selections', []) if ui_state else []
            
            return SetupState()
        
        # Add normalized attributes to existing state object if they don't exist
        if not hasattr(current_state, 'MrX_position') and hasattr(current_state, 'MrX_position'):
            current_state.MrX_position = current_state.MrX_position
        
        # Add UI state if provided
        if ui_state:
            for key, value in ui_state.items():
                setattr(current_state, key, value)
        
        # Mark as not setup mode
        current_state.setup_mode = False
        
        return current_state
    
    def get_historical_state(self, step: int):
        """Get historical game state with normalized attribute access"""
        if not hasattr(self.game, 'game_history') or not self.game.game_history:
            raise ValueError("No game history available")
        
        if step >= len(self.game.game_history):
            raise ValueError(f"Step {step} is beyond game history length {len(self.game.game_history)}")
        
        historical_state = self.game.game_history[step]
        
        # Add normalized attributes if needed
        if not hasattr(historical_state, 'MrX_position') and hasattr(historical_state, 'MrX_position'):
            historical_state.MrX_position = historical_state.MrX_position
        
        return historical_state
    def _get_attribute(self, obj, attr_names: List[str], default=None):
        """Get attribute from object trying multiple possible names"""
        for attr_name in attr_names:
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)
        return default
    
    def _normalize_turn(self, state) -> str:
        """Normalize turn information to consistent format"""
        # Try different turn attribute names and formats
        turn_attr = self._get_attribute(state, ['turn', 'current_turn', 'player_turn'])
        
        if turn_attr is None:
            return 'detective'  # Default
        
        # Handle enum objects
        if hasattr(turn_attr, 'value'):
            turn_value = turn_attr.value.lower()
        elif hasattr(turn_attr, 'name'):
            turn_value = turn_attr.name.lower()
        else:
            turn_value = str(turn_attr).lower()
        
        # Normalize to standard format
        if 'detective' in turn_value or 'det' in turn_value:
            return 'detective'
        elif 'mr' in turn_value or 'x' in turn_value:
            return 'MrX'
        else:
            return 'detective'  # Default
    
    def draw_board_image(self):
        """Draw the board image as background"""
        if self.show_board_image and self.board_image is not None:
            # Use calibrated alpha if available
            alpha = self.calibration.get('image_alpha', 0.8) if hasattr(self, 'calibration') else 0.8
            # Display the image with the same extent as the normalized coordinates
            self.ax.imshow(self.board_image, extent=[-1, 1, -1, 1], alpha=alpha, aspect='auto')
    
    def toggle_board_image(self):
        """Toggle board image visibility"""
        self.show_board_image = not self.show_board_image
        return self.show_board_image
    
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
    
    def draw_edges_with_parallel_positioning(self, alpha=0.6, highlighted_edges=None, show_edges=True):
        """Draw edges with parallel positioning for multiple transport types"""
        # Skip drawing edges if show_edges is False (for board image overlay mode)
        if not show_edges:
            return
            
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
    
    def update_enhanced_tickets_display(self, tickets_display, state=None):
        """Update enhanced visual tickets display"""
        if not tickets_display or not hasattr(tickets_display, 'update_tickets'):
            return
        
        # Use current game state if no state provided
        current_state = state or self.game.game_state
        if not current_state:
            return
        
        # Use the enhanced component's update method
        tickets_display.update_tickets(current_state, self.game)
    
    def get_node_colors_and_sizes(self, mode: str = "live", step: Optional[int] = None,
                                 node_size: int = 300, ui_state: Optional[Dict] = None) -> tuple:
        """
        Unified node coloring system for all visualizers
        
        Args:
            mode: "live" (GameVisualizer), "history" (VideoExporter/Replay)
            step: Step number for history mode
            node_size: Base node size (300 for normal, 400 for video)
            ui_state: UI-specific state variables (for GameVisualizer)
        
        Returns:
            Tuple of (node_colors, node_sizes) lists
        """
        # Get state based on mode
        if mode == "live":
            state = self.get_live_state(ui_state)
        elif mode == "history":
            state = self.get_historical_state(step)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        node_colors = []
        node_sizes = []
        
        for node in self.game.graph.nodes():
            color, size = self._get_node_color_and_size(mode, node, state, node_size)
            node_colors.append(color)
            node_sizes.append(size)
        
        return node_colors, node_sizes

    def _get_node_color_and_size(self, mode: str, node: int, state, base_size: int) -> tuple:
        """Get color and size for a single node based on state - simplified logic"""
        
        # Setup mode (GameVisualizer only)
        if hasattr(state, 'setup_mode') and state.setup_mode:
            if hasattr(state, 'selected_positions') and node in state.selected_positions:
                color = 'blue' if len(state.selected_positions) <= self.game.num_detectives else 'red'
                return color, base_size
            else:
                return 'lightgray', base_size
        
        # Get normalized attributes
        detective_positions = getattr(state, 'detective_positions', [])
        MrX_position = getattr(state, 'MrX_position', 0)
        MrX_visible = getattr(state, 'MrX_visible', True)
        
        # UI-specific attributes
        active_player_positions = getattr(state, 'active_player_positions', [])
        detective_selections = getattr(state, 'detective_selections', [])
        
        # Handle special cases in priority order
        
        # 1. Node with both detective and Mr. X (collision)
        if node in detective_positions and node == MrX_position:
            return 'yellow', int(base_size * 1.2)
        
        # 2. Active player positions (current player's position - highlighted)
        if node in active_player_positions:
            if node in detective_positions:
                return 'cyan', base_size  # Active detective
            else:
                return 'orange', base_size  # Active Mr. X
        
        # 3. Detective selections (selected destinations)
        selected_destinations = []
        for selection in detective_selections:
            if isinstance(selection, tuple) and len(selection) >= 1:
                selected_destinations.append(selection[0])
            elif isinstance(selection, int):
                selected_destinations.append(selection)
        
        if node in selected_destinations:
            return 'green', base_size  # Selected destination
        
        # 4. Detective positions (normal blue)
        if node in detective_positions:
            return 'blue', base_size
        
        # 5. Mr. X position
        if node == MrX_position:
            # Check visibility for Shadow Chase games
            if hasattr(self.game, '__class__') and 'ShadowChase' in self.game.__class__.__name__:
                if MrX_visible or mode == "history":
                    return 'red', base_size
                else:
                    return 'lightgray', base_size
            else:
                return 'red', base_size
        
        # 6. Empty nodes (default)
        return 'lightgray', base_size
    


    def _check_game_over_status(self, state):
        """Check if game is over at given state"""
        original_state = getattr(self.game, 'game_state', None)
        try:
            self.game.game_state = state
            return self.game.is_game_over() if hasattr(self.game, 'is_game_over') else False
        finally:
            self.game.game_state = original_state



    def _get_winner_at_step(self, state):
        """Get winner at given state"""
        original_state = getattr(self.game, 'game_state', None)
        try:
            self.game.game_state = state
            if hasattr(self.game, 'get_winner'):
                winner_obj = self.game.get_winner()
                return winner_obj.value if hasattr(winner_obj, 'value') else str(winner_obj)
        finally:
            self.game.game_state = original_state
        return None

