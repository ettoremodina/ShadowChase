import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class UnifiedGameState:
    """Unified state representation for all visualizers"""
    # Core game state
    detective_positions: List[int]
    mr_x_position: int
    turn_count: int
    current_turn: str  # 'detective' or 'mr_x'
    
    # Mr. X visibility
    mr_x_visible: bool = True
    
    # Tickets
    detective_tickets: Optional[Dict] = None
    mr_x_tickets: Optional[Dict] = None
    
    # Move information
    last_move_ticket: Optional[str] = None
    previous_position: Optional[int] = None
    double_move_active: bool = False
    
    # Game status
    game_over: bool = False
    winner: Optional[str] = None
    
    # UI-specific state (for GameVisualizer)
    setup_mode: bool = False
    selected_positions: List[int] = None
    active_player_positions: List[int] = None
    detective_selections: List[int] = None
    
    def __post_init__(self):
        if self.selected_positions is None:
            self.selected_positions = []
        if self.active_player_positions is None:
            self.active_player_positions = []
        if self.detective_selections is None:
            self.detective_selections = []


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
    
    def get_unified_state(self, mode: str = "live", step: Optional[int] = None, 
                         ui_state: Optional[Dict] = None) -> UnifiedGameState:
        """
        Get unified state representation for any visualizer mode
        
        Args:
            mode: "live" (current game state), "history" (from game history)
            step: Step number for history mode
            ui_state: UI-specific state variables (for GameVisualizer)
        
        Returns:
            UnifiedGameState object with normalized attributes
        """
        if mode == "live":
            return self._get_live_state(ui_state)
        elif mode == "history":
            return self._get_historical_state(step)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _get_live_state(self, ui_state: Optional[Dict] = None) -> UnifiedGameState:
        """Get unified state from current game state"""
        # Get current game state
        current_state = getattr(self.game, 'game_state', None)
        
        if not current_state:
            # No active game - return setup state
            return UnifiedGameState(
                detective_positions=[],
                mr_x_position=0,
                turn_count=0,
                current_turn='detective',
                setup_mode=True,
                selected_positions=ui_state.get('selected_positions', []) if ui_state else [],
                active_player_positions=ui_state.get('active_player_positions', []) if ui_state else [],
                detective_selections=ui_state.get('detective_selections', []) if ui_state else []
            )
        
        # Normalize attribute names from current state
        detective_positions = self._get_attribute(current_state, ['detective_positions'])
        mr_x_position = self._get_attribute(current_state, ['MrX_position', 'mr_x_position'])
        turn_count = self._get_attribute(current_state, ['turn_count'], default=0)
        
        # Normalize turn information
        current_turn = self._normalize_turn(current_state)
        
        # Get Mr. X visibility
        mr_x_visible = self._get_attribute(current_state, ['mr_x_visible'], default=True)
        
        # Get tickets
        detective_tickets = self._get_attribute(current_state, ['detective_tickets'])
        mr_x_tickets = self._get_attribute(current_state, ['mr_x_tickets'])
        
        # Get move information
        last_move_ticket = self._get_attribute(current_state, ['last_move_ticket'])
        previous_position = self._get_attribute(current_state, ['previous_position'])
        double_move_active = self._get_attribute(current_state, ['double_move_active'], default=False)
        
        # Check game status
        game_over = self.game.is_game_over() if hasattr(self.game, 'is_game_over') else False
        winner = None
        if game_over and hasattr(self.game, 'get_winner'):
            winner_obj = self.game.get_winner()
            winner = winner_obj.value if hasattr(winner_obj, 'value') else str(winner_obj)
        
        return UnifiedGameState(
            detective_positions=detective_positions or [],
            mr_x_position=mr_x_position or 0,
            turn_count=turn_count,
            current_turn=current_turn,
            mr_x_visible=mr_x_visible,
            detective_tickets=detective_tickets,
            mr_x_tickets=mr_x_tickets,
            last_move_ticket=last_move_ticket,
            previous_position=previous_position,
            double_move_active=double_move_active,
            game_over=game_over,
            winner=winner,
            setup_mode=False,
            selected_positions=ui_state.get('selected_positions', []) if ui_state else [],
            active_player_positions=ui_state.get('active_player_positions', []) if ui_state else [],
            detective_selections=ui_state.get('detective_selections', []) if ui_state else []
        )
    
    def _get_historical_state(self, step: int) -> UnifiedGameState:
        """Get unified state from game history"""
        if not hasattr(self.game, 'game_history') or not self.game.game_history:
            raise ValueError("No game history available")
        
        if step >= len(self.game.game_history):
            raise ValueError(f"Step {step} is beyond game history length {len(self.game.game_history)}")
        
        historical_state = self.game.game_history[step]
        
        # Normalize attribute names from historical state
        detective_positions = self._get_attribute(historical_state, ['detective_positions'])
        mr_x_position = self._get_attribute(historical_state, ['MrX_position', 'mr_x_position'])
        turn_count = self._get_attribute(historical_state, ['turn_count'], default=step)
        
        # Normalize turn information
        current_turn = self._normalize_turn(historical_state)
        
        # Get Mr. X visibility
        mr_x_visible = self._get_attribute(historical_state, ['mr_x_visible'], default=True)
        
        # Get tickets
        detective_tickets = self._get_attribute(historical_state, ['detective_tickets'])
        mr_x_tickets = self._get_attribute(historical_state, ['mr_x_tickets'])
        
        # Get move information
        last_move_ticket = self._get_attribute(historical_state, ['last_move_ticket'])
        previous_position = self._get_attribute(historical_state, ['previous_position'])
        double_move_active = self._get_attribute(historical_state, ['double_move_active'], default=False)
        
        # Check game status at this step
        game_over = False
        winner = None
        
        # Temporarily set game state to check game over status
        original_state = getattr(self.game, 'game_state', None)
        try:
            self.game.game_state = historical_state
            if hasattr(self.game, 'is_game_over'):
                game_over = self.game.is_game_over()
                if game_over and hasattr(self.game, 'get_winner'):
                    winner_obj = self.game.get_winner()
                    winner = winner_obj.value if hasattr(winner_obj, 'value') else str(winner_obj)
        finally:
            self.game.game_state = original_state
        
        return UnifiedGameState(
            detective_positions=detective_positions or [],
            mr_x_position=mr_x_position or 0,
            turn_count=turn_count,
            current_turn=current_turn,
            mr_x_visible=mr_x_visible,
            detective_tickets=detective_tickets,
            mr_x_tickets=mr_x_tickets,
            last_move_ticket=last_move_ticket,
            previous_position=previous_position,
            double_move_active=double_move_active,
            game_over=game_over,
            winner=winner,
            setup_mode=False
        )
    
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
            return 'mr_x'
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
        for i in range(self.game.num_detectives):
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
        # Get unified state
        unified_state = self.get_unified_state(mode, step, ui_state)
        
        node_colors = []
        node_sizes = []
        
        for node in self.game.graph.nodes():
            color, size = self._get_node_color_and_size(mode, node, unified_state, node_size)
            node_colors.append(color)
            node_sizes.append(size)
        
        return node_colors, node_sizes

    def _get_node_color_and_size(self, mode: str, node: int, state: UnifiedGameState, base_size: int) -> tuple:
        """Get color and size for a single node based on unified state"""
        
        # Setup mode (GameVisualizer only)
        if state.setup_mode:
            if node in state.selected_positions:
                color = 'blue' if len(state.selected_positions) <= self.game.num_detectives else 'red'
                return color, base_size
            else:
                return 'lightgray', base_size
        
        # Game mode - handle special cases first
        
        # Node with both detective and Mr. X (collision)
        if node in state.detective_positions and node == state.mr_x_position:
            return 'yellow', int(base_size * 1.2)
        
        # Active player positions (GameVisualizer specific)
        if node in state.active_player_positions:
            if node in state.detective_positions:
                return 'cyan', base_size
            else:
                return 'orange', base_size
        
        # Detective selections (GameVisualizer specific)
        if node in state.detective_selections:
            return 'purple', base_size
        
        # Detective positions
        if node in state.detective_positions:
            return 'blue', base_size
        
        # Mr. X position
        if node == state.mr_x_position:
            # Check visibility for Shadow Chase games
            if hasattr(self.game, '__class__') and 'ShadowChase' in self.game.__class__.__name__:
                if state.mr_x_visible or mode == "history":
                    return 'red', base_size
                else:
                    return 'lightgray', base_size
            else:
                return 'red', base_size
        
        # Empty nodes
        # For video export, make empty nodes slightly smaller
        if hasattr(self, '__class__') and 'Video' in self.__class__.__name__:
            return 'lightgray', int(base_size * 0.8)
        else:
            return 'lightgray', base_size
    


