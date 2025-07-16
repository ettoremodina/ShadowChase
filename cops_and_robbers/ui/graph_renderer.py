"""
Graph Renderer Component

Handles NetworkX-based graph visualization with matplotlib for the Scotland Yard game.
Provides color-coded transport types and interactive node/edge selection.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable
from ..core.game import Game, Player, TransportType


class GraphRenderer:
    """Handles graph visualization and interaction"""
    
    # Color schemes for different transport types
    TRANSPORT_COLORS = {
        TransportType.TAXI: '#FFD700',      # Yellow
        TransportType.BUS: '#4169E1',       # Royal Blue  
        TransportType.UNDERGROUND: '#DC143C', # Crimson
        TransportType.FERRY: '#228B22',     # Forest Green
        TransportType.BLACK: '#2F2F2F'      # Dark Gray
    }
    
    # Node colors for players
    PLAYER_COLORS = {
        'cop': '#0066CC',        # Blue
        'robber': '#CC0000',     # Red
        'selected': '#FF6600',   # Orange
        'default': '#CCCCCC',    # Light Gray
        'highlighted': '#FFFF00' # Yellow highlight
    }
    
    def __init__(self, parent_frame: tk.Frame, game: Game):
        """Initialize graph renderer"""
        self.parent_frame = parent_frame
        self.game = game
        self.graph = game.graph
        
        # Event callbacks
        self.on_node_click: Optional[Callable[[int], None]] = None
        self.on_edge_click: Optional[Callable[[int, int], None]] = None
        
        # State tracking
        self.selected_nodes: Set[int] = set()
        self.highlighted_nodes: Set[int] = set()
        self.highlighted_edges: Set[Tuple[int, int]] = set()
        self.available_moves: Set[int] = set()
        
        # Layout and rendering
        self.pos = None
        self.figure = None
        self.canvas = None
        self.ax = None
        
        self._setup_matplotlib()
        self._calculate_layout()
    
    def _setup_matplotlib(self):
        """Setup matplotlib figure and canvas"""
        # Create figure
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click events
        self.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Styling
        self.figure.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.ax.axis('off')
    
    def _calculate_layout(self):
        """Calculate optimal graph layout"""
        # Try different layouts based on graph properties
        n_nodes = self.graph.number_of_nodes()
        
        if n_nodes <= 20:
            # Small graphs - use spring layout with more iterations
            self.pos = nx.spring_layout(self.graph, k=2, iterations=100, seed=42)
        elif nx.is_planar(self.graph):
            # Planar graphs - use planar layout if possible
            try:
                self.pos = nx.planar_layout(self.graph)
            except:
                self.pos = nx.spring_layout(self.graph, k=1.5, iterations=50, seed=42)
        else:
            # Large graphs - use faster layout
            self.pos = nx.spring_layout(self.graph, k=1, iterations=30, seed=42)
        
        # Normalize positions to better fit the canvas
        if self.pos:
            positions = np.array(list(self.pos.values()))
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            range_pos = max_pos - min_pos
            
            # Add some padding and normalize
            for node in self.pos:
                self.pos[node] = (self.pos[node] - min_pos) / range_pos * 0.8 + 0.1
    
    def render(self):
        """Render the complete graph with current state"""
        self.ax.clear()
        self.ax.axis('off')
        
        if not self.pos:
            return
        
        # Draw edges with transport type colors
        self._draw_edges()
        
        # Draw nodes with player positions
        self._draw_nodes()
        
        # Add labels
        self._draw_labels()
        
        # Refresh canvas
        self.canvas.draw()
    
    def _draw_edges(self):
        """Draw graph edges with transport type coloring"""
        # Group edges by transport type
        edge_groups = {}
        
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            edge_type = data.get('edge_type', 1)  # Default to taxi
            
            # Map edge_type to TransportType
            transport_type = self._get_transport_type(edge_type)
            
            if transport_type not in edge_groups:
                edge_groups[transport_type] = []
            edge_groups[transport_type].append((source, target))
        
        # Draw each transport type with its color
        for transport_type, edges in edge_groups.items():
            color = self.TRANSPORT_COLORS.get(transport_type, '#CCCCCC')
            
            # Determine edge width and style
            width = 3.0 if (edges[0] if edges else None) in self.highlighted_edges else 1.5
            alpha = 1.0 if any(edge in self.highlighted_edges for edge in edges) else 0.6
            
            nx.draw_networkx_edges(
                self.graph, self.pos,
                edgelist=edges,
                edge_color=color,
                width=width,
                alpha=alpha,
                ax=self.ax
            )
    
    def _draw_nodes(self):
        """Draw graph nodes with player positions and states"""
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            color, size = self._get_node_appearance(node)
            node_colors.append(color)
            node_sizes.append(size)
        
        nx.draw_networkx_nodes(
            self.graph, self.pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=self.ax
        )
    
    def _draw_labels(self):
        """Draw node labels"""
        # Only show labels for important nodes or when graph is small
        if self.graph.number_of_nodes() <= 50:
            nx.draw_networkx_labels(
                self.graph, self.pos,
                font_size=8,
                font_weight='bold',
                ax=self.ax
            )
    
    def _get_node_appearance(self, node: int) -> Tuple[str, int]:
        """Get color and size for a node based on current game state"""
        base_size = 300
        
        # Check if node has players
        if hasattr(self.game, 'game_state') and self.game.game_state:
            state = self.game.game_state
            
            # Cop positions
            if node in state.cop_positions:
                return self.PLAYER_COLORS['cop'], base_size * 1.5
            
            # Robber position  
            if node == state.robber_position:
                return self.PLAYER_COLORS['robber'], base_size * 1.5
        
        # Selected nodes
        if node in self.selected_nodes:
            return self.PLAYER_COLORS['selected'], base_size * 1.3
        
        # Highlighted nodes (available moves)
        if node in self.highlighted_nodes:
            return self.PLAYER_COLORS['highlighted'], base_size * 1.2
        
        # Available moves
        if node in self.available_moves:
            return self.PLAYER_COLORS['highlighted'], base_size * 1.1
        
        # Default
        return self.PLAYER_COLORS['default'], base_size
    
    def _get_transport_type(self, edge_type: int) -> TransportType:
        """Convert edge_type integer to TransportType enum"""
        transport_map = {
            1: TransportType.TAXI,
            2: TransportType.BUS,
            3: TransportType.UNDERGROUND,
            4: TransportType.BLACK,
            5: TransportType.FERRY
        }
        return transport_map.get(edge_type, TransportType.TAXI)
    
    def _on_click(self, event):
        """Handle mouse clicks on the graph"""
        if event.inaxes != self.ax:
            return
        
        # Find closest node
        if not self.pos or event.xdata is None or event.ydata is None:
            return
        
        click_pos = np.array([event.xdata, event.ydata])
        closest_node = None
        min_distance = float('inf')
        
        for node, pos in self.pos.items():
            distance = np.linalg.norm(np.array(pos) - click_pos)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Check if click is close enough to a node
        if min_distance < 0.05:  # Threshold for node selection
            if self.on_node_click:
                self.on_node_click(closest_node)
    
    def set_selected_nodes(self, nodes: Set[int]):
        """Set which nodes are selected"""
        self.selected_nodes = nodes.copy()
        self.render()
    
    def set_highlighted_nodes(self, nodes: Set[int]):
        """Set which nodes should be highlighted"""
        self.highlighted_nodes = nodes.copy()
        self.render()
    
    def set_available_moves(self, moves: Set[int]):
        """Set which nodes are available as moves"""
        self.available_moves = moves.copy()
        self.render()
    
    def set_highlighted_edges(self, edges: Set[Tuple[int, int]]):
        """Set which edges should be highlighted"""
        self.highlighted_edges = edges.copy()
        self.render()
    
    def clear_selections(self):
        """Clear all selections and highlights"""
        self.selected_nodes.clear()
        self.highlighted_nodes.clear()
        self.highlighted_edges.clear()
        self.available_moves.clear()
        self.render()
    
    def set_node_click_callback(self, callback: Callable[[int], None]):
        """Set callback for node clicks"""
        self.on_node_click = callback
    
    def set_edge_click_callback(self, callback: Callable[[int, int], None]):
        """Set callback for edge clicks"""
        self.on_edge_click = callback
    
    def update_game_state(self, game: Game):
        """Update the game reference and re-render"""
        self.game = game
        self.render()
    
    def export_image(self, filename: str):
        """Export current graph visualization to image file"""
        self.figure.savefig(filename, dpi=300, bbox_inches='tight')
