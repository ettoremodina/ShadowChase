#!/usr/bin/env python3
"""
Board overlay calibration script for Scotland Yard game.
This script helps align the board image with the graph node positions.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
import numpy as np
from ShadowChase.services.board_loader import create_extracted_board_game, load_board_metadata
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class BoardCalibrator:
    def __init__(self):
        self.game = create_extracted_board_game(num_detectives=3)
        self.metadata = load_board_metadata()
        self.board_image = None
        self.board_image_path = "data/board.png"
        
        # Calibration parameters
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.image_alpha = 0.7
        
        # UI components
        self.root = None
        self.fig = None
        self.ax = None
        self.canvas = None
        
        # Load board image and calculate initial positions
        self.load_board_image()
        self.calculate_positions()
        
    def load_board_image(self):
        """Load the board image"""
        try:
            if os.path.exists(self.board_image_path):
                self.board_image = mpimg.imread(self.board_image_path)
                print(f"Board image loaded: {self.board_image.shape}")
                print(f"Image dimensions: {self.board_image.shape[1]} x {self.board_image.shape[0]} pixels")
            else:
                print(f"Board image not found at: {self.board_image_path}")
        except Exception as e:
            print(f"Error loading board image: {e}")
            
    def calculate_positions(self):
        """Calculate initial normalized positions"""
        if not hasattr(self.game, 'node_positions') or not self.game.node_positions:
            print("No node positions available in game")
            return
            
        positions = self.game.node_positions
        bounds = self.metadata.get('board_bounds', {})
        
        print(f"Board bounds from metadata: {bounds}")
        print(f"Number of nodes: {len(positions)}")
        
        # Get actual coordinate ranges from the data
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        actual_x_min, actual_x_max = min(x_coords), max(x_coords)
        actual_y_min, actual_y_max = min(y_coords), max(y_coords)
        
        print(f"Actual coordinate ranges:")
        print(f"  X: {actual_x_min} to {actual_x_max} (range: {actual_x_max - actual_x_min})")
        print(f"  Y: {actual_y_min} to {actual_y_max} (range: {actual_y_max - actual_y_min})")
        
        # Calculate aspect ratios
        coord_aspect = (actual_x_max - actual_x_min) / (actual_y_max - actual_y_min)
        if self.board_image is not None:
            image_aspect = self.board_image.shape[1] / self.board_image.shape[0]
            print(f"Coordinate aspect ratio: {coord_aspect:.3f}")
            print(f"Image aspect ratio: {image_aspect:.3f}")
            print(f"Aspect ratio difference: {abs(coord_aspect - image_aspect):.3f}")
        
        # Store original positions for reference
        self.original_positions = positions.copy()
        self.bounds = {
            'x_min': actual_x_min, 'x_max': actual_x_max,
            'y_min': actual_y_min, 'y_max': actual_y_max
        }
        
    def normalize_positions(self):
        """Normalize positions with current calibration parameters"""
        if not hasattr(self, 'original_positions'):
            return {}
            
        normalized_pos = {}
        
        # Apply scaling and offset to the bounds
        x_range = (self.bounds['x_max'] - self.bounds['x_min']) * self.x_scale
        y_range = (self.bounds['y_max'] - self.bounds['y_min']) * self.y_scale
        
        for node, (x, y) in self.original_positions.items():
            # Normalize to [-1, 1] range with calibration parameters
            norm_x = 2 * ((x - self.bounds['x_min']) / x_range) - 1 + self.x_offset
            norm_y = -(2 * ((y - self.bounds['y_min']) / y_range) - 1) + self.y_offset  # Flip Y and apply offset
            
            normalized_pos[node] = (norm_x, norm_y)
            
        return normalized_pos
        
    def create_ui(self):
        """Create calibration UI"""
        self.root = tk.Tk()
        self.root.title("Board Overlay Calibrator")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Create controls
        self.create_controls(control_frame)
        
        # Right panel for graph
        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial draw
        self.update_display()
        
    def create_controls(self, parent):
        """Create calibration controls"""
        ttk.Label(parent, text="Board Overlay Calibration", font=('Arial', 14, 'bold')).pack(pady=(0, 20))
        
        # X Offset
        ttk.Label(parent, text="X Offset:").pack(anchor="w")
        self.x_offset_var = tk.DoubleVar(value=self.x_offset)
        x_offset_scale = ttk.Scale(parent, from_=-1.0, to=1.0, variable=self.x_offset_var, 
                                  orient="horizontal", command=self.on_parameter_change)
        x_offset_scale.pack(fill="x", pady=(0, 10))
        
        # Y Offset
        ttk.Label(parent, text="Y Offset:").pack(anchor="w")
        self.y_offset_var = tk.DoubleVar(value=self.y_offset)
        y_offset_scale = ttk.Scale(parent, from_=-1.0, to=1.0, variable=self.y_offset_var,
                                  orient="horizontal", command=self.on_parameter_change)
        y_offset_scale.pack(fill="x", pady=(0, 10))
        
        # X Scale
        ttk.Label(parent, text="X Scale:").pack(anchor="w")
        self.x_scale_var = tk.DoubleVar(value=self.x_scale)
        x_scale_scale = ttk.Scale(parent, from_=0.5, to=2.0, variable=self.x_scale_var,
                                 orient="horizontal", command=self.on_parameter_change)
        x_scale_scale.pack(fill="x", pady=(0, 10))
        
        # Y Scale
        ttk.Label(parent, text="Y Scale:").pack(anchor="w")
        self.y_scale_var = tk.DoubleVar(value=self.y_scale)
        y_scale_scale = ttk.Scale(parent, from_=0.5, to=2.0, variable=self.y_scale_var,
                                 orient="horizontal", command=self.on_parameter_change)
        y_scale_scale.pack(fill="x", pady=(0, 10))
        
        # Image Alpha
        ttk.Label(parent, text="Image Alpha:").pack(anchor="w")
        self.alpha_var = tk.DoubleVar(value=self.image_alpha)
        alpha_scale = ttk.Scale(parent, from_=0.0, to=1.0, variable=self.alpha_var,
                               orient="horizontal", command=self.on_parameter_change)
        alpha_scale.pack(fill="x", pady=(0, 20))
        
        # Buttons
        ttk.Button(parent, text="Reset to Default", command=self.reset_parameters).pack(fill="x", pady=5)
        ttk.Button(parent, text="Save Calibration", command=self.save_calibration).pack(fill="x", pady=5)
        ttk.Button(parent, text="Load Calibration", command=self.load_calibration).pack(fill="x", pady=5)
        
        # Info display
        info_frame = ttk.LabelFrame(parent, text="Information")
        info_frame.pack(fill="x", pady=(20, 0))
        
        self.info_text = tk.Text(info_frame, height=10, width=30, wrap=tk.WORD)
        self.info_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Update info
        self.update_info_display()
        
    def on_parameter_change(self, value=None):
        """Handle parameter changes"""
        self.x_offset = self.x_offset_var.get()
        self.y_offset = self.y_offset_var.get()
        self.x_scale = self.x_scale_var.get()
        self.y_scale = self.y_scale_var.get()
        self.image_alpha = self.alpha_var.get()
        
        self.update_display()
        self.update_info_display()
        
    def reset_parameters(self):
        """Reset calibration parameters to default"""
        self.x_offset_var.set(0.0)
        self.y_offset_var.set(0.0)
        self.x_scale_var.set(1.0)
        self.y_scale_var.set(1.0)
        self.alpha_var.set(0.7)
        self.on_parameter_change()
        
    def update_display(self):
        """Update the display with current calibration"""
        if not self.ax:
            return
            
        self.ax.clear()
        
        # Draw board image if available
        if self.board_image is not None:
            self.ax.imshow(self.board_image, extent=[-1, 1, -1, 1], alpha=self.image_alpha, aspect='auto')
        
        # Get normalized positions with current calibration
        pos = self.normalize_positions()
        
        if pos:
            # Draw all nodes
            x_coords = [pos[node][0] for node in self.game.graph.nodes() if node in pos]
            y_coords = [pos[node][1] for node in self.game.graph.nodes() if node in pos]
            
            self.ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8, edgecolors='black', linewidth=1)
            
            # Draw node labels for a subset of nodes to avoid clutter
            sample_nodes = list(self.game.graph.nodes())[::10]  # Every 10th node
            for node in sample_nodes:
                if node in pos:
                    x, y = pos[node]
                    self.ax.annotate(str(node), (x, y), xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, color='blue', weight='bold')
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.set_title("Board Overlay Calibration", fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def update_info_display(self):
        """Update information display"""
        if not hasattr(self, 'info_text'):
            return
            
        info = f"""Current Parameters:
X Offset: {self.x_offset:.3f}
Y Offset: {self.y_offset:.3f}
X Scale: {self.x_scale:.3f}
Y Scale: {self.y_scale:.3f}
Image Alpha: {self.image_alpha:.3f}

Board Info:
Nodes: {len(self.game.graph.nodes())}
Edges: {len(self.game.graph.edges())}

Instructions:
- Adjust offsets to move nodes
- Adjust scales to resize coordinate space
- Red dots are graph nodes
- Blue numbers are node IDs (sample)
"""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        
    def save_calibration(self):
        """Save current calibration parameters"""
        calibration = {
            'x_offset': self.x_offset,
            'y_offset': self.y_offset,
            'x_scale': self.x_scale,
            'y_scale': self.y_scale,
            'image_alpha': self.image_alpha,
            'bounds': self.bounds
        }
        
        try:
            with open('data/board_calibration.json', 'w') as f:
                json.dump(calibration, f, indent=2)
            messagebox.showinfo("Success", "Calibration saved to data/board_calibration.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save calibration: {e}")
            
    def load_calibration(self):
        """Load calibration parameters"""
        try:
            with open('data/board_calibration.json', 'r') as f:
                calibration = json.load(f)
                
            self.x_offset_var.set(calibration.get('x_offset', 0.0))
            self.y_offset_var.set(calibration.get('y_offset', 0.0))
            self.x_scale_var.set(calibration.get('x_scale', 1.0))
            self.y_scale_var.set(calibration.get('y_scale', 1.0))
            self.alpha_var.set(calibration.get('image_alpha', 0.7))
            
            self.on_parameter_change()
            messagebox.showinfo("Success", "Calibration loaded from data/board_calibration.json")
        except FileNotFoundError:
            messagebox.showwarning("Warning", "No calibration file found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration: {e}")
            
    def run(self):
        """Run the calibrator"""
        self.create_ui()
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting Board Overlay Calibrator...")
    calibrator = BoardCalibrator()
    calibrator.run()

if __name__ == "__main__":
    main()
