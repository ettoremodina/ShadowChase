#!/usr/bin/env python3
"""
Shadow Chase Board Creator

Interactive tool to extract nodes and edges from a Shadow Chase board image.
Run with phase argument: python createBoard.py --phase [nodes|taxi|bus|underground|ferry]

Phase-specific controls:
NODES PHASE:
- Left click: Place node
- Right click: Cancel current operation
- 'z': Toggle zoom mode
- 's': Save progress
- 'l': Load progress
- 'c': Complete phase and save
- 'q': Quit

EDGE PHASES (taxi/bus/underground/ferry):
- Left click: Select node for edge
- Right click: Cancel current selection
- 'z': Toggle zoom mode
- 's': Save progress
- 'l': Load progress
- 'c': Complete phase and save
- 'q': Quit

General controls:
- 'Backspace': Undo last action
"""

import cv2
import numpy as np
import json
import os
import argparse
from enum import Enum
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import messagebox, filedialog
from dataclasses import dataclass, asdict

class Phase(Enum):
    NODES = "nodes"
    TAXI = "taxi"
    BUS = "bus"
    UNDERGROUND = "underground"
    FERRY = "ferry"

class Mode(Enum):
    NODE_PLACEMENT = "Node Placement"
    EDGE_PLACEMENT = "Edge Placement"

@dataclass
class Node:
    id: int
    x: float
    y: float
    
@dataclass
class Edge:
    node1: int
    node2: int
    transport_type: str

class ShadowChaseBoardCreator:
    def __init__(self, image_path: str, phase: Phase):
        self.image_path = image_path
        self.phase = phase
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        self.image = self.original_image.copy()
        self.display_image = self.image.copy()
        
        # Data structures
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.next_node_id = 1
        
        # UI state
        self.mode = Mode.NODE_PLACEMENT if phase == Phase.NODES else Mode.EDGE_PLACEMENT
        self.selected_node = None
        self.zoom_mode = False
        self.zoom_factor = 1.0
        self.zoom_center = (0, 0)
        
        # Colors for different elements (BGR format)
        self.colors = {
            'node': (0, 0, 0),          # Black - most visible
            'taxi': (0, 0, 0),      # Yellow (BGR format) - for taxi edges
            'bus': (0, 0, 0),           # Black
            'underground': (0, 0, 0),   # Black
            'ferry': (0, 0, 0),         # Black
            'selected': (0, 255, 255),  # Yellow for selected node
        }
        
        # History for undo functionality
        self.history = []
        
        self.window_name = f"Shadow Chase Board Creator - {phase.value.upper()} Phase"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def save_state(self):
        """Save current state to history for undo functionality"""
        state = {
            'nodes': dict(self.nodes),
            'edges': list(self.edges),
            'next_node_id': self.next_node_id
        }
        self.history.append(state)
        # Keep only last 50 states to prevent memory issues
        if len(self.history) > 50:
            self.history.pop(0)
    
    def undo_last_action(self):
        """Undo the last action"""
        if self.history:
            state = self.history.pop()
            self.nodes = state['nodes']
            self.edges = state['edges']
            self.next_node_id = state['next_node_id']
            self.selected_node = None
            print("Undid last action")
        else:
            print("Nothing to undo")
    
    def get_display_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Convert image coordinates to display coordinates (accounting for zoom)"""
        if self.zoom_mode:
            # Apply zoom transformation
            display_x = int((x - self.zoom_center[0]) * self.zoom_factor + self.display_image.shape[1] // 2)
            display_y = int((y - self.zoom_center[1]) * self.zoom_factor + self.display_image.shape[0] // 2)
            return display_x, display_y
        return x, y
    
    def get_image_coordinates(self, display_x: int, display_y: int) -> Tuple[int, int]:
        """Convert display coordinates to image coordinates"""
        if self.zoom_mode:
            # Reverse zoom transformation
            x = int((display_x - self.display_image.shape[1] // 2) / self.zoom_factor + self.zoom_center[0])
            y = int((display_y - self.display_image.shape[0] // 2) / self.zoom_factor + self.zoom_center[1])
            return x, y
        return display_x, display_y
    
    def find_nearest_node(self, x: int, y: int, threshold: int = 20) -> Optional[int]:
        """Find the nearest node within threshold distance"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node in self.nodes.items():
            distance = np.sqrt((node.x - x)**2 + (node.y - y)**2)
            if distance < threshold and distance < min_distance:
                min_distance = distance
                nearest_node = node_id
                
        return nearest_node
    
    def add_node(self, x: int, y: int):
        """Add a new node at the specified coordinates"""
        self.save_state()
        node = Node(self.next_node_id, x, y)
        self.nodes[self.next_node_id] = node
        print(f"Added node {self.next_node_id} at ({x}, {y})")
        self.next_node_id += 1
    
    def add_edge(self, node1_id: int, node2_id: int, transport_type: str):
        """Add an edge between two nodes"""
        # Check if edge already exists
        for edge in self.edges:
            if (edge.node1 == node1_id and edge.node2 == node2_id and edge.transport_type == transport_type) or \
               (edge.node1 == node2_id and edge.node2 == node1_id and edge.transport_type == transport_type):
                print(f"Edge already exists between nodes {node1_id} and {node2_id} for {transport_type}")
                return
        
        self.save_state()
        edge = Edge(node1_id, node2_id, transport_type)
        self.edges.append(edge)
        print(f"Added {transport_type} edge between nodes {node1_id} and {node2_id}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates to image coordinates
            img_x, img_y = self.get_image_coordinates(x, y)
            
            if self.phase == Phase.NODES:
                self.add_node(img_x, img_y)
            else:
                # Edge placement for specific transport type
                nearest_node = self.find_nearest_node(img_x, img_y)
                if nearest_node is not None:
                    if self.selected_node is None:
                        self.selected_node = nearest_node
                        print(f"Selected node {nearest_node}. Click another node to create {self.phase.value} edge.")
                    else:
                        if nearest_node != self.selected_node:
                            self.add_edge(self.selected_node, nearest_node, self.phase.value)
                        self.selected_node = None
                else:
                    print("No node found near click location")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancel current operation
            if self.selected_node is not None:
                print("Cancelled edge selection")
                self.selected_node = None
    
    def update_zoom(self, center_x: int, center_y: int):
        """Update zoom view"""
        if self.zoom_mode:
            self.zoom_center = self.get_image_coordinates(center_x, center_y)
            # Create zoomed view
            h, w = self.image.shape[:2]
            zoom_w = int(w / self.zoom_factor)
            zoom_h = int(h / self.zoom_factor)
            
            # Calculate crop boundaries
            x1 = max(0, self.zoom_center[0] - zoom_w // 2)
            y1 = max(0, self.zoom_center[1] - zoom_h // 2)
            x2 = min(w, x1 + zoom_w)
            y2 = min(h, y1 + zoom_h)
            
            cropped = self.image[y1:y2, x1:x2]
            self.display_image = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            self.display_image = self.image.copy()
    
    def draw_elements(self):
        """Draw all nodes and edges on the display image"""
        self.display_image = self.original_image.copy()
        
        # Dim the image for taxi mode to make yellow edges more visible
        if self.phase == Phase.TAXI:
            # Apply dimming - multiply by 0.4 to make it much darker
            self.display_image = (self.display_image * 0.4).astype(np.uint8)
        
        # Only draw edges if we're in nodes phase OR if they match the current phase
        if self.phase == Phase.NODES:
            # In nodes phase, draw all edges
            for edge in self.edges:
                if edge.node1 in self.nodes and edge.node2 in self.nodes:
                    node1 = self.nodes[edge.node1]
                    node2 = self.nodes[edge.node2]
                    
                    color = self.colors.get(edge.transport_type, (0, 0, 0))
                    thickness = 4 if edge.transport_type == 'underground' else 3
                    
                    cv2.line(self.display_image, 
                            (int(node1.x), int(node1.y)), 
                            (int(node2.x), int(node2.y)), 
                            color, thickness)
        else:
            # In edge phases, only draw edges of the current transport type
            for edge in self.edges:
                if edge.transport_type == self.phase.value and edge.node1 in self.nodes and edge.node2 in self.nodes:
                    node1 = self.nodes[edge.node1]
                    node2 = self.nodes[edge.node2]
                    
                    color = self.colors.get(edge.transport_type, (0, 0, 0))
                    # Make taxi edges extra thick and visible
                    thickness = 6 if self.phase == Phase.TAXI else (4 if edge.transport_type == 'underground' else 3)
                    
                    cv2.line(self.display_image, 
                            (int(node1.x), int(node1.y)), 
                            (int(node2.x), int(node2.y)), 
                            color, thickness)
        
        # Draw nodes
        for node_id, node in self.nodes.items():
            color = self.colors['selected'] if node_id == self.selected_node else self.colors['node']
            # Use larger, more visible circles - extra large in taxi mode for better visibility
            circle_size = 10 if self.phase == Phase.TAXI else 8
            border_size = 3 if self.phase == Phase.TAXI else 2
            
            cv2.circle(self.display_image, (int(node.x), int(node.y)), circle_size, color, -1)
            cv2.circle(self.display_image, (int(node.x), int(node.y)), circle_size, (255, 255, 255), border_size)  # White border
            
            # Draw node ID with better visibility
            text = str(node_id)
            font_size = 0.7 if self.phase == Phase.TAXI else 0.6
            text_thickness = 3 if self.phase == Phase.TAXI else 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)[0]
            text_x = int(node.x) - text_size[0] // 2
            text_y = int(node.y) + text_size[1] // 2
            
            # White background for text
            cv2.putText(self.display_image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness + 2)
            # Black text on top
            cv2.putText(self.display_image, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), text_thickness)
        
        # Apply zoom if active
        if self.zoom_mode:
            self.update_zoom(self.display_image.shape[1] // 2, self.display_image.shape[0] // 2)
    
    def draw_ui_info(self):
        """Draw UI information on the image"""
        info_lines = [
            f"Phase: {self.phase.value.upper()}",
            f"Nodes: {len(self.nodes)}",
            f"Edges ({self.phase.value}): {len([e for e in self.edges if e.transport_type == self.phase.value])}" if self.phase != Phase.NODES else f"Total Edges: {len(self.edges)}",
            f"Zoom: {'ON' if self.zoom_mode else 'OFF'}",
            "",
            "Controls:",
            "'c' - Complete phase",
            "'s' - Save progress", 
            "'l' - Load progress",
            "'z' - Toggle zoom",
            "'q' - Quit"
        ]
        
        if self.selected_node:
            info_lines.insert(4, f"Selected: Node {self.selected_node}")
        
        # Draw background rectangle for better visibility
        max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in info_lines if line])
        rect_height = len(info_lines) * 20 + 20
        cv2.rectangle(self.display_image, (5, 5), (max_width + 20, rect_height), (0, 0, 0), -1)
        cv2.rectangle(self.display_image, (5, 5), (max_width + 20, rect_height), (255, 255, 255), 2)
        
        y_offset = 25
        for i, line in enumerate(info_lines):
            if line:  # Skip empty lines for text rendering
                cv2.putText(self.display_image, line, (15, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_progress(self, filename: str = "board_progress.json"):
        """Save current progress to a JSON file"""
        data = {
            'nodes': {str(k): asdict(v) for k, v in self.nodes.items()},
            'edges': [asdict(edge) for edge in self.edges],
            'next_node_id': self.next_node_id,
            'phase': self.phase.value
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Progress saved to {filename}")
    
    def load_progress(self, filename: str = "board_progress.json"):
        """Load progress from a JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.nodes = {int(k): Node(**v) for k, v in data['nodes'].items()}
            self.edges = [Edge(**edge) for edge in data['edges']]
            self.next_node_id = data.get('next_node_id', 1)
            self.selected_node = None
            print(f"Progress loaded from {filename}")
        except FileNotFoundError:
            print(f"No progress file found: {filename}")
        except Exception as e:
            print(f"Error loading progress: {e}")
    
    def complete_phase(self):
        """Complete current phase and save data"""
        if self.phase == Phase.NODES:
            # Save nodes
            os.makedirs("board_data", exist_ok=True)
            nodes_data = {
                str(node.id): {"x": node.x, "y": node.y} 
                for node in self.nodes.values()
            }
            
            with open("board_data/nodes.json", 'w') as f:
                json.dump(nodes_data, f, indent=2)
            
            print(f"✓ NODES PHASE COMPLETED!")
            print(f"✓ Saved {len(self.nodes)} nodes to board_data/nodes.json")
            print()
            print("Next steps:")
            print("  python createBoard.py --phase taxi")
            print("  python createBoard.py --phase bus") 
            print("  python createBoard.py --phase underground")
            print("  python createBoard.py --phase ferry")
            
        else:
            # Save edges for specific transport type
            os.makedirs("board_data", exist_ok=True)
            phase_edges = [
                [edge.node1, edge.node2] for edge in self.edges 
                if edge.transport_type == self.phase.value
            ]
            
            filename = f"board_data/{self.phase.value}_edges.json"
            with open(filename, 'w') as f:
                json.dump(phase_edges, f, indent=2)
            
            print(f"✓ {self.phase.value.upper()} PHASE COMPLETED!")
            print(f"✓ Saved {len(phase_edges)} {self.phase.value} edges to {filename}")
        
        # Also save current progress
        self.save_progress()
        return True
    
    def export_to_json(self):
        """Export all collected data to separate JSON files"""
        # Create output directory
        os.makedirs("board_data", exist_ok=True)
        
        # Export nodes
        nodes_data = {
            str(node.id): {"x": node.x, "y": node.y} 
            for node in self.nodes.values()
        }
        
        with open("board_data/nodes.json", 'w') as f:
            json.dump(nodes_data, f, indent=2)
        
        # Group edges by transport type
        edges_by_type = {}
        for edge in self.edges:
            transport_type = edge.transport_type
            if transport_type not in edges_by_type:
                edges_by_type[transport_type] = []
            edges_by_type[transport_type].append([edge.node1, edge.node2])
        
        # Export each transport type to separate files
        for transport_type, edge_list in edges_by_type.items():
            filename = f"board_data/{transport_type}_edges.json"
            with open(filename, 'w') as f:
                json.dump(edge_list, f, indent=2)
        
        print("Exported all data to board_data/ directory:")
        print(f"- nodes.json ({len(self.nodes)} nodes)")
        for transport_type, edge_list in edges_by_type.items():
            print(f"- {transport_type}_edges.json ({len(edge_list)} edges)")
    
    def run(self):
        """Main application loop"""
        print(f"Shadow Chase Board Creator - {self.phase.value.upper()} Phase")
        print("=" * 50)
        
        if self.phase == Phase.NODES:
            print("NODES PHASE: Click on the image to place nodes at station locations")
            print("Controls:")
            print("  Left click: Place node")
            print("  Right click: Cancel")
            print("  'z': Toggle zoom mode")
            print("  's': Save progress")
            print("  'l': Load progress")
            print("  'c': Complete nodes phase")
            print("  'q': Quit")
            print("  'Backspace': Undo last action")
        else:
            print(f"{self.phase.value.upper()} EDGES PHASE: Click pairs of nodes to connect with {self.phase.value} routes")
            print("Controls:")
            print("  Left click: Select node for edge")
            print("  Right click: Cancel selection")
            print("  'z': Toggle zoom mode")
            print("  's': Save progress")
            print("  'l': Load progress")
            print(f"  'c': Complete {self.phase.value} edges phase")
            print("  'q': Quit")
            print("  'Backspace': Undo last action")
        
        print()
        
        # Try to load existing progress
        self.load_progress()
        
        while True:
            self.draw_elements()
            # self.draw_ui_info()
            
            cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Complete current phase
                if self.complete_phase():
                    break
            elif key == ord('z'):
                self.zoom_mode = not self.zoom_mode
                self.zoom_factor = 2.0 if self.zoom_mode else 1.0
                print(f"Zoom mode: {'ON' if self.zoom_mode else 'OFF'}")
            elif key == ord('s'):
                self.save_progress()
            elif key == ord('l'):
                self.load_progress()
            elif key == 8:  # Backspace
                self.undo_last_action()
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Shadow Chase Board Creator')
    parser.add_argument('--phase', 
                        choices=['nodes', 'taxi', 'bus', 'underground', 'ferry'],
                        required=True,
                        help='Phase to run: nodes, taxi, bus, underground, or ferry')
    parser.add_argument('--image', 
                        default='./data/board.jpg',
                        help='Path to the board image (default: board.jpg)')

    args = parser.parse_args()
    
    image_path = args.image
    phase = Phase(args.phase)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Available image files:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"  - {f}")
        return
    
    try:
        creator = ShadowChaseBoardCreator(image_path, phase)
        creator.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()