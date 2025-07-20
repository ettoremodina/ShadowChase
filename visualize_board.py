#!/usr/bin/env python3
"""
Simple Scotland Yard Board Visualizer
Reads board_progress.json and shows the graph with proper node positioning
"""

import json
import matplotlib.pyplot as plt

def load_board_data(progress_file="board_progress.json"):
    """Load board data from JSON file"""
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    # Extract nodes and edges
    nodes = {}
    for node_id, node_data in data['nodes'].items():
        nodes[int(node_id)] = (float(node_data['x']), float(node_data['y']))
    
    edges = []
    for edge_data in data.get('edges', []):
        edges.append((edge_data['node1'], edge_data['node2'], edge_data['transport_type']))
    
    return nodes, edges

def visualize_board(save_path=None):
    """Create and display the Scotland Yard board visualization"""
    # Load data
    nodes, edges = load_board_data()
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f8f8f8')
    
    # Colors and widths for transport types
    transport_style = {
        'taxi': {'color': 'yellow', 'width': 2, 'alpha': 0.7},
        'bus': {'color': 'orange', 'width': 3, 'alpha': 0.8}, 
        'underground': {'color': 'red', 'width': 4, 'alpha': 0.9},
        'ferry': {'color': 'purple', 'width': 2.5, 'alpha': 0.7}
    }
    
    # Draw edges by transport type
    legend_elements = []
    transport_counts = {}
    
    for node1, node2, transport_type in edges:
        if node1 in nodes and node2 in nodes:
            x1, y1 = nodes[node1]
            x2, y2 = nodes[node2]
            
            style = transport_style.get(transport_type, {'color': 'gray', 'width': 2, 'alpha': 0.7})
            ax.plot([x1, x2], [y1, y2], 
                   color=style['color'], 
                   linewidth=style['width'], 
                   alpha=style['alpha'])
            
            # Count edges by type
            transport_counts[transport_type] = transport_counts.get(transport_type, 0) + 1
    
    # Draw nodes
    for node_id, (x, y) in nodes.items():
        # Node circle
        ax.scatter(x, y, c='lightgreen', s=120, edgecolors='black', 
                  linewidth=1.5, zorder=3)
        # Node ID
        ax.text(x, y, str(node_id), ha='center', va='center', 
               fontsize=7, fontweight='bold', color='black', zorder=4)
    
    # Create legend
    for transport_type, count in transport_counts.items():
        style = transport_style.get(transport_type, {'color': 'gray', 'width': 2})
        legend_elements.append(
            plt.Line2D([0], [0], color=style['color'], 
                      linewidth=style['width'],
                      label=f'{transport_type.capitalize()}: {count}')
        )
    
    # Set up plot
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match image coordinates
    ax.set_title('Scotland Yard Board - Extracted Graph', 
                fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add statistics
    stats_text = f"Nodes: {len(nodes)} | Edges: {len(edges)}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Simple command line interface
    import sys
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_board(save_path)
