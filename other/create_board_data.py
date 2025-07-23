#!/usr/bin/env python3
"""
One-time script to convert board_progress.json to optimized CSV files.
Run this once to create the data files, then use the optimized board_loader.py.
"""

import json
import csv
import os

def create_optimized_board_data(progress_file: str = "board_progress.json", 
                              output_dir: str = "data"):
    """
    Create optimized CSV files from board_progress.json for fast loading
    
    Creates:
    - nodes.csv: Node positions and metadata
    - edges.csv: Edge connections with transport types
    - board_metadata.json: Game configuration data
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {progress_file}...")
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    # Create nodes DataFrame
    nodes_data = []
    for node_id, node_data in data['nodes'].items():
        nodes_data.append({
            'node_id': int(node_id),
            'x': float(node_data['x']),
            'y': float(node_data['y'])
        })
    
    nodes_file = os.path.join(output_dir, 'nodes.csv')
    with open(nodes_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['node_id', 'x', 'y'])
        writer.writeheader()
        writer.writerows(nodes_data)
    print(f"Created {nodes_file} with {len(nodes_data)} nodes")
    
    # Create edges DataFrame
    edges_data = []
    transport_mapping = {
        'taxi': 1,
        'bus': 2,
        'underground': 3,
        'ferry': 4
    }
    
    for edge_data in data.get('edges', []):
        source = edge_data['node1']
        target = edge_data['node2']
        transport_type = edge_data['transport_type']
        edge_type = transport_mapping.get(transport_type, 1)
        
        edges_data.append({
            'source': source,
            'target': target,
            'transport_type': transport_type,
            'edge_type': edge_type
        })
    
    edges_file = os.path.join(output_dir, 'edges.csv')
    with open(edges_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'target', 'transport_type', 'edge_type'])
        writer.writeheader()
        writer.writerows(edges_data)
    print(f"Created {edges_file} with {len(edges_data)} edges")
    
    # Calculate bounds from nodes data
    x_coords = [node['x'] for node in nodes_data]
    y_coords = [node['y'] for node in nodes_data]
    
    # Create metadata file with game configuration
    metadata = {
        'total_nodes': len(nodes_data),
        'total_edges': len(edges_data),
        'transport_types': list(transport_mapping.keys()),
        'transport_mapping': transport_mapping,
        'board_bounds': {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        },
        'source_file': progress_file,
        'created_by': 'create_board_data.py'
    }
    
    metadata_file = os.path.join(output_dir, 'board_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Created {metadata_file} with board configuration")
    
    # Create transport type colors for visualization (optional)
    transport_colors = {
        'taxi': '#FFFF00',        # Yellow
        'bus': '#FF4500',         # Orange Red  
        'underground': '#FF0000', # Red
        'ferry': '#8A2BE2'        # Blue Violet
    }
    
    colors_file = os.path.join(output_dir, 'transport_colors.json')
    with open(colors_file, 'w') as f:
        json.dump(transport_colors, f, indent=2)
    print(f"Created {colors_file} with visualization colors")
    
    print(f"\nBoard data export complete!")
    print(f"Files created in {output_dir}:")
    print(f"  - nodes.csv ({len(nodes_data)} nodes)")
    print(f"  - edges.csv ({len(edges_data)} edges)")
    print(f"  - board_metadata.json (configuration)")
    print(f"  - transport_colors.json (visualization)")
    
    return len(nodes_data), len(edges_data)

if __name__ == "__main__":
    try:
        nodes_count, edges_count = create_optimized_board_data()
        print(f"\nSuccess! Created optimized board data files.")
    except FileNotFoundError:
        print("Error: board_progress.json not found in current directory")
    except Exception as e:
        print(f"Error creating board data: {e}")
