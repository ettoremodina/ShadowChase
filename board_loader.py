#!/usr/bin/env python3
"""
Board graph loader for Scotland Yard game.
Loads graph data from the extracted board_progress.json file.
"""

import json
import networkx as nx
import pandas as pd
from typing import Dict, Tuple
from cops_and_robbers.core.game import ScotlandYardGame, TransportType

def load_board_graph_from_json(progress_file: str = "board_progress.json") -> Tuple[nx.MultiGraph, Dict[int, Tuple[float, float]]]:
    """
    Load Scotland Yard graph and node positions from board_progress.json
    
    Returns:
        - MultiGraph: NetworkX multigraph with transport type edges
        - Dict: Node positions as {node_id: (x, y)}
    """
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    # Create multigraph to support multiple transport types between nodes
    G = nx.MultiGraph()
    
    # Load nodes with positions
    node_positions = {}
    for node_id, node_data in data['nodes'].items():
        node_id_int = int(node_id)
        G.add_node(node_id_int)
        node_positions[node_id_int] = (float(node_data['x']), float(node_data['y']))
    
    # Load edges with transport types
    for edge_data in data.get('edges', []):
        source = edge_data['node1']
        target = edge_data['node2']
        transport_type = edge_data['transport_type']
        
        # Convert transport type name to enum value
        transport_mapping = {
            'taxi': 1,
            'bus': 2, 
            'underground': 3,
            'ferry': 4
        }
        
        edge_type = transport_mapping.get(transport_type, 1)
        G.add_edge(source, target, edge_type=edge_type, transport_type=transport_type)
    
    return G, node_positions

def create_extracted_board_game(num_detectives: int = 3, progress_file: str = "board_progress.json") -> ScotlandYardGame:
    """
    Create Scotland Yard game using the extracted board data
    
    Args:
        num_detectives: Number of detective players
        progress_file: Path to board_progress.json file
        
    Returns:
        ScotlandYardGame: Game instance with extracted board
    """
    graph, node_positions = load_board_graph_from_json(progress_file)
    game = ScotlandYardGame(graph, num_detectives)
    
    # Store node positions for visualization
    game.node_positions = node_positions
    
    return game

def export_board_to_csv(progress_file: str = "board_progress.json", 
                       output_file: str = "data/extracted_board_edgelist.csv") -> None:
    """
    Export board data to CSV format compatible with existing game loaders
    
    Args:
        progress_file: Input board_progress.json file
        output_file: Output CSV file path
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    # Prepare CSV data
    csv_data = []
    for edge_data in data.get('edges', []):
        source = edge_data['node1']
        target = edge_data['node2']
        transport_type = edge_data['transport_type']
        
        # Convert transport type to numeric code
        transport_mapping = {
            'taxi': 1,
            'bus': 2,
            'underground': 3, 
            'ferry': 4
        }
        
        edge_type = transport_mapping.get(transport_type, 1)
        csv_data.append({
            'source': source,
            'target': target, 
            'edge_type': edge_type
        })
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    print(f"Exported {len(csv_data)} edges to {output_file}")


def create_board_visualization_data(progress_file: str = "board_progress.json"):
    """
    Create visualization data for use with matplotlib or other visualization tools
    
    Returns:
        Dict containing nodes, edges, and visualization settings
    """
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    # Extract nodes and edges for visualization
    nodes = {}
    for node_id, node_data in data['nodes'].items():
        nodes[int(node_id)] = (float(node_data['x']), float(node_data['y']))
    
    edges = []
    for edge_data in data.get('edges', []):
        edges.append((edge_data['node1'], edge_data['node2'], edge_data['transport_type']))
    
    # Transport colors and settings for visualization
    transport_colors = {
        'taxi': '#FFFF00',        # Yellow
        'bus': '#FF4500',         # Orange Red
        'underground': '#FF0000', # Red
        'ferry': '#8A2BE2'        # Blue Violet
    }
    
    transport_widths = {
        'taxi': 1.5,
        'bus': 2.5, 
        'underground': 3.5,
        'ferry': 2.0
    }
    
    return {
        'nodes': nodes,
        'edges': edges,
        'colors': transport_colors,
        'widths': transport_widths,
        'metadata': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'source_file': progress_file
        }
    }


if __name__ == "__main__":
    # Test the board loading
    try:
        graph, positions = load_board_graph_from_json()
        print(f"Successfully loaded board with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        print(f"Node positions available: {len(positions)}")
        
        # Export to CSV for compatibility
        export_board_to_csv()
        
        # Test game creation
        game = create_extracted_board_game(3)
        print(f"Created Scotland Yard game with {game.num_cops} detectives")
        
        # Test visualization data
        viz_data = create_board_visualization_data()
        print(f"Created visualization data: {viz_data['metadata']}")
        
    except FileNotFoundError:
        print("board_progress.json not found. Make sure you've extracted the board data first.")
    except Exception as e:
        print(f"Error: {e}")
