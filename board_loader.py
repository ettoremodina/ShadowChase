#!/usr/bin/env python3
"""
Optimized board graph loader for Scotland Yard game.
Loads graph data from pre-generated CSV files for fast loading.
"""

import json
import csv
import networkx as nx
from typing import Dict, Tuple
from ScotlandYard.core.game import ScotlandYardGame, TransportType

def load_board_graph_from_csv(nodes_file: str = "data/nodes.csv", 
                             edges_file: str = "data/edges.csv") -> Tuple[nx.MultiGraph, Dict[int, Tuple[float, float]]]:
    """
    Load Scotland Yard graph and node positions from CSV files (FAST LOADING)
    
    Args:
        nodes_file: Path to nodes.csv file
        edges_file: Path to edges.csv file
    
    Returns:
        - MultiGraph: NetworkX multigraph with transport type edges
        - Dict: Node positions as {node_id: (x, y)}
    """
    # Create multigraph to support multiple transport types between nodes
    G = nx.MultiGraph()
    
    # Load nodes with positions from CSV
    node_positions = {}
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['node_id'])
            x = float(row['x'])
            y = float(row['y'])
            G.add_node(node_id)
            node_positions[node_id] = (x, y)
    
    # Load edges from CSV
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = int(row['source'])
            target = int(row['target'])
            transport_type = row['transport_type']
            edge_type = int(row['edge_type'])
            
            G.add_edge(source, target, edge_type=edge_type, transport_type=transport_type)
    
    return G, node_positions

def create_extracted_board_game(num_detectives: int = 3, 
                               nodes_file: str = "data/nodes.csv",
                               edges_file: str = "data/edges.csv") -> ScotlandYardGame:
    """
    Create Scotland Yard game using pre-generated CSV data (FAST LOADING)
    
    Args:
        num_detectives: Number of detective players
        nodes_file: Path to nodes.csv file  
        edges_file: Path to edges.csv file
        
    Returns:
        ScotlandYardGame: Game instance with loaded board
    """
    graph, node_positions = load_board_graph_from_csv(nodes_file, edges_file)
    game = ScotlandYardGame(graph, num_detectives)
    
    # Store node positions for visualization
    game.node_positions = node_positions
    
    return game

def load_board_metadata(metadata_file: str = "data/board_metadata.json") -> dict:
    """
    Load board metadata and configuration from JSON file
    
    Args:
        metadata_file: Path to board_metadata.json
        
    Returns:
        Dict containing board configuration and statistics
    """
    with open(metadata_file, 'r') as f:
        return json.load(f)

def load_transport_colors(colors_file: str = "data/transport_colors.json") -> dict:
    """
    Load transport type colors for visualization
    
    Args:
        colors_file: Path to transport_colors.json
        
    Returns:
        Dict mapping transport types to colors
    """
    with open(colors_file, 'r') as f:
        return json.load(f)

def create_board_visualization_data(nodes_file: str = "data/nodes.csv", 
                                  edges_file: str = "data/edges.csv",
                                  colors_file: str = "data/transport_colors.json"):
    """
    Create visualization data from CSV files for matplotlib or other visualization tools
    
    Returns:
        Dict containing nodes, edges, and visualization settings
    """
    # Load nodes
    nodes = {}
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['node_id'])
            nodes[node_id] = (float(row['x']), float(row['y']))
    
    # Load edges
    edges = []
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = int(row['source'])
            target = int(row['target'])
            transport_type = row['transport_type']
            edges.append((source, target, transport_type))
    
    # Load colors
    transport_colors = load_transport_colors(colors_file)
    
    # Transport widths for visualization
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
            'source_files': [nodes_file, edges_file, colors_file]
        }
    }


# LEGACY FUNCTIONS (for backwards compatibility - SLOWER)
# These functions are kept for backwards compatibility but are DEPRECATED
# Use the CSV-based functions above for better performance

def load_board_graph_from_json(progress_file: str = "board_progress.json") -> Tuple[nx.MultiGraph, Dict[int, Tuple[float, float]]]:
    """
    DEPRECATED: Load Scotland Yard graph from JSON (SLOW - use load_board_graph_from_csv instead)
    
    This function is kept for backwards compatibility but is slower than CSV loading.
    Use load_board_graph_from_csv() for better performance.
    """
    print("WARNING: Using deprecated JSON loading. Use load_board_graph_from_csv() for better performance.")
    
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

def export_board_to_csv(progress_file: str = "board_progress.json", 
                       output_file: str = "data/extracted_board_edgelist.csv") -> None:
    """
    DEPRECATED: Export board data to CSV format (use create_board_data.py instead)
    
    This function is kept for backwards compatibility.
    Use create_board_data.py script for comprehensive CSV export.
    """
    print("WARNING: Using deprecated CSV export. Use create_board_data.py script instead.")
    
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
    import pandas as pd
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    print(f"Exported {len(csv_data)} edges to {output_file}")


if __name__ == "__main__":
    # Test the optimized CSV-based board loading
    try:
        print("Testing optimized CSV-based board loading...")
        
        # Load from CSV files (FAST)
        graph, positions = load_board_graph_from_csv()
        print(f"✓ Successfully loaded board with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        print(f"✓ Node positions available: {len(positions)}")
        
        # Test game creation
        game = create_extracted_board_game(3)
        print(f"✓ Created Scotland Yard game with {game.num_detectives} detectives")
        
        # Load metadata
        metadata = load_board_metadata()
        print(f"✓ Board metadata: {metadata['total_nodes']} nodes, {metadata['total_edges']} edges")
        
        # Test visualization data
        viz_data = create_board_visualization_data()
        print(f"✓ Created visualization data: {viz_data['metadata']}")
        
        print("\n✅ All tests passed! Board loading is now optimized.")
        print("🚀 Use create_extracted_board_game() for fast game creation.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data files not found: {e}")
        print("💡 Run 'python create_board_data.py' first to generate the CSV files.")
    except Exception as e:
        print(f"❌ Error: {e}")
