"""Example game configurations and demonstrations"""

import networkx as nx
import pandas as pd
import sys
import os

# Add the ScotlandYard package directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)  # This is the ScotlandYard directory
project_root = os.path.dirname(package_dir)  # This is the ScotlandYardRL directory

# Add project root to sys.path so we can import ScotlandYard package
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from the ScotlandYard package
from ScotlandYard.core.game import Game, StandardMovement, DistanceKMovement, CaptureWinCondition, DistanceKWinCondition, ScotlandYardGame

def create_path_graph_game(n: int, num_detectives: int = 1) -> Game:
    """Create game on path graph"""
    graph = nx.path_graph(n)
    return Game(graph, num_detectives)

def create_cycle_graph_game(n: int, num_detectives: int = 1) -> Game:
    """Create game on cycle graph"""
    graph = nx.cycle_graph(n)
    return Game(graph, num_detectives)

def create_complete_graph_game(n: int, num_detectives: int = 1) -> Game:
    """Create game on complete graph"""
    graph = nx.complete_graph(n)
    return Game(graph, num_detectives)

def create_grid_graph_game(m: int, n: int, num_detectives: int = 1) -> Game:
    """Create game on grid graph"""
    graph = nx.grid_2d_graph(m, n)
    # Convert to simple integer labels
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    return Game(graph, num_detectives)

def create_petersen_graph_game(num_detectives: int = 1) -> Game:
    """Create game on Petersen graph"""
    graph = nx.petersen_graph()
    return Game(graph, num_detectives)

def create_distance_k_game(graph: nx.Graph, k: int, num_detectives: int = 1) -> Game:
    """Create game with distance-k movement"""
    detective_movement = DistanceKMovement(k)
    MrX_movement = DistanceKMovement(k)
    return Game(graph, num_detectives, detective_movement, MrX_movement)

def create_distance_k_win_game(graph: nx.Graph, k: int, num_detectives: int = 1) -> Game:
    """Create game with distance-k win condition"""
    win_condition = DistanceKWinCondition(k, graph)
    return Game(graph, num_detectives, win_condition=win_condition)

def create_scotlandYard_game(num_detectives: int = 3) -> ScotlandYardGame:
    """Create game on Scotland Yard graph with full rules"""
    def create_graph_from_csv(path):
        df = pd.read_csv(path)
        G = nx.MultiGraph()  # Use MultiGraph to support multiple edges
        
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = int(row['edge_type'])
            G.add_edge(source, target, edge_type=edge_type)
        
        return G

    graph = create_graph_from_csv("data/edgelist.csv") 
    return ScotlandYardGame(graph, num_detectives)

def create_simple_scotland_yard_game(num_detectives: int = 3, 
                                   show_MrX: bool = True,
                                   use_tickets: bool = False) -> Game:
    """Create simplified Scotland Yard game for learning"""
    def create_graph_from_csv(path):
        df = pd.read_csv(path)
        G = nx.MultiGraph()  # Use MultiGraph to support multiple edges
        
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = int(row['edge_type'])
            G.add_edge(source, target, edge_type=edge_type)
        
        return G

    graph = create_graph_from_csv("data/edgelist.csv")
    
    if use_tickets:
        # Use Scotland Yard rules with tickets
        return ScotlandYardGame(graph, num_detectives)
    else:
        # Use basic rules without tickets
        game = Game(graph, num_detectives)
        # Add a flag to indicate this is a Scotland Yard map but with basic rules
        game.is_scotland_yard_map = True
        return game

def create_test_scotland_yard_game(num_detectives: int = 2) -> ScotlandYardGame:
    """Create game on small test Scotland Yard graph with full rules"""
    def create_graph_from_csv(path):
        df = pd.read_csv(path)
        G = nx.MultiGraph()  # Use MultiGraph to support multiple edges
        
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = int(row['edge_type'])
            G.add_edge(source, target, edge_type=edge_type)
        
        return G

    graph = create_graph_from_csv("data/test_edgelist.csv") 
    return ScotlandYardGame(graph, num_detectives)

def create_simple_test_scotland_yard_game(num_detectives: int = 2, 
                                        show_MrX: bool = True,
                                        use_tickets: bool = False) -> Game:
    """Create simplified test Scotland Yard game for learning"""
    def create_graph_from_csv(path):
        df = pd.read_csv(path)
        G = nx.MultiGraph()  # Use MultiGraph to support multiple edges
        
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = int(row['edge_type'])
            G.add_edge(source, target, edge_type=edge_type)
        
        return G

    graph = create_graph_from_csv("data/test_edgelist.csv")
    
    if use_tickets:
        # Use Scotland Yard rules with tickets
        return ScotlandYardGame(graph, num_detectives)
    else:
        # Use basic rules without tickets
        game = Game(graph, num_detectives)
        # Add a flag to indicate this is a Scotland Yard map but with basic rules
        game.is_scotland_yard_map = True
        return game

def create_extracted_board_game(num_detectives: int = 3) -> ScotlandYardGame:
    """Create Scotland Yard game using extracted board data from board_progress.json"""
    try:
        from ScotlandYard.services.board_loader import create_extracted_board_game as _create_game
        return _create_game(num_detectives)
    except ImportError:
        print("Warning: board_loader not available, falling back to CSV data")
        return create_scotlandYard_game(num_detectives)
    except FileNotFoundError:
        print("Warning: board_progress.json not found, falling back to CSV data")
        return create_scotlandYard_game(num_detectives)