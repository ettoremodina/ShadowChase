"""Example game configurations and demonstrations"""

import networkx as nx
import pandas as pd
from ..core.game import Game, StandardMovement, DistanceKMovement, CaptureWinCondition, DistanceKWinCondition
from ..solver.minimax_solver import MinimaxSolver
from ..ui.game_visualizer import GameVisualizer

def create_path_graph_game(n: int, num_cops: int = 1) -> Game:
    """Create game on path graph"""
    graph = nx.path_graph(n)
    return Game(graph, num_cops)

def create_cycle_graph_game(n: int, num_cops: int = 1) -> Game:
    """Create game on cycle graph"""
    graph = nx.cycle_graph(n)
    return Game(graph, num_cops)

def create_complete_graph_game(n: int, num_cops: int = 1) -> Game:
    """Create game on complete graph"""
    graph = nx.complete_graph(n)
    return Game(graph, num_cops)

def create_grid_graph_game(m: int, n: int, num_cops: int = 1) -> Game:
    """Create game on grid graph"""
    graph = nx.grid_2d_graph(m, n)
    # Convert to simple integer labels
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    return Game(graph, num_cops)

def create_petersen_graph_game(num_cops: int = 1) -> Game:
    """Create game on Petersen graph"""
    graph = nx.petersen_graph()
    return Game(graph, num_cops)

def create_distance_k_game(graph: nx.Graph, k: int, num_cops: int = 1) -> Game:
    """Create game with distance-k movement"""
    cop_movement = DistanceKMovement(k)
    robber_movement = DistanceKMovement(k)
    return Game(graph, num_cops, cop_movement, robber_movement)

def create_distance_k_win_game(graph: nx.Graph, k: int, num_cops: int = 1) -> Game:
    """Create game with distance-k win condition"""
    win_condition = DistanceKWinCondition(k, graph)
    return Game(graph, num_cops, win_condition=win_condition)

def create_scotlandYard_game(num_cops: int = 1) -> Game:
    """Create game on Scotland Yard graph"""
    # Scotland Yard graph is a specific structure, here we use a predefined graph
    def create_graph_from_csv(path):
        df = pd.read_csv(path, usecols=[0, 1])
        G = nx.Graph()  # Changed from DiGraph to Graph for undirected edges
        G.add_edges_from(df.itertuples(index=False, name=None))
        return G

    graph = create_graph_from_csv("data\edgelist.csv") 
    return Game(graph, num_cops)


def demo_path_game():
    """Demonstrate game on path graph"""
    print("Path Graph Game Demo")
    game = create_path_graph_game(5, 1)
    
    # Solve the game
    solver = MinimaxSolver(game)
    result = solver.solve([0], 4)
    
    print(f"Cops can win: {result.cops_can_win}")
    if result.game_length:
        print(f"Game length: {result.game_length}")
    
    # Visualize
    visualizer = GameVisualizer(game)
    visualizer.run()

def demo_cycle_game():
    """Demonstrate game on cycle graph"""
    print("Cycle Graph Game Demo")
    game = create_cycle_graph_game(6, 1)
    
    visualizer = GameVisualizer(game)
    visualizer.run()

def demo_grid_game():
    """Demonstrate game on grid graph"""
    print("Grid Graph Game Demo")
    game = create_grid_graph_game(3, 3, 2)
    
    visualizer = GameVisualizer(game)
    visualizer.run()

if __name__ == "__main__":
    # Run different demos
    print("Choose a demo:")
    print("1. Path Graph")
    print("2. Cycle Graph") 
    print("3. Grid Graph")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        demo_path_game()
    elif choice == "2":
        demo_cycle_game()
    elif choice == "3":
        demo_grid_game()
    else:
        print("Invalid choice")

