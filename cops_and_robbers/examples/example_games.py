"""Example game configurations and demonstrations"""

import networkx as nx
import pandas as pd
from ..core.game import Game, StandardMovement, DistanceKMovement, CaptureWinCondition, DistanceKWinCondition, ScotlandYardGame
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

def create_scotlandYard_game(num_detectives: int = 3) -> ScotlandYardGame:
    """Create game on Scotland Yard graph with full rules"""
    def create_graph_from_csv(path):
        df = pd.read_csv(path)
        G = nx.Graph()
        
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = int(row['edge_type'])
            G.add_edge(source, target, edge_type=edge_type)
        
        return G

    graph = create_graph_from_csv("data/edgelist.csv") 
    return ScotlandYardGame(graph, num_detectives)

def create_simple_scotland_yard_game(num_cops: int = 3, 
                                   show_robber: bool = True,
                                   use_tickets: bool = False) -> Game:
    """Create simplified Scotland Yard game for learning"""
    def create_graph_from_csv(path):
        df = pd.read_csv(path)
        G = nx.Graph()
        
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_type = int(row['edge_type'])
            G.add_edge(source, target, edge_type=edge_type)
        
        return G

    graph = create_graph_from_csv("data/edgelist.csv")
    
    if use_tickets:
        # Use Scotland Yard rules
        return ScotlandYardGame(graph, num_cops)
    else:
        # Use basic rules
        game = Game(graph, num_cops)
        game.simplified_mode = True
        game.show_robber = show_robber
        return game

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

def demo_scotland_yard_game():
    """Demonstrate full Scotland Yard game"""
    print("Scotland Yard Game Demo")
    game = create_scotlandYard_game(3)
    
    # Initialize with random positions
    import random
    nodes = list(game.graph.nodes())
    detective_positions = random.sample(nodes, 3)
    mr_x_position = random.choice([n for n in nodes if n not in detective_positions])
    
    game.initialize_scotland_yard_game(detective_positions, mr_x_position)
    
    print(f"Detectives at: {detective_positions}")
    print(f"Mr. X at: {mr_x_position} (hidden)")
    
    # Show ticket counts
    for i in range(3):
        tickets = game.get_detective_tickets(i)
        print(f"Detective {i+1}: {tickets}")
    
    mr_x_tickets = game.get_mr_x_tickets()
    print(f"Mr. X: {mr_x_tickets}")

def demo_simple_scotland_yard():
    """Demonstrate simplified Scotland Yard game"""
    print("Simple Scotland Yard Game Demo")
    game = create_simple_scotland_yard_game(num_cops=2, show_robber=True, use_tickets=False)
    
    # Use basic initialization
    game.initialize_game([1, 3], 100)
    
    visualizer = GameVisualizer(game)
    visualizer.run()

if __name__ == "__main__":
    # Run different demos
    print("Choose a demo:")
    print("1. Path Graph")
    print("2. Cycle Graph") 
    print("3. Grid Graph")
    print("4. Simple Scotland Yard")
    print("5. Full Scotland Yard")
    
    choice = input("Enter choice (1-5): ")
    
    if choice == "1":
        demo_path_game()
    elif choice == "2":
        demo_cycle_game()
    elif choice == "3":
        demo_grid_game()
    elif choice == "4":
        demo_simple_scotland_yard()
    elif choice == "5":
        demo_scotland_yard_game()
    else:
        print("Invalid choice")

