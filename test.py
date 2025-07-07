import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import sys
import os
import random

# Add the project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cops_and_robbers.ui.game_visualizer import GameVisualizer
from cops_and_robbers.core.game import Game, Player, ScotlandYardGame, TicketType, TransportType
from cops_and_robbers.examples.example_games import *
from cops_and_robbers.solver.minimax_solver import MinimaxSolver

def print_game_state(game):
    """Print current game state"""
    state = game.get_state_representation()
    print(f"\n=== Turn {state['turn_count']} - {state['turn'].upper()}'S TURN ===")
    print(f"Cops at: {state['cop_positions']}")
    print(f"Robber at: {state['robber_position']}")
    print(f"Game over: {state['game_over']}")
    if state['winner']:
        print(f"Winner: {state['winner'].upper()}")

def show_valid_moves(game, player, position=None):
    """Show valid moves for a player"""
    if player == Player.COPS and position is not None:
        moves = game.get_valid_moves(Player.COPS, position)
        print(f"Cop at {position} can move to: {sorted(moves)}")
    elif player == Player.ROBBER:
        moves = game.get_valid_moves(Player.ROBBER)
        print(f"Robber can move to: {sorted(moves)}")
    return moves

def test_basic_game():
    """Test basic game mechanics"""
    print("=== TESTING BASIC GAME MECHANICS ===")
    
    # Create small grid game
    m, n = 3, 3
    graph = nx.grid_2d_graph(m, n)
    # Convert to simple integer labels
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    
    print(f"Graph nodes: {sorted(graph.nodes())}")
    print(f"Graph edges: {list(graph.edges())}")
    
    # Initialize game: 2 cops at positions 0,1 and robber at position 7
    # Make sure positions don't conflict
    game = Game(graph, 2)
    cop_positions = [0, 1]
    robber_position = 7
    
    # Ensure no position conflicts
    if robber_position not in cop_positions:
        game.initialize_game(cop_positions, robber_position)
    else:
        # Use different positions if there's a conflict
        game.initialize_game([0, 2], 7)
    
    print_game_state(game)
    
    # Show valid moves for each cop
    for i, cop_pos in enumerate(game.game_state.cop_positions):
        print(f"\nCop {i+1}:")
        show_valid_moves(game, Player.COPS, cop_pos)
    
    # Make cops move
    print("\n--- COPS MOVE ---")
    cop1_moves = show_valid_moves(game, Player.COPS, 0)
    cop2_moves = show_valid_moves(game, Player.COPS, 1)
    
    # Move cops to new positions
    new_cop_positions = [3, 4]  # Example moves
    success = game.make_move(new_positions=new_cop_positions)
    print(f"Cops move to {new_cop_positions}: {'Success' if success else 'Failed'}")
    
    print_game_state(game)
    
    # Show robber's valid moves
    print("\n--- ROBBER'S TURN ---")
    robber_moves = show_valid_moves(game, Player.ROBBER)
    
    # Move robber
    new_robber_pos = 8  # Example move
    success = game.make_move(new_robber_pos=new_robber_pos)
    print(f"Robber moves to {new_robber_pos}: {'Success' if success else 'Failed'}")
    
    print_game_state(game)

def test_game_until_end():
    """Play a simple game until completion"""
    print("\n\n=== PLAYING COMPLETE GAME ===")
    
    # Simple path graph for quick game
    graph = nx.path_graph(5)  # Nodes 0-4 in a line
    game = Game(graph, 1)
    game.initialize_game([0], 4)  # Cop at 0, robber at 4
    
    turn = 0
    max_turns = 10
    
    while not game.is_game_over() and turn < max_turns:
        print_game_state(game)
        
        if game.game_state.turn == Player.COPS:
            # Simple strategy: cop moves toward robber
            cop_pos = game.game_state.cop_positions[0]
            robber_pos = game.game_state.robber_position
            
            moves = show_valid_moves(game, Player.COPS, cop_pos)
            
            # Choose move that gets closer to robber
            best_move = cop_pos
            for move in moves:
                if abs(move - robber_pos) < abs(best_move - robber_pos):
                    best_move = move
            
            print(f"Cop chooses to move to: {best_move}")
            game.make_move(new_positions=[best_move])
            
        else:  # Robber's turn
            moves = show_valid_moves(game, Player.ROBBER)
            
            # Simple strategy: move away from cops
            robber_pos = game.game_state.robber_position
            cop_pos = game.game_state.cop_positions[0]
            
            best_move = robber_pos
            for move in moves:
                if abs(move - cop_pos) > abs(best_move - cop_pos):
                    best_move = move
            
            print(f"Robber chooses to move to: {best_move}")
            game.make_move(new_robber_pos=best_move)
        
        turn += 1
    
    print_game_state(game)
    if game.is_game_over():
        print(f"\nGame ended after {turn} turns!")
    else:
        print(f"\nGame stopped after {max_turns} turns (no winner yet)")

def test_scotland_yard_game():
    """Test Scotland Yard specific mechanics"""
    print("\n\n=== TESTING SCOTLAND YARD GAME ===")
    
    # Create Scotland Yard game
    game = create_scotlandYard_game(2)
    
    # Initialize with specific positions
    detective_positions = [1, 13]
    mr_x_position = 100
    
    game.initialize_scotland_yard_game(detective_positions, mr_x_position)
    
    print(f"Game initialized:")
    print(f"Detectives at: {detective_positions}")
    print(f"Mr. X at: {mr_x_position} (hidden: {not game.game_state.mr_x_visible})")
    
    # Show initial tickets
    print("\nInitial tickets:")
    for i in range(2):
        tickets = game.get_detective_tickets(i)
        print(f"Detective {i+1}: {tickets}")
    
    mr_x_tickets = game.get_mr_x_tickets()
    print(f"Mr. X: {mr_x_tickets}")

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

def demo_scotland_yard_visualizer():
    """Demonstrate full Scotland Yard game with visualizer"""
    print("Scotland Yard Game with Visualizer")
    game = create_scotlandYard_game(3)
    
    visualizer = GameVisualizer(game)
    visualizer.run()

def demo_test_scotland_yard():
    """Demonstrate test Scotland Yard game with small graph"""
    print("Test Scotland Yard Game Demo (10 nodes)")
    from cops_and_robbers.examples.example_games import create_test_scotland_yard_game
    
    game = create_test_scotland_yard_game(2)
    
    # Initialize with specific positions
    detective_positions = [1, 3]
    mr_x_position = 8
    
    game.initialize_scotland_yard_game(detective_positions, mr_x_position)
    
    print(f"Detectives at: {detective_positions}")
    print(f"Mr. X at: {mr_x_position}")
    
    # Show ticket counts
    for i in range(2):
        tickets = game.get_detective_tickets(i)
        print(f"Detective {i+1}: {tickets}")
    
    mr_x_tickets = game.get_mr_x_tickets()
    print(f"Mr. X: {mr_x_tickets}")
    
    visualizer = GameVisualizer(game)
    visualizer.run()

def demo_simple_test_scotland_yard():
    """Demonstrate simple test Scotland Yard game"""
    print("Simple Test Scotland Yard Game Demo")
    from cops_and_robbers.examples.example_games import create_simple_test_scotland_yard_game
    
    game = create_simple_test_scotland_yard_game(num_cops=2, show_robber=True, use_tickets=False)
    
    # Use basic initialization
    game.initialize_game([1, 3], 8)
    
    visualizer = GameVisualizer(game)
    visualizer.run()

# Run tests
if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Basic Game Test")
    print("2. Complete Game Test") 
    print("3. Scotland Yard Test")
    print("4. Path Graph Demo")
    print("5. Cycle Graph Demo")
    print("6. Grid Graph Demo")
    print("7. Simple Scotland Yard Demo")
    print("8. Full Scotland Yard Demo")
    print("9. Scotland Yard Visualizer")
    print("10. Test Scotland Yard (10 nodes)")
    print("11. Simple Test Scotland Yard (10 nodes)")
    
    choice = input("Enter choice (1-11): ")
    
    if choice == "1":
        test_basic_game()
    elif choice == "2":
        test_game_until_end()
    elif choice == "3":
        test_scotland_yard_game()
    elif choice == "4":
        demo_path_game()
    elif choice == "5":
        demo_cycle_game()
    elif choice == "6":
        demo_grid_game()
    elif choice == "7":
        demo_simple_scotland_yard()
    elif choice == "8":
        demo_scotland_yard_game()
    elif choice == "9":
        demo_scotland_yard_visualizer()
    elif choice == "10":
        demo_test_scotland_yard()
    elif choice == "11":
        demo_simple_test_scotland_yard()
    else:
        print("Invalid choice")
