import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
from cops_and_robbers import GameVisualizer, Game, Player
from cops_and_robbers.examples.example_games import *
from cops_and_robbers.core.game import ScotlandYardGame, TicketType, TransportType

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
    game = Game(graph, 2)
    game.initialize_game([0, 1], 7)
    
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
    
    # Test Mr. X move
    print("\n--- Mr. X Move ---")
    current_pos = game.game_state.robber_position
    neighbors = list(game.graph.neighbors(current_pos))
    if neighbors:
        target = neighbors[0]
        edge_data = game.graph.get_edge_data(current_pos, target)
        transport = TransportType(edge_data.get('edge_type', 1))
        
        print(f"Mr. X moves from {current_pos} to {target} via {transport.name}")
        success = game.make_scotland_yard_move(Player.MR_X, target, transport)
        print(f"Move success: {success}")
        
        # Check tickets after move
        mr_x_tickets_after = game.get_mr_x_tickets()
        print(f"Mr. X tickets after move: {mr_x_tickets_after}")
    
    # Test detective move
    print("\n--- Detective Move ---")
    detective_id = 0
    current_pos = game.game_state.cop_positions[detective_id]
    neighbors = list(game.graph.neighbors(current_pos))
    valid_neighbors = [n for n in neighbors if n not in game.game_state.cop_positions]
    
    if valid_neighbors:
        target = valid_neighbors[0]
        edge_data = game.graph.get_edge_data(current_pos, target)
        transport = TransportType(edge_data.get('edge_type', 1))
        
        print(f"Detective {detective_id+1} moves from {current_pos} to {target} via {transport.name}")
        success = game.make_scotland_yard_move(Player.DETECTIVES, target, transport, detective_id)
        print(f"Move success: {success}")
        
        # Check tickets after move
        detective_tickets_after = game.get_detective_tickets(detective_id)
        mr_x_tickets_after = game.get_mr_x_tickets()
        print(f"Detective {detective_id+1} tickets after move: {detective_tickets_after}")
        print(f"Mr. X tickets after detective move: {mr_x_tickets_after}")

def test_simple_vs_full_scotland_yard():
    """Compare simple and full Scotland Yard rules"""
    print("\n\n=== COMPARING SIMPLE VS FULL RULES ===")
    
    # Simple game
    print("Simple Scotland Yard (visible robber, no tickets):")
    simple_game = create_simple_scotland_yard_game(2, show_robber=True, use_tickets=False)
    simple_game.initialize_game([1, 13], 100)
    print(f"Robber position visible: {hasattr(simple_game, 'show_robber') and simple_game.show_robber}")
    
    # Full game
    print("\nFull Scotland Yard (hidden robber, tickets):")
    full_game = create_scotlandYard_game(2)
    full_game.initialize_scotland_yard_game([1, 13], 100)
    print(f"Mr. X position hidden: {not full_game.game_state.mr_x_visible}")
    print(f"Uses tickets: {isinstance(full_game, ScotlandYardGame)}")

# Run tests
if __name__ == "__main__":
    test_basic_game()
    test_game_until_end()
    test_scotland_yard_game()
    test_simple_vs_full_scotland_yard()
