import networkx as nx
import sys
import os
import random

# Add the project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ScotlandYard.ui.game_visualizer import GameVisualizer
from ScotlandYard.core.game import Game, Player, ScotlandYardGame, TicketType, TransportType
from ScotlandYard.examples.example_games import *

def print_game_state(game):
    """Print current game state"""
    state = game.get_state_representation()
    print(f"\n=== Turn {state['turn_count']} - {state['turn'].upper()}'S TURN ===")
    print(f"detectives at: {state['detective_positions']}")
    print(f"MrX at: {state['MrX_position']}")
    print(f"Game over: {state['game_over']}")
    if state['winner']:
        print(f"Winner: {state['winner'].upper()}")

def show_valid_moves(game, player, position=None):
    """Show valid moves for a player"""
    if player == Player.DETECTIVES and position is not None:
        moves = game.get_valid_moves(Player.DETECTIVES, position)
        if isinstance(game, ScotlandYardGame):
            print(f"detective at {position} can move to: {[(dest, transport.name) for dest, transport in moves]}")
        else:
            print(f"detective at {position} can move to: {sorted(moves)}")
    elif player == Player.MRX:
        moves = game.get_valid_moves(Player.MRX)
        if isinstance(game, ScotlandYardGame):
            print(f"MrX can move to: {[(dest, transport.name) for dest, transport in moves]}")
        else:
            print(f"MrX can move to: {sorted(moves)}")
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
    
    # Initialize game: 2 detectives at positions 0,1 and MrX at position 7
    # Make sure positions don't conflict
    game = Game(graph, 2)
    detective_positions = [0, 1]
    MrX_position = 7
    
    # Ensure no position conflicts
    if MrX_position not in detective_positions:
        game.initialize_game(detective_positions, MrX_position)
    else:
        # Use different positions if there's a conflict
        game.initialize_game([0, 2], 7)
    
    print_game_state(game)
    
    # Show valid moves for each detective
    for i, detective_pos in enumerate(game.game_state.detective_positions):
        print(f"\ndetective {i+1}:")
        show_valid_moves(game, Player.DETECTIVES, detective_pos)
    
    # Make detectives move
    print("\n--- detectives MOVE ---")
    detective1_moves = show_valid_moves(game, Player.DETECTIVES, 0)
    detective2_moves = show_valid_moves(game, Player.DETECTIVES, 1)
    
    # Move detectives to new positions
    new_detective_positions = [3, 4]  # Example moves
    success = game.make_move(new_positions=new_detective_positions)
    print(f"detectives move to {new_detective_positions}: {'Success' if success else 'Failed'}")
    
    print_game_state(game)
    
    # Show MrX's valid moves
    print("\n--- MrX'S TURN ---")
    MrX_moves = show_valid_moves(game, Player.MRX)
    
    # Move MrX
    new_MrX_pos = 8  # Example move
    success = game.make_move(new_MrX_pos=new_MrX_pos)
    print(f"MrX moves to {new_MrX_pos}: {'Success' if success else 'Failed'}")
    
    print_game_state(game)

def test_game_until_end():
    """Play a simple game until completion"""
    print("\n\n=== PLAYING COMPLETE GAME ===")
    
    # Simple path graph for quick game
    graph = nx.path_graph(5)  # Nodes 0-4 in a line
    game = Game(graph, 1)
    game.initialize_game([0], 4)  # detective at 0, MrX at 4
    
    turn = 0
    max_turns = 10
    
    while not game.is_game_over() and turn < max_turns:
        print_game_state(game)
        
        if game.game_state.turn == Player.DETECTIVES:
            # Simple strategy: detective moves toward MrX
            detective_pos = game.game_state.detective_positions[0]
            MrX_pos = game.game_state.MrX_position
            
            moves = show_valid_moves(game, Player.DETECTIVES, detective_pos)
            
            # Choose move that gets closer to MrX
            best_move = detective_pos
            for move in moves:
                if abs(move - MrX_pos) < abs(best_move - MrX_pos):
                    best_move = move
            
            print(f"detective chooses to move to: {best_move}")
            game.make_move(new_positions=[best_move])
            
        else:  # MrX's turn
            moves = show_valid_moves(game, Player.MRX)
            
            # Simple strategy: move away from detectives
            MrX_pos = game.game_state.MrX_position
            detective_pos = game.game_state.detective_positions[0]
            
            best_move = MrX_pos
            for move in moves:
                if abs(move - detective_pos) > abs(best_move - detective_pos):
                    best_move = move
            
            print(f"MrX chooses to move to: {best_move}")
            game.make_move(new_MrX_pos=best_move)
        
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
    solver = None
    result = solver.solve([0], 4)
    
    print(f"detectives can win: {result.detectives_can_win}")
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
    game = create_simple_scotland_yard_game(num_detectives=2, show_MrX=True, use_tickets=False)
    
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
    from ScotlandYard.examples.example_games import create_test_scotland_yard_game
    
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
    from ScotlandYard.examples.example_games import create_simple_test_scotland_yard_game
    
    game = create_simple_test_scotland_yard_game(num_detectives=2, show_MrX=True, use_tickets=False)
    
    # Use basic initialization
    game.initialize_game([1, 3], 8)
    
    visualizer = GameVisualizer(game)
    visualizer.run()

def demo_extracted_board_game(num_detectives: int = 3, auto_init: bool = True):
    """Create Scotland Yard game using the extracted board data"""
    game = create_extracted_board_game(num_detectives)

    # Define predefined positions using the method from game_logic.py
    if auto_init:
        import random
        starting_cards = [13,26,29,34,50,53,91,103,112,132,138,141,155,174,197,94, 117, 198]
        available_nodes = list(game.graph.nodes())
        
        # Filter starting cards to only include nodes that exist in the graph
        valid_starting_cards = [pos for pos in starting_cards if pos in available_nodes]
        
        # Ensure we have enough valid positions
        if len(valid_starting_cards) < num_detectives + 1:
            # Add more random positions if needed
            remaining_nodes = [n for n in available_nodes if n not in valid_starting_cards]
            additional_needed = (num_detectives + 1) - len(valid_starting_cards)
            valid_starting_cards.extend(random.sample(remaining_nodes, min(additional_needed, len(remaining_nodes))))
        
        sample = random.sample(valid_starting_cards, num_detectives + 1)
        detective_positions = sample[1:num_detectives+1]        
        mr_x_position = sample[0]
        
        positions = detective_positions + [mr_x_position]
    else:
        positions = None

    # Initialize visualizer with positions but don't auto-start
    visualizer = GameVisualizer(game, auto_positions=positions)
    
    # # Set game mode to human vs human by default (can be changed in UI)
    # mode_map = {
    #     "human_vs_human": {'detectives': 'Human', 'mr_x': 'Human'},
    #     "human_det_vs_ai_mrx": {'detectives': 'Human', 'mr_x': 'AI'},
    #     "ai_det_vs_human_mrx": {'detectives': 'AI', 'mr_x': 'Human'},
    #     "ai_vs_ai": {'detectives': 'AI', 'mr_x': 'AI'}
    # }
    # visualizer.game_mode = mode_map['ai_vs_ai']
    
    visualizer.run()





if __name__ == "__main__":
    demo_extracted_board_game(5)
# Run tests
# if __name__ == "__main__":
#     print("Choose a demo:")
#     print("1. Basic Game Test")
#     print("2. Complete Game Test") 
#     print("3. Scotland Yard Test")
#     print("4. Path Graph Demo")
#     print("5. Cycle Graph Demo")
#     print("6. Grid Graph Demo")
#     print("7. Simple Scotland Yard Demo")
#     print("8. Full Scotland Yard Demo")
#     print("9. Scotland Yard Visualizer")
#     print("10. Test Scotland Yard (10 nodes)")
#     print("11. Simple Test Scotland Yard (10 nodes)")
    
#     choice = input("Enter choice (1-11): ")
    
#     if choice == "1":
#         test_basic_game()
#     elif choice == "2":
#         test_game_until_end()
#     elif choice == "3":
#         test_scotland_yard_game()
#     elif choice == "4":
#         demo_path_game()
#     elif choice == "5":
#         demo_cycle_game()
#     elif choice == "6":
#         demo_grid_game()
#     elif choice == "7":
#         demo_simple_scotland_yard()
#     elif choice == "8":
#         demo_scotland_yard_game()
#     elif choice == "9":
#         demo_scotland_yard_visualizer()
#     elif choice == "10":
#         demo_test_scotland_yard()
#     elif choice == "11":
#         demo_simple_test_scotland_yard()
#     else:
#         print("Invalid choice")
