"""
Simulation runner for Scotland Yard games
Runs multiple games and collects statistics
"""

import random
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from cops_and_robbers.core.game import Player, ScotlandYardGame
from cops_and_robbers.examples.example_games import create_scotlandYard_game, create_simple_scotland_yard_game
import json

@dataclass
class GameResult:
    """Results from a single game"""
    game_id: int
    winner: Player
    turns: int
    final_detective_positions: List[int]
    final_mr_x_position: int
    mr_x_tickets_remaining: Dict
    detective_tickets_remaining: Dict
    game_length_seconds: float

@dataclass
class SimulationStats:
    """Statistics from multiple games"""
    total_games: int
    mr_x_wins: int
    detective_wins: int
    average_game_length: float
    average_turns: int
    win_percentages: Dict[Player, float]

class RandomPlayer:
    """Simple random player for simulation"""
    
    def choose_move(self, game: ScotlandYardGame, player: Player, 
                   player_id: Optional[int] = None) -> Tuple[int, int]:
        """Choose a random valid move"""
        if player == Player.MR_X:
            current_pos = game.game_state.robber_position
            valid_moves = self._get_mr_x_moves(game, current_pos)
            if valid_moves:
                return random.choice(valid_moves)
        elif player == Player.DETECTIVES and player_id is not None:
            current_pos = game.game_state.cop_positions[player_id]
            valid_moves = self._get_detective_moves(game, player_id, current_pos)
            if valid_moves:
                return random.choice(valid_moves)
        
        return None, None
    
    def _get_mr_x_moves(self, game: ScotlandYardGame, position: int) -> List[Tuple[int, int]]:
        """Get valid moves for Mr. X"""
        moves = []
        for neighbor in game.graph.neighbors(position):
            edge_data = game.graph.get_edge_data(position, neighbor)
            transport_type = edge_data.get('edge_type', 1)
            
            # Check if Mr. X has appropriate tickets
            if self._mr_x_has_ticket(game, transport_type):
                moves.append((neighbor, transport_type))
        
        return moves
    
    def _get_detective_moves(self, game: ScotlandYardGame, detective_id: int, 
                           position: int) -> List[Tuple[int, int]]:
        """Get valid moves for a detective"""
        moves = []
        for neighbor in game.graph.neighbors(position):
            # Check if position is occupied
            if neighbor in game.game_state.cop_positions:
                continue
            
            edge_data = game.graph.get_edge_data(position, neighbor)
            transport_type = edge_data.get('edge_type', 1)
            
            # Check if detective has appropriate tickets
            if self._detective_has_ticket(game, detective_id, transport_type):
                moves.append((neighbor, transport_type))
        
        return moves
    
    def _mr_x_has_ticket(self, game: ScotlandYardGame, transport_type: int) -> bool:
        """Check if Mr. X has required ticket"""
        from cops_and_robbers.core.game import TicketType, TransportType
        
        ticket_mapping = {
            1: TicketType.TAXI,
            2: TicketType.BUS,
            3: TicketType.UNDERGROUND
        }
        
        ticket_type = ticket_mapping.get(transport_type, TicketType.TAXI)
        tickets = game.get_mr_x_tickets()
        
        # Can also use black tickets
        return (tickets.get(ticket_type, 0) > 0 or 
                tickets.get(TicketType.BLACK, 0) > 0)
    
    def _detective_has_ticket(self, game: ScotlandYardGame, detective_id: int, 
                            transport_type: int) -> bool:
        """Check if detective has required ticket"""
        from cops_and_robbers.core.game import TicketType
        
        ticket_mapping = {
            1: TicketType.TAXI,
            2: TicketType.BUS,
            3: TicketType.UNDERGROUND
        }
        
        ticket_type = ticket_mapping.get(transport_type, TicketType.TAXI)
        tickets = game.get_detective_tickets(detective_id)
        
        return tickets.get(ticket_type, 0) > 0

class GameSimulator:
    """Runs multiple game simulations"""
    
    def __init__(self, use_full_rules: bool = True, num_detectives: int = 3):
        self.use_full_rules = use_full_rules
        self.num_detectives = num_detectives
        self.mr_x_player = RandomPlayer()
        self.detective_players = [RandomPlayer() for _ in range(num_detectives)]
    
    def run_single_game(self, game_id: int, max_turns: int = 50, 
                       verbose: bool = False) -> GameResult:
        """Run a single game simulation"""
        start_time = time.time()
        
        # Create game
        if self.use_full_rules:
            game = create_scotlandYard_game(self.num_detectives)
            # Initialize with random positions
            nodes = list(game.graph.nodes())
            detective_positions = random.sample(nodes, self.num_detectives)
            mr_x_position = random.choice([n for n in nodes if n not in detective_positions])
            game.initialize_scotland_yard_game(detective_positions, mr_x_position)
        else:
            game = create_simple_scotland_yard_game(self.num_detectives, 
                                                  show_robber=True, use_tickets=False)
            nodes = list(game.graph.nodes())
            detective_positions = random.sample(nodes, self.num_detectives)
            mr_x_position = random.choice([n for n in nodes if n not in detective_positions])
            game.initialize_game(detective_positions, mr_x_position)
        
        turn_count = 0
        
        if verbose:
            print(f"\nGame {game_id} started:")
            print(f"Detectives at: {detective_positions}")
            print(f"Mr. X at: {mr_x_position}")
        
        # Game loop
        while not game.is_game_over() and turn_count < max_turns:
            if self.use_full_rules:
                success = self._play_scotland_yard_turn(game, verbose)
            else:
                success = self._play_simple_turn(game, verbose)
            
            if not success:
                if verbose:
                    print("Game ended due to invalid move")
                break
            
            turn_count += 1
            
            if verbose and turn_count % 5 == 0:
                print(f"Turn {turn_count}, Game continues...")
        
        end_time = time.time()
        
        # Collect results
        winner = game.get_winner() or Player.MR_X  # Default to Mr. X if no winner
        
        result = GameResult(
            game_id=game_id,
            winner=winner,
            turns=turn_count,
            final_detective_positions=game.game_state.cop_positions.copy(),
            final_mr_x_position=game.game_state.robber_position,
            mr_x_tickets_remaining=game.get_mr_x_tickets() if self.use_full_rules else {},
            detective_tickets_remaining={i: game.get_detective_tickets(i) 
                                       for i in range(self.num_detectives)} if self.use_full_rules else {},
            game_length_seconds=end_time - start_time
        )
        
        if verbose:
            print(f"Game {game_id} ended: {winner.value} wins in {turn_count} turns")
        
        return result
    
    def _play_scotland_yard_turn(self, game: ScotlandYardGame, verbose: bool) -> bool:
        """Play one turn of Scotland Yard"""
        if game.game_state.turn == Player.MR_X:
            # Mr. X's turn
            new_pos, transport = self.mr_x_player.choose_move(game, Player.MR_X)
            if new_pos is None:
                return False
            
            from cops_and_robbers.core.game import TransportType
            success = game.make_scotland_yard_move(Player.MR_X, new_pos, 
                                                 TransportType(transport))
            if verbose:
                print(f"Mr. X moves to {new_pos} via {TransportType(transport).name}")
            return success
        
        else:
            # Detectives' turns
            for detective_id in range(self.num_detectives):
                new_pos, transport = self.detective_players[detective_id].choose_move(
                    game, Player.DETECTIVES, detective_id)
                
                if new_pos is None:
                    continue
                
                from cops_and_robbers.core.game import TransportType
                success = game.make_scotland_yard_move(Player.DETECTIVES, new_pos,
                                                     TransportType(transport), detective_id)
                if verbose:
                    print(f"Detective {detective_id} moves to {new_pos} via {TransportType(transport).name}")
                
                if not success:
                    return False
            
            return True
    
    def _play_simple_turn(self, game, verbose: bool) -> bool:
        """Play one turn of simple game"""
        if game.game_state.turn == Player.COPS:
            # Cops' turn
            new_positions = []
            for i, cop_pos in enumerate(game.game_state.cop_positions):
                valid_moves = game.get_valid_moves(Player.COPS, cop_pos)
                if valid_moves:
                    new_pos = random.choice(list(valid_moves))
                    new_positions.append(new_pos)
                else:
                    new_positions.append(cop_pos)
            
            success = game.make_move(new_positions=new_positions)
            if verbose:
                print(f"Cops move to {new_positions}")
            return success
        
        else:
            # Robber's turn
            valid_moves = game.get_valid_moves(Player.ROBBER)
            if valid_moves:
                new_pos = random.choice(list(valid_moves))
                success = game.make_move(new_robber_pos=new_pos)
                if verbose:
                    print(f"Robber moves to {new_pos}")
                return success
            
            return False
    
    def run_simulation(self, num_games: int, max_turns: int = 50, 
                      verbose: bool = False) -> SimulationStats:
        """Run multiple games and collect statistics"""
        print(f"Running {num_games} games...")
        
        results = []
        for i in range(num_games):
            if i % 100 == 0 and i > 0:
                print(f"Completed {i}/{num_games} games")
            
            result = self.run_single_game(i, max_turns, verbose and i < 3)
            results.append(result)
        
        # Calculate statistics
        mr_x_wins = sum(1 for r in results if r.winner == Player.MR_X)
        detective_wins = len(results) - mr_x_wins
        
        avg_turns = sum(r.turns for r in results) / len(results)
        avg_time = sum(r.game_length_seconds for r in results) / len(results)
        
        win_percentages = {
            Player.MR_X: (mr_x_wins / num_games) * 100,
            Player.DETECTIVES: (detective_wins / num_games) * 100
        }
        
        stats = SimulationStats(
            total_games=num_games,
            mr_x_wins=mr_x_wins,
            detective_wins=detective_wins,
            average_game_length=avg_time,
            average_turns=avg_turns,
            win_percentages=win_percentages
        )
        
        return stats, results

def print_simulation_results(stats: SimulationStats, results: List[GameResult]):
    """Print detailed simulation results"""
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Total games: {stats.total_games}")
    print(f"Mr. X wins: {stats.mr_x_wins} ({stats.win_percentages[Player.MR_X]:.1f}%)")
    print(f"Detective wins: {stats.detective_wins} ({stats.win_percentages[Player.DETECTIVES]:.1f}%)")
    print(f"Average game length: {stats.average_turns:.1f} turns")
    print(f"Average time per game: {stats.average_game_length:.3f} seconds")
    
    # Additional statistics
    mr_x_turn_lengths = [r.turns for r in results if r.winner == Player.MR_X]
    detective_turn_lengths = [r.turns for r in results if r.winner == Player.DETECTIVES]
    
    if mr_x_turn_lengths:
        print(f"Average Mr. X win length: {sum(mr_x_turn_lengths) / len(mr_x_turn_lengths):.1f} turns")
    
    if detective_turn_lengths:
        print(f"Average Detective win length: {sum(detective_turn_lengths) / len(detective_turn_lengths):.1f} turns")

def save_results(stats: SimulationStats, results: List[GameResult], filename: str):
    """Save results to JSON file"""
    data = {
        'stats': {
            'total_games': stats.total_games,
            'mr_x_wins': stats.mr_x_wins,
            'detective_wins': stats.detective_wins,
            'average_game_length': stats.average_game_length,
            'average_turns': stats.average_turns,
            'win_percentages': {k.value: v for k, v in stats.win_percentages.items()}
        },
        'results': [
            {
                'game_id': r.game_id,
                'winner': r.winner.value,
                'turns': r.turns,
                'final_detective_positions': r.final_detective_positions,
                'final_mr_x_position': r.final_mr_x_position,
                'game_length_seconds': r.game_length_seconds
            }
            for r in results
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    print("Scotland Yard Simulation Runner")
    print("1. Simple rules (visible robber, no tickets)")
    print("2. Full Scotland Yard rules")
    
    choice = input("Choose game type (1-2): ")
    use_full_rules = choice == "2"
    
    num_games = int(input("Number of games to simulate: "))
    num_detectives = int(input("Number of detectives (2-5): "))
    verbose = input("Verbose output for first few games? (y/n): ").lower() == 'y'
    
    simulator = GameSimulator(use_full_rules, num_detectives)
    stats, results = simulator.run_simulation(num_games, verbose=verbose)
    
    print_simulation_results(stats, results)
    
    save_file = input("Save results to file? (filename or 'n'): ")
    if save_file != 'n':
        if not save_file.endswith('.json'):
            save_file += '.json'
        save_results(stats, results, save_file)
