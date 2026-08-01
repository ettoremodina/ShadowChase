"""Selectable demonstrations of the game rules, boards, and visualizer.

Each demo builds a game, optionally plays it, and returns it. None of them
opens an ml_logger run or touches the run lifecycle: the command that selects a
demo owns the run and records whatever the demo returns.
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import networkx as nx

from ml_logger import get_logger
from ShadowChase.core.game import Game, Player, ShadowChaseGame
from ShadowChase.examples.example_games import (
    create_cycle_graph_game,
    create_extracted_board_game,
    create_grid_graph_game,
    create_path_graph_game,
    create_shadowChase_game,
    create_simple_shadow_chase_game,
    create_simple_test_shadow_chase_game,
    create_test_shadow_chase_game,
)
from ShadowChase.ui.game_visualizer import GameVisualizer


logger = get_logger(__name__)

DEFAULT_DEMO = "extracted"

# Preset starting cards from the physical board.
STARTING_CARDS = [13, 26, 29, 34, 50, 53, 91, 103, 112, 132, 138, 141, 155,
                  174, 197, 94, 117, 198]


@dataclass(frozen=True)
class Demo:
    """One selectable demonstration and the way a command should run it."""

    description: str
    play: Callable[..., Optional[Game]]
    default_detectives: int
    configurable_detectives: bool
    uses_visualizer: bool


def log_game_state(game: Game) -> None:
    """Log the current game state"""
    state = game.get_state_representation()
    logger.info(
        "Turn %s - %s's turn",
        state['turn_count'],
        state['turn'].upper(),
    )
    logger.info("Detectives at: %s", state['detective_positions'])
    logger.info("Mr. X at: %s", state['MrX_position'])
    logger.info("Game over: %s", state['game_over'])
    if state['winner']:
        logger.info("Winner: %s", state['winner'].upper())


def log_valid_moves(game: Game, player: Player, position: Optional[int] = None):
    """Log valid moves for a player and return them"""
    if player == Player.DETECTIVES and position is not None:
        moves = game.get_valid_moves(Player.DETECTIVES, position)
        if isinstance(game, ShadowChaseGame):
            logger.info(
                "Detective at %s can move to: %s",
                position,
                [(dest, transport.name) for dest, transport in moves],
            )
        else:
            logger.info(
                "Detective at %s can move to: %s",
                position,
                sorted(moves),
            )
    elif player == Player.MRX:
        moves = game.get_valid_moves(Player.MRX)
        if isinstance(game, ShadowChaseGame):
            logger.info(
                "Mr. X can move to: %s",
                [(dest, transport.name) for dest, transport in moves],
            )
        else:
            logger.info("Mr. X can move to: %s", sorted(moves))
    return moves


def log_ticket_allocation(game: ShadowChaseGame, num_detectives: int) -> None:
    """Log the ticket counts every player currently holds."""
    for detective in range(num_detectives):
        logger.info(
            "Detective %d: %s",
            detective + 1,
            game.get_detective_tickets(detective),
        )
    logger.info("Mr. X: %s", game.get_MrX_tickets())


def launch_visualizer(game: Game, headless: bool,
                      auto_positions: Optional[list] = None) -> None:
    """Open the interactive board unless the caller asked for headless output."""
    if headless:
        logger.info("Headless mode: skipping the interactive visualizer")
        return
    visualizer = GameVisualizer(game, auto_positions=auto_positions)
    visualizer.run()


def random_start_positions(game: Game, num_detectives: int,
                           candidates: Optional[list] = None) -> tuple:
    """Draw Mr. X and detective start positions without overlaps."""
    nodes = list(game.graph.nodes())
    if candidates is None:
        detective_positions = random.sample(nodes, num_detectives)
        MrX_position = random.choice(
            [node for node in nodes if node not in detective_positions]
        )
        return detective_positions, MrX_position

    valid = [position for position in candidates if position in nodes]
    if len(valid) < num_detectives + 1:
        remaining = [node for node in nodes if node not in valid]
        needed = (num_detectives + 1) - len(valid)
        valid.extend(random.sample(remaining, min(needed, len(remaining))))
    sample = random.sample(valid, num_detectives + 1)
    return sample[1:num_detectives + 1], sample[0]


def demo_basic_game(num_detectives: int = 2, headless: bool = False) -> Game:
    """Test basic game mechanics"""
    logger.info("Testing basic game mechanics")

    graph = nx.grid_2d_graph(3, 3)
    # Convert to simple integer labels
    graph = nx.relabel_nodes(
        graph,
        {node: index for index, node in enumerate(graph.nodes())},
    )
    logger.info("Graph nodes: %s", sorted(graph.nodes()))
    logger.info("Graph edges: %s", list(graph.edges()))

    game = Game(graph, 2)
    game.initialize_game([0, 1], 7)
    log_game_state(game)

    for index, detective_position in enumerate(game.game_state.detective_positions):
        logger.info("Detective %d:", index + 1)
        log_valid_moves(game, Player.DETECTIVES, detective_position)

    logger.info("Detectives move")
    new_detective_positions = [3, 4]
    success = game.make_move(new_positions=new_detective_positions)
    logger.info(
        "Detectives move to %s: %s",
        new_detective_positions,
        'Success' if success else 'Failed',
    )
    log_game_state(game)

    logger.info("Mr. X's turn")
    log_valid_moves(game, Player.MRX)
    success = game.make_move(new_MrX_pos=8)
    logger.info("Mr. X moves to %s: %s", 8, 'Success' if success else 'Failed')

    log_game_state(game)
    return game


def demo_game_until_end(num_detectives: int = 1, headless: bool = False) -> Game:
    """Play a simple game until completion"""
    logger.info("Playing a complete game")

    game = Game(nx.path_graph(5), 1)
    game.initialize_game([0], 4)  # detective at 0, MrX at 4

    turn = 0
    max_turns = 10

    while not game.is_game_over() and turn < max_turns:
        log_game_state(game)
        MrX_position = game.game_state.MrX_position
        detective_position = game.game_state.detective_positions[0]

        if game.game_state.turn == Player.DETECTIVES:
            # Simple strategy: detective moves toward MrX
            moves = log_valid_moves(game, Player.DETECTIVES, detective_position)
            best_move = min(
                [detective_position, *moves],
                key=lambda move: abs(move - MrX_position),
            )
            logger.info("Detective chooses to move to: %s", best_move)
            game.make_move(new_positions=[best_move])
        else:
            # Simple strategy: move away from detectives
            moves = log_valid_moves(game, Player.MRX)
            best_move = max(
                [MrX_position, *moves],
                key=lambda move: abs(move - detective_position),
            )
            logger.info("Mr. X chooses to move to: %s", best_move)
            game.make_move(new_MrX_pos=best_move)

        turn += 1

    log_game_state(game)
    if game.is_game_over():
        logger.info("Game ended after %d turns", turn)
    else:
        logger.info("Game stopped after %d turns (no winner yet)", max_turns)
    return game


def demo_shadow_chase_tickets(num_detectives: int = 2,
                              headless: bool = False) -> ShadowChaseGame:
    """Test Shadow Chase specific mechanics"""
    logger.info("Testing Shadow Chase ticket mechanics")

    game = create_shadowChase_game(2)
    game.initialize_shadow_chase_game([1, 13], 100)

    logger.info("Game initialized")
    logger.info("Detectives at: %s", [1, 13])
    logger.info(
        "Mr. X at: %s (hidden: %s)",
        100,
        not game.game_state.MrX_visible,
    )
    logger.info("Initial tickets:")
    log_ticket_allocation(game, 2)
    return game


def demo_shadow_chase_game(num_detectives: int = 3,
                           headless: bool = False) -> ShadowChaseGame:
    """Demonstrate full Shadow Chase game"""
    logger.info("Shadow Chase game demo")
    game = create_shadowChase_game(num_detectives)

    detective_positions, MrX_position = random_start_positions(game, num_detectives)
    game.initialize_shadow_chase_game(detective_positions, MrX_position)

    logger.info("Detectives at: %s", detective_positions)
    logger.info("Mr. X at: %s (hidden)", MrX_position)
    log_ticket_allocation(game, num_detectives)
    return game


def demo_test_shadow_chase(num_detectives: int = 2,
                           headless: bool = False) -> ShadowChaseGame:
    """Demonstrate test Shadow Chase game with small graph"""
    logger.info("Test Shadow Chase game demo (10 nodes)")

    game = create_test_shadow_chase_game(2)
    game.initialize_shadow_chase_game([1, 3], 8)

    logger.info("Detectives at: %s", [1, 3])
    logger.info("Mr. X at: %s", 8)
    log_ticket_allocation(game, 2)

    launch_visualizer(game, headless)
    return game


def demo_path_game(num_detectives: int = 1, headless: bool = False) -> Game:
    """Demonstrate game on path graph"""
    logger.info("Path graph game demo")
    game = create_path_graph_game(5, num_detectives)

    # Solver-based analysis was never implemented for this demo.
    logger.warning("No solver is available; skipping win-condition analysis")

    launch_visualizer(game, headless)
    return game


def demo_cycle_game(num_detectives: int = 1, headless: bool = False) -> Game:
    """Demonstrate game on cycle graph"""
    logger.info("Cycle graph game demo")
    game = create_cycle_graph_game(6, num_detectives)

    launch_visualizer(game, headless)
    return game


def demo_grid_game(num_detectives: int = 2, headless: bool = False) -> Game:
    """Demonstrate game on grid graph"""
    logger.info("Grid graph game demo")
    game = create_grid_graph_game(3, 3, num_detectives)

    launch_visualizer(game, headless)
    return game


def demo_simple_shadow_chase(num_detectives: int = 2,
                             headless: bool = False) -> ShadowChaseGame:
    """Demonstrate simplified Shadow Chase game"""
    logger.info("Simple Shadow Chase game demo")
    game = create_simple_shadow_chase_game(
        num_detectives=2,
        show_MrX=True,
        use_tickets=False,
    )
    game.initialize_game([1, 3], 100)

    launch_visualizer(game, headless)
    return game


def demo_simple_test_shadow_chase(num_detectives: int = 2,
                                  headless: bool = False) -> ShadowChaseGame:
    """Demonstrate simple test Shadow Chase game"""
    logger.info("Simple test Shadow Chase game demo")
    game = create_simple_test_shadow_chase_game(
        num_detectives=2,
        show_MrX=True,
        use_tickets=False,
    )
    game.initialize_game([1, 3], 8)

    launch_visualizer(game, headless)
    return game


def demo_shadow_chase_visualizer(num_detectives: int = 3,
                                 headless: bool = False) -> ShadowChaseGame:
    """Demonstrate full Shadow Chase game with visualizer"""
    logger.info("Shadow Chase game with visualizer")
    game = create_shadowChase_game(num_detectives)

    launch_visualizer(game, headless)
    return game


def demo_extracted_board_game(num_detectives: int = 3, headless: bool = False,
                              auto_init: bool = True) -> ShadowChaseGame:
    """Create Shadow Chase game using the extracted board data"""
    game = create_extracted_board_game(num_detectives)

    positions = None
    if auto_init:
        detective_positions, MrX_position = random_start_positions(
            game,
            num_detectives,
            candidates=list(STARTING_CARDS),
        )
        positions = detective_positions + [MrX_position]

    # Hand the visualizer its positions without auto-starting the game.
    launch_visualizer(game, headless, auto_positions=positions)
    return game


DEMOS: dict[str, Demo] = {
    "basic": Demo(
        description="Fixed 3x3 grid walkthrough of the basic move rules",
        play=demo_basic_game,
        default_detectives=2,
        configurable_detectives=False,
        uses_visualizer=False,
    ),
    "until-end": Demo(
        description="Scripted path-graph game played to completion",
        play=demo_game_until_end,
        default_detectives=1,
        configurable_detectives=False,
        uses_visualizer=False,
    ),
    "tickets": Demo(
        description="Full board setup printing the initial ticket allocation",
        play=demo_shadow_chase_tickets,
        default_detectives=2,
        configurable_detectives=False,
        uses_visualizer=False,
    ),
    "random-start": Demo(
        description="Full board initialized at random starting positions",
        play=demo_shadow_chase_game,
        default_detectives=3,
        configurable_detectives=True,
        uses_visualizer=False,
    ),
    "path": Demo(
        description="Path graph in the visualizer",
        play=demo_path_game,
        default_detectives=1,
        configurable_detectives=True,
        uses_visualizer=True,
    ),
    "cycle": Demo(
        description="Cycle graph in the visualizer",
        play=demo_cycle_game,
        default_detectives=1,
        configurable_detectives=True,
        uses_visualizer=True,
    ),
    "grid": Demo(
        description="3x3 grid graph in the visualizer",
        play=demo_grid_game,
        default_detectives=2,
        configurable_detectives=True,
        uses_visualizer=True,
    ),
    "simple": Demo(
        description="Ticketless full board in the visualizer",
        play=demo_simple_shadow_chase,
        default_detectives=2,
        configurable_detectives=False,
        uses_visualizer=True,
    ),
    "board": Demo(
        description="Full board in the visualizer without preset positions",
        play=demo_shadow_chase_visualizer,
        default_detectives=3,
        configurable_detectives=True,
        uses_visualizer=True,
    ),
    "test-board": Demo(
        description="Ten-node test board in the visualizer",
        play=demo_test_shadow_chase,
        default_detectives=2,
        configurable_detectives=False,
        uses_visualizer=True,
    ),
    "simple-test": Demo(
        description="Ticketless ten-node test board in the visualizer",
        play=demo_simple_test_shadow_chase,
        default_detectives=2,
        configurable_detectives=False,
        uses_visualizer=True,
    ),
    DEFAULT_DEMO: Demo(
        description="Extracted board with preset starting cards (default)",
        play=demo_extracted_board_game,
        default_detectives=5,
        configurable_detectives=True,
        uses_visualizer=True,
    ),
}
