"""Shared fixtures for Shadow Chase characterization tests."""

from pathlib import Path

import networkx as nx
import pytest

from ml_logger import get_logger
from ShadowChase.core.game import ShadowChaseGame, TransportType


logger = get_logger(__name__)


@pytest.fixture
def project_root() -> Path:
    """Return the repository root independently of the working directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def transport_graph() -> nx.MultiGraph:
    """Build a small deterministic graph containing every standard transport."""
    graph = nx.MultiGraph()
    edges = [
        (1, 2, TransportType.TAXI),
        (2, 3, TransportType.BUS),
        (3, 4, TransportType.UNDERGROUND),
        (4, 5, TransportType.TAXI),
        (5, 6, TransportType.BUS),
        (6, 1, TransportType.UNDERGROUND),
        (2, 5, TransportType.TAXI),
    ]
    for source, destination, transport in edges:
        graph.add_edge(source, destination, edge_type=transport.value)
    logger.debug("Created characterization graph with %d edges", len(edges))
    return graph


@pytest.fixture
def shadow_game(transport_graph: nx.MultiGraph) -> ShadowChaseGame:
    """Create the canonical initialized game used by behavior tests."""
    game = ShadowChaseGame(transport_graph, num_detectives=2)
    game.initialize_shadow_chase_game([1, 4], 3)
    return game

