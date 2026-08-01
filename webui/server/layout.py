"""Board geometry for the web interface.

The Tkinter visualizer drew the board with matplotlib: node positions were
normalized to ``[-1, 1]`` using ``data/board_calibration.json`` and the board
photo was stretched over the same extent. This module reproduces that mapping
exactly, but emits SVG user units instead, so the browser overlay lands on the
same pixels the user already calibrated.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

BOARD_IMAGE_PATH = "data/board.jpg"
CALIBRATION_PATH = "data/board_calibration.json"

# Fallback canvas for games without a board photo (grid, path, cycle demos).
SYNTHETIC_SIZE = (1200.0, 900.0)

DEFAULT_CALIBRATION = {
    "x_offset": 0.0,
    "y_offset": 0.0,
    "x_scale": 1.0,
    "y_scale": 1.0,
    "image_alpha": 1.0,
}

# Transport identity. ``TransportType(4)`` is BLACK in the engine, but the board
# draws those routes as the river ferry, which is how the Tkinter legend labeled
# them too. Ticket contexts keep calling it "black" — see TICKET_STYLES.
TRANSPORT_STYLES: Dict[int, Dict[str, object]] = {
    1: {"key": "taxi", "name": "Taxi", "color": "#F5C542", "width": 2.6},
    2: {"key": "bus", "name": "Bus", "color": "#3E8FF0", "width": 3.4},
    3: {"key": "underground", "name": "Underground", "color": "#E4483C", "width": 4.2},
    4: {"key": "ferry", "name": "Ferry", "color": "#37B67A", "width": 3.4},
    5: {"key": "ferry", "name": "Ferry", "color": "#37B67A", "width": 3.4},
}

# How a transport reads when it is spent as a ticket rather than drawn as a route.
TICKET_STYLES: Dict[int, Dict[str, str]] = {
    1: {"key": "taxi", "name": "Taxi", "color": "#F5C542"},
    2: {"key": "bus", "name": "Bus", "color": "#3E8FF0"},
    3: {"key": "underground", "name": "Underground", "color": "#E4483C"},
    4: {"key": "black", "name": "Black", "color": "#C9D1DE"},
    5: {"key": "ferry", "name": "Ferry", "color": "#37B67A"},
}

# Some edge lists in data/ carry transport codes the engine has no name for
# (data/edgelist.csv stores most of its edges as type 0). Those connections are
# still real for basic movement rules, so draw them rather than hide them.
UNKNOWN_STYLE = {
    "key": "route",
    "name": "Route",
    "color": "#7C8798",
    "width": 2.4,
}

# Station ring color: the fastest line a station is on wins, the way a real
# transit map marks interchanges.
_RANK = [3, 2, 4, 5, 1]


@dataclass
class BoardLayout:
    """Screen-space board: everything the client needs to draw the map once."""

    width: float
    height: float
    nodes: List[Dict[str, object]] = field(default_factory=list)
    edges: List[Dict[str, object]] = field(default_factory=list)
    image: Optional[Dict[str, object]] = None
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        transports = {
            str(value): {
                "key": style["key"],
                "name": style["name"],
                "color": style["color"],
            }
            for value, style in TRANSPORT_STYLES.items()
        }
        # Include any transport code this board actually uses, named or not.
        for edge in self.edges:
            transports.setdefault(
                str(edge["transport"]),
                {"key": edge["key"], "name": edge["key"].title(), "color": edge["color"]},
            )

        return {
            "width": self.width,
            "height": self.height,
            "nodes": self.nodes,
            "edges": self.edges,
            "image": self.image,
            "transports": transports,
            "tickets": {str(v): s for v, s in TICKET_STYLES.items()},
        }


def load_calibration(path: str = CALIBRATION_PATH) -> Dict[str, float]:
    """Read the saved overlay calibration, falling back to an identity mapping."""
    calibration = dict(DEFAULT_CALIBRATION)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            calibration.update(
                {k: v for k, v in saved.items() if k in DEFAULT_CALIBRATION}
            )
    except (OSError, ValueError):
        pass
    return calibration


def read_jpeg_size(path: str) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` from a JPEG header without decoding pixels."""
    try:
        with open(path, "rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None
            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                size_bytes = handle.read(2)
                if len(size_bytes) < 2:
                    return None
                (segment_length,) = struct.unpack(">H", size_bytes)
                # SOF0..SOF15, skipping the non-frame markers in that range.
                if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                    handle.read(1)  # sample precision
                    height, width = struct.unpack(">HH", handle.read(4))
                    return width, height
                handle.seek(segment_length - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None


def _edge_transports(graph: nx.Graph) -> Dict[Tuple[int, int], List[int]]:
    """Collapse the graph into ``{(u, v): [transport values]}`` with u < v."""
    collected: Dict[Tuple[int, int], List[int]] = {}
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        transports = data.get("transports")
        if not transports:
            edge_type = data.get("edge_type")
            transports = [edge_type] if edge_type is not None else [1]
        key = (min(u, v), max(u, v))
        bucket = collected.setdefault(key, [])
        for transport in transports:
            if transport not in bucket:
                bucket.append(transport)
    for bucket in collected.values():
        bucket.sort()
    return collected


def _parallel_offsets(
    start: Tuple[float, float],
    end: Tuple[float, float],
    count: int,
    spacing: float,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Fan ``count`` parallel lines around the segment, as the old board did."""
    if count <= 1:
        return [(start, end)]

    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return [(start, end)] * count

    # Unit perpendicular.
    px, py = -dy / length, dx / length
    lines = []
    for index in range(count):
        shift = (index - (count - 1) / 2) * spacing
        offset = (px * shift, py * shift)
        lines.append(
            (
                (start[0] + offset[0], start[1] + offset[1]),
                (end[0] + offset[0], end[1] + offset[1]),
            )
        )
    return lines


def _spring_positions(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    """Same layout the Tkinter board used when a game had no board photo."""
    raw = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    return {node: (float(xy[0]), float(xy[1])) for node, xy in raw.items()}


_BOARD_POSITIONS: Optional[Dict[int, Tuple[float, float]]] = None
_BOARD_POSITIONS_LOADED = False


def _known_board_positions() -> Optional[Dict[int, Tuple[float, float]]]:
    """Station coordinates for the scanned London board, loaded once."""
    global _BOARD_POSITIONS, _BOARD_POSITIONS_LOADED
    if not _BOARD_POSITIONS_LOADED:
        _BOARD_POSITIONS_LOADED = True
        try:
            from ShadowChase.services.board_loader import load_board_graph_from_csv

            _, positions = load_board_graph_from_csv()
            _BOARD_POSITIONS = positions
        except Exception:  # noqa: BLE001 - falls back to a spring layout
            _BOARD_POSITIONS = None
    return _BOARD_POSITIONS


def resolve_node_positions(game) -> Optional[Dict[int, Tuple[float, float]]]:
    """Use the game's own coordinates, or the board's if it is the same map.

    Boards built from the edge lists carry no coordinates even when they are
    the London map. If the station set matches, the scanned board's geometry
    applies, so those games get the photo overlay too.
    """
    positions = getattr(game, "node_positions", None)
    if positions:
        return positions

    known = _known_board_positions()
    if not known:
        return None
    stations = set(game.graph.nodes())
    board_stations = set(known.keys())
    # Near-misses count: data/edgelist.csv covers 199 of the board's 200
    # stations. Small graphs that merely happen to number their nodes 1..10 do
    # not — they are their own maps, not this one.
    if stations <= board_stations and len(stations) >= len(board_stations) * 0.9:
        return {station: known[station] for station in stations}
    return None


def build_layout(game) -> BoardLayout:
    """Project ``game.graph`` into SVG user units."""
    graph = game.graph
    node_positions = resolve_node_positions(game)

    image: Optional[Dict[str, object]] = None
    positions: Dict[int, Tuple[float, float]] = {}

    if node_positions:
        calibration = load_calibration()
        size = read_jpeg_size(BOARD_IMAGE_PATH) or (2036, 1618)
        width, height = float(size[0]), float(size[1])

        xs = [pos[0] for pos in node_positions.values()]
        ys = [pos[1] for pos in node_positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = (x_max - x_min) * calibration["x_scale"] or 1.0
        y_range = (y_max - y_min) * calibration["y_scale"] or 1.0

        for node, (x, y) in node_positions.items():
            # Normalize to the matplotlib [-1, 1] extent, then to SVG pixels.
            nx_norm = 2 * ((x - x_min) / x_range) - 1 + calibration["x_offset"]
            ny_norm = -(2 * ((y - y_min) / y_range) - 1) + calibration["y_offset"]
            positions[node] = (
                (nx_norm + 1) / 2 * width,
                (1 - ny_norm) / 2 * height,
            )

        if os.path.exists(BOARD_IMAGE_PATH):
            image = {
                "url": "/api/board/image",
                "width": width,
                "height": height,
                "alpha": calibration.get("image_alpha", 1.0),
            }
    else:
        width, height = SYNTHETIC_SIZE
        margin = 70.0
        raw = _spring_positions(graph)
        for node, (x, y) in raw.items():
            positions[node] = (
                margin + (x + 1) / 2 * (width - 2 * margin),
                margin + (1 - y) / 2 * (height - 2 * margin),
            )

    spacing = max(6.0, min(width, height) * 0.011)

    edges: List[Dict[str, object]] = []
    node_transports: Dict[int, set] = {node: set() for node in graph.nodes()}
    for (u, v), transports in _edge_transports(graph).items():
        if u not in positions or v not in positions:
            continue
        lines = _parallel_offsets(positions[u], positions[v], len(transports), spacing)
        for transport, (start, end) in zip(transports, lines):
            style = TRANSPORT_STYLES.get(transport, UNKNOWN_STYLE)
            node_transports[u].add(transport)
            node_transports[v].add(transport)
            edges.append(
                {
                    "u": u,
                    "v": v,
                    "transport": transport,
                    "key": style["key"],
                    "color": style["color"],
                    "width": style["width"],
                    "x1": round(start[0], 2),
                    "y1": round(start[1], 2),
                    "x2": round(end[0], 2),
                    "y2": round(end[1], 2),
                }
            )

    nodes: List[Dict[str, object]] = []
    for node in graph.nodes():
        if node not in positions:
            continue
        x, y = positions[node]
        serves = sorted(node_transports.get(node, set()))
        ring = next(
            (TRANSPORT_STYLES[t]["color"] for t in _RANK if t in serves),
            UNKNOWN_STYLE["color"],
        )
        nodes.append(
            {
                "id": node,
                "x": round(x, 2),
                "y": round(y, 2),
                "serves": serves,
                "ring": ring,
            }
        )
    nodes.sort(key=lambda item: item["id"])

    return BoardLayout(
        width=width,
        height=height,
        nodes=nodes,
        edges=edges,
        image=image,
        positions=positions,
    )


def transport_label(transport_value: int, *, as_ticket: bool = False) -> str:
    """Human name for a transport value, in route or ticket vocabulary."""
    table = TICKET_STYLES if as_ticket else TRANSPORT_STYLES
    style = table.get(transport_value)
    return str(style["name"]) if style else f"Transport {transport_value}"


def iter_transport_values(transports: Iterable) -> List[int]:
    """Normalize enum members or raw ints into transport values."""
    values = []
    for transport in transports:
        value = getattr(transport, "value", transport)
        if value not in values:
            values.append(value)
    return values
