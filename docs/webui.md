# The Shadow Chase board (web interface)

A local FastAPI server renders the game in a browser. The rules still come from
`ShadowChase.core` — nothing about movement, tickets or win conditions was
touched. This layer only decides what you may click next and what the panel
says.

```bash
python -m webui                 # start the server and open the board
python -m webui --port 8123     # pick a port
python -m webui --no-browser    # start headless and open the URL yourself
python -m webui --reload        # reload the server when source files change
```

## Why a browser

The board is 200 stations photographed at 2036 × 1618. Matplotlib had to redraw
the whole figure for every hover and could not zoom. As SVG the map pans and
zooms at any magnification, stations stay crisp, and the same renderer serves
both the live game and the replay.

## Layout

```
┌──────┬────────────────────────────────┬──────────────────┐
│ rail │  reveal ruler · turn · Mr. X   │                  │
│      ├────────────────────────────────┤   control panel  │
│  ▣   │                                │                  │
│  ⟲   │        the map (SVG)           │   turn card      │
│  ▤   │        pan · zoom · click      │   where you can  │
│      │                                │     go           │
│  ▥   │                                │   tickets        │
│  ⌕   ├────────────────────────────────┤                  │
│      │  legend            − + Fit     │  [ actions ]     │
└──────┴────────────────────────────────┴──────────────────┘
```

The left rail holds the destinations: new game, saved games, export video, and
two live toggles — board photo and suspect stations. Training sits at the
bottom, disabled until the refactor lands.

The panel is one column of decisions and never scrolls its actions out of
reach: the primary button is pinned to the bottom.

## The reveal ruler

The strip across the top is the game's own schedule. One tick per Mr. X turn,
with the reveal turns (3, 8, 13, 18, 24) raised and marked in red; the current
turn is lit. The caption counts down to the next surfacing. This is the
question the whole game turns on and the old panel never showed it.

## Stations

Stations are drawn as Underground roundels. The ring carries the fastest line
that calls there; the disc carries whoever is standing on it — a numbered cyan
roundel for a detective, a dark roundel marked X for Mr. X, which flares red on
the turns he has to surface. Reachable stations grow a ring in the colour of
the ticket that gets you there, and the ones you have already picked this turn
are ringed in green.

With the board photo on, the map's own numbers do the labelling, so stations
fade back to hairlines and only the ones that matter stay lit. With the photo
off, the route lines and every station number are drawn instead.

## What each control does

| Control | Behaviour |
| --- | --- |
| Click a station | Picks it as the destination for the active piece |
| Ticket buttons | Appear when more than one ticket reaches that station |
| Confirm move | Sends the staged move to the engine (also `Enter`) |
| Stand still | Only enabled for a detective with no legal route |
| Undo picks | Drops this turn's staged picks; the game state is untouched |
| Arm double move | Mr. X only; the first leg keeps the turn, the second ends it |
| Play agent move | Shown when the side to move is agent-controlled |
| Board photo | Swaps the scanned map for the route diagram |
| Suspect stations | Shades every station Mr. X could be on, on detective turns |
| Saved games | Opens a replay in a new window |
| Export video | Renders a saved game to MP4 under `exports/` |

Keyboard: `Enter` confirms, `Esc` cancels a ticket choice or closes a dialog,
`f` fits the board. In the replay window: arrows step, `space` plays, `Home`
and `End` jump to the ends.

## Replay window

Opening a saved game loads every recorded state at once, so the timeline
scrubs without touching the server. The move log reads newest first and each
turn is clickable to jump there. Mr. X's position is always shown — the badge
says whether the detectives could see him at the time.

## Server shape

```
webui/
  __main__.py          launcher
  server/
    app.py             routes and the board catalogue
    session.py         the interactive state machine
    layout.py          board geometry, calibration, transport styles
    saves.py           finding and opening saved games
    replay.py          a finished game as scrubable steps
    video.py           background MP4 export
  client/
    index.html         play window
    replay.html        replay window
    css/               tokens, board, application
    js/                api, board renderer, two window controllers
```

Every mutating route answers with the whole state snapshot, so the browser
never merges partial updates. The engine stays authoritative; the client draws
what it is told.

Board geometry reproduces the old matplotlib mapping exactly: node coordinates
are normalized with `data/board_calibration.json` and projected onto the photo,
so the overlay lands on the pixels that were already calibrated.

## Boards

`extracted` (the scanned London board) is the default. `london-simple` is the
same map under basic movement rules. The rest are the small graphs the demos
used: the ten-station test board with and without tickets, a 3×3 grid, a path
and a cycle.

**`create_shadowChase_game` is not offered.** It reads `data/edgelist.csv`,
which stores 330 of its 447 edges as `edge_type 0` — a code `TransportType`
rejects — so ticket rules raise `ValueError: 0 is not a valid TransportType` on
the first Mr. X move. `main.py --demo board` hits the same crash today. The
board is a duplicate of the London map, which already works from
`data/edges.csv`, so it is left out rather than shipped broken. Fixing the CSV
would bring it back.

## Dependencies

`fastapi` and `uvicorn[standard]`, installed into `.venv`. Neither touches the
pinned CUDA build of PyTorch.

```bash
uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"
```
