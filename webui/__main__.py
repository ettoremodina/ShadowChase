"""Launch the Shadow Chase board in a browser.

    python -m webui                 # start the server and open the board
    python -m webui --port 8123     # pick a port
    python -m webui --no-browser    # start headless, open the URL yourself
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

# Boards, saves and calibration are all read with paths relative to the repo
# root, so run from there whatever directory the command was typed in.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Shadow Chase web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window on start")
    parser.add_argument("--reload", action="store_true",
                        help="Reload the server when source files change")
    args = parser.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser and not args.reload:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print(f"Shadow Chase is running at {url}")
    uvicorn.run(
        "webui.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
