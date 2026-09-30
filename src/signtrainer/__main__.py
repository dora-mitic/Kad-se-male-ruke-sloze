"""Entry point: `python -m signtrainer` starts the local web UI and opens the browser."""

import argparse
import threading
import webbrowser

import uvicorn

from signtrainer import config
from signtrainer.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL sign trainer (local web UI)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    url = f"http://{config.HOST}:{args.port}"
    print(f"Open {url} (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    # Short graceful shutdown: the endless video stream would otherwise block Ctrl+C.
    uvicorn.run(
        create_app(args.camera),
        host=config.HOST,
        port=args.port,
        log_level="warning",
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
