"""Record our own landmark dataset from the webcam (landmarks only, no images).

Opens a page that shows each letter in turn; press SPACE to record a short clip.

Usage:
    python scripts/record_landmarks.py --person p01 --session s01 --lighting "dnevno"
"""

import argparse
import threading
import webbrowser

import uvicorn

from signtrainer import config
from signtrainer.recorder import ClipRecorder
from signtrainer.server import create_app

STATIC_LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", required=True, help="pseudonymous id, e.g. p01 (never a real name)")
    parser.add_argument("--session", required=True, help="e.g. s01; use a new one per sitting")
    parser.add_argument("--lighting", default="", help='short note, e.g. "dnevno", "lampa", "mrak"')
    parser.add_argument("--rounds", type=int, default=3, help="passes through the alphabet")
    parser.add_argument("--seconds", type=float, default=2.0, help="length of one clip")
    parser.add_argument("--labels", default="".join(STATIC_LETTERS), help="letters to record")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    def make_recorder(worker):
        return ClipRecorder(worker, args.person, args.session, args.lighting,
                            list(args.labels), args.rounds, args.seconds)

    url = f"http://{config.HOST}:{args.port}/record"
    print(f"Open {url} (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()
    uvicorn.run(create_app(args.camera, make_recorder), host=config.HOST, port=args.port,
                log_level="warning", timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
