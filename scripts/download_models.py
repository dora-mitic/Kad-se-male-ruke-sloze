"""Download the MediaPipe models needed at runtime (one time, then everything is offline)."""

import urllib.request

from signtrainer import config


def main() -> None:
    path = config.HAND_MODEL_PATH
    if path.exists():
        print(f"Already present: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.HAND_MODEL_URL}")
    urllib.request.urlretrieve(config.HAND_MODEL_URL, path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
