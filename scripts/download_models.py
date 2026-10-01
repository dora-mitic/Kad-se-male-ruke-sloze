"""Download the MediaPipe models needed at runtime (one time, then everything is offline)."""

import urllib.request

from signtrainer import config


MODELS = [
    (config.HAND_MODEL_URL, config.HAND_MODEL_PATH),
    (config.POSE_MODEL_URL, config.POSE_MODEL_PATH),
]


def main() -> None:
    for url, path in MODELS:
        if path.exists():
            print(f"Already present: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
