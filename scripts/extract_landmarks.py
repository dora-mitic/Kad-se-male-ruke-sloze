"""Extract hand landmarks from the Kaggle "ASL Alphabet" images.

Only landmarks are kept, not images. Output: data/processed/kaggle.npz with
    points      (N, 21, 3) raw landmarks in pixel units (normalised later, at training)
    handedness  (N,)       "Left" / "Right" as reported by MediaPipe
    labels      (N,)       letter
    person      (N,)       "kaggle" (the dataset is treated as one person)
    files       (N,)       source image name, for debugging

Usage:
    python scripts/extract_landmarks.py                 # all images (takes a while)
    python scripts/extract_landmarks.py --per-class 200 # quick run
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from signtrainer import config
from signtrainer.capture import first_hand, load_hand_landmarker

KAGGLE_DIR = config.DATA_DIR / "kaggle" / "asl_alphabet_train" / "asl_alphabet_train"
OUT_PATH = config.DATA_DIR / "processed" / "kaggle.npz"

# Static letters only: J and Z need motion (M6); the dataset's extra classes
# ("space", "del", "nothing") are not letters.
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]

_landmarker = None


def _init_worker() -> None:
    global _landmarker
    _landmarker = load_hand_landmarker(vision.RunningMode.IMAGE)


def _process(path: Path):
    # cv2.imread can't open non-ASCII paths on Windows; decode from bytes instead.
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    pts, handedness = first_hand(result, bgr.shape[1], bgr.shape[0])
    if pts is None:
        return None
    return pts, handedness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=None, help="max images per letter")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not KAGGLE_DIR.exists():
        raise SystemExit(f"Not found: {KAGGLE_DIR}\nSee DATA.md for how to download the dataset.")

    points, handedness, labels, files = [], [], [], []
    stats = {}
    start = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for letter in LETTERS:
            paths = sorted((KAGGLE_DIR / letter).glob("*.jpg"))[: args.per_class]
            results = pool.map(_process, paths, chunksize=32)
            found = 0
            for path, res in zip(paths, results):
                if res is None:
                    continue
                points.append(res[0])
                handedness.append(res[1])
                labels.append(letter)
                files.append(path.name)
                found += 1
            stats[letter] = {"images": len(paths), "hands_found": found}
            print(f"{letter}: {found}/{len(paths)} hands found  ({time.time() - start:.0f} s)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        points=np.stack(points),
        handedness=np.array(handedness),
        labels=np.array(labels),
        person=np.array(["kaggle"] * len(labels)),
        files=np.array(files),
    )
    total = sum(s["images"] for s in stats.values())
    print(f"\nSaved {len(labels)} samples from {total} images to {OUT_PATH}")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "kaggle_extraction.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
