"""Loading landmark datasets into one common format.

Every loader returns a dict of equal-length arrays:
    points      (N, 21, 3) raw landmarks in pixel units
    handedness  (N,)       "Left" / "Right"
    labels      (N,)       sign label, e.g. "A"
    person      (N,)       who signed it ("kaggle", "p01", ...), used for the split
"""

from pathlib import Path

import numpy as np

from signtrainer import config
from signtrainer.features import normalize_landmarks

KAGGLE_NPZ = config.DATA_DIR / "processed" / "kaggle.npz"
OWN_DIR = config.DATA_DIR / "own"

KEYS = ("points", "handedness", "labels", "person")


def empty() -> dict:
    return {
        "points": np.zeros((0, 21, 3), np.float32),
        "handedness": np.array([], dtype=str),
        "labels": np.array([], dtype=str),
        "person": np.array([], dtype=str),
    }


def load_kaggle(path: Path = KAGGLE_NPZ) -> dict:
    if not path.exists():
        return empty()
    data = np.load(path)
    return {k: data[k] for k in KEYS}


def load_own(root: Path = OWN_DIR) -> dict:
    """Read every clip in data/own/<person>/<session>/<label>_<n>.npz."""
    parts = []
    for clip in sorted(root.glob("*/*/*.npz")):
        data = np.load(clip)
        n = len(data["points"])
        parts.append({
            "points": data["points"],
            "handedness": data["handedness"],
            "labels": np.full(n, clip.stem.rsplit("_", 1)[0]),
            "person": np.full(n, clip.parent.parent.name),
        })
    return concat(parts)


def concat(parts: list[dict]) -> dict:
    parts = [p for p in parts if len(p["labels"])]
    if not parts:
        return empty()
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def select(data: dict, mask: np.ndarray) -> dict:
    return {k: data[k][mask] for k in KEYS}


def features(data: dict) -> np.ndarray:
    """Normalised feature matrix (N, 63)."""
    if not len(data["labels"]):
        return np.zeros((0, 63), np.float32)
    return np.stack([normalize_landmarks(p, h) for p, h in zip(data["points"], data["handedness"])])
