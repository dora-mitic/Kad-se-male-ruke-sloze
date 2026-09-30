"""Body reference points (forehead, chin, chest) from MediaPipe pose landmarks.

Many phrase signs are defined by *where* the hand is: HELLO starts at the forehead,
THANK YOU at the chin, I/ME and LOVE at the chest. The pose model gives eyes, mouth
and shoulders; from those we estimate the three anchors below.

All coordinates are 2D pixel units of the (mirrored) frame.
"""

import numpy as np

# MediaPipe pose landmark indices
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

FINGERTIPS = (4, 8, 12, 16, 20)
ANCHOR_NAMES = ("forehead", "chin", "chest")

# A fingertip closer than this (in shoulder widths) counts as "at" the anchor.
NEAR_THRESHOLD = 0.3


def compute_anchors(pose: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """Estimate anchors from pose landmarks.

    Args:
        pose: (33, 2+) pose landmarks in pixel units.

    Returns:
        ({"forehead": xy, "chin": xy, "chest": xy}, body scale in pixels).
        The scale is the shoulder width, used to make distances independent of how
        far the person stands from the camera.
    """
    pose = np.asarray(pose, dtype=np.float32)[:, :2]
    eyes = (pose[LEFT_EYE] + pose[RIGHT_EYE]) / 2
    mouth = (pose[MOUTH_LEFT] + pose[MOUTH_RIGHT]) / 2
    down = mouth - eyes  # eye -> mouth vector, points "down the face"

    shoulders = (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER]) / 2
    scale = float(np.linalg.norm(pose[LEFT_SHOULDER] - pose[RIGHT_SHOULDER]))
    if scale < 1e-6:
        raise ValueError("degenerate pose: shoulders coincide")

    anchors = {
        "forehead": eyes - 0.9 * down,  # roughly one eye-mouth distance above the eyes
        "chin": mouth + 0.6 * down,
        # sternum: a bit below the middle of the shoulder line
        "chest": shoulders + np.array([0.0, 0.3 * scale], dtype=np.float32),
    }
    return anchors, scale


def anchor_distances(hand: np.ndarray, anchors: dict[str, np.ndarray], scale: float) -> dict[str, float]:
    """Smallest fingertip-to-anchor distance for each anchor, in shoulder widths."""
    tips = np.asarray(hand, dtype=np.float32)[list(FINGERTIPS), :2]
    return {
        name: float(np.min(np.linalg.norm(tips - xy, axis=1)) / scale)
        for name, xy in anchors.items()
    }


def nearest_anchor(distances: dict[str, float], threshold: float = NEAR_THRESHOLD) -> str | None:
    """Name of the closest anchor if the hand is touching distance of it, else None."""
    if not distances:
        return None
    name = min(distances, key=distances.get)
    return name if distances[name] < threshold else None
