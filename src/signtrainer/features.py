"""Hand landmark normalisation: turns 21 raw landmarks into a feature vector.

The goal is that the same handshape gives (almost) the same vector no matter where
the hand is in the frame, how far it is from the camera, or which hand is used.
"""

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9  # base knuckle of the middle finger


def normalize_landmarks(points, handedness: str = "Right") -> np.ndarray:
    """Normalise one hand and return a flat vector of 63 floats.

    Args:
        points: array-like of shape (21, 3) with x, y, z in a common unit
            (pixel coordinates from the capture code, not MediaPipe's 0..1 values,
            because those are stretched by the image aspect ratio).
        handedness: "Right" or "Left" as reported by MediaPipe on a mirrored frame.

    Steps:
        1. translate so the wrist is the origin (position invariance)
        2. divide by wrist -> middle-finger knuckle distance (distance invariance)
        3. mirror left hands onto right hands, so one model serves both hands
    """
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected shape ({NUM_LANDMARKS}, 3), got {pts.shape}")

    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        raise ValueError("degenerate hand: wrist and middle knuckle coincide")
    pts /= scale

    if handedness == "Left":
        pts[:, 0] *= -1

    return pts.reshape(-1)
