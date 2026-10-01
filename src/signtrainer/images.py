"""Preparing reference sign images for the corner card."""

import cv2
import numpy as np

# Pixels this close to white (or this transparent) count as empty background.
WHITE_LEVEL = 245
ALPHA_LEVEL = 10
MARGIN = 0.04  # keep a small border around the drawing, as a fraction of its size


def trim_to_content(data: bytes) -> bytes:
    """Crop away empty (transparent or white) borders and return a PNG.

    The source images are hand-cut, so the drawing often sits in a lot of empty
    space; trimming lets it fill the card. Returns the input unchanged if it
    can't be decoded or is entirely empty.
    """
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3:
        return data

    if img.shape[2] == 4:
        content = img[:, :, 3] > ALPHA_LEVEL
        content &= img[:, :, :3].min(axis=2) < WHITE_LEVEL
    else:
        content = img.min(axis=2) < WHITE_LEVEL
    ys, xs = np.nonzero(content)
    if len(xs) == 0:
        return data

    pad = int(MARGIN * max(np.ptp(xs), np.ptp(ys))) + 1
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    ok, png = cv2.imencode(".png", img[y0:y1, x0:x1])
    return png.tobytes() if ok else data
