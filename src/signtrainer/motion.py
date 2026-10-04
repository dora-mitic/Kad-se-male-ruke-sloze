"""Motion letters (J and Z): recognised from a short window of frames, not one frame.

J is the I handshape with the pinky drawing a hook; Z is the index finger drawing a Z.
In a single frame they look like I and D, so the classifier looks at the last
~1.8 s of hand frames: the paths of the wrist, index tip and pinky tip, plus the
average handshape. It answers "J", "Z" or "none".

Frames without a hand are simply not in the window (the recorder skips them too),
so training clips and the live buffer are built the same way.
"""

from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np

from signtrainer import config
from signtrainer.features import MIDDLE_MCP, WRIST, normalize_landmarks

MOTION_LABELS = ("J", "Z")
NONE = "none"
MOTION_MODEL_PATH = config.MODELS_DIR / "motion.joblib"

WINDOW = 32  # max frames in a window (~1.8 s at the ~18 FPS the app runs at)
MIN_FRAMES = 10  # shorter windows are not classified
RESAMPLE = 16  # every path is resampled to this many points
TRACKED = (WRIST, 8, 20)  # wrist, index tip (Z), pinky tip (J)
NUM_FEATURES = len(TRACKED) * RESAMPLE * 2 + 63

# Speed (hand lengths per frame) above which a frame counts as part of a stroke.
STROKE_SPEED = 0.15
STROKE_GAP = 4  # a stroke may pause this many frames (the corners of the Z)


def majority_hand(handedness) -> str:
    """MediaPipe sometimes flips the label for a frame or two; trust the majority."""
    return Counter(str(h) for h in handedness).most_common(1)[0][0]


def window_features(points, handedness) -> np.ndarray:
    """Feature vector for one window of frames.

    Args:
        points: (T, 21, 3) raw landmarks in pixel units, T >= 2.
        handedness: (T,) "Left" / "Right" per frame.
    """
    pts = np.array(points, dtype=np.float32)
    hand = majority_hand(handedness)
    scale = float(np.median(np.linalg.norm(pts[:, MIDDLE_MCP] - pts[:, WRIST], axis=1)))
    if scale < 1e-6:
        raise ValueError("degenerate hand in window")

    paths = pts[:, TRACKED, :2] / scale  # z is too noisy for paths
    paths -= paths.mean(axis=0)  # where on screen the letter is drawn doesn't matter
    if hand == "Left":
        paths[..., 0] *= -1  # a left-hand J is the mirror image of a right-hand one

    t_old = np.linspace(0.0, 1.0, len(pts))
    t_new = np.linspace(0.0, 1.0, RESAMPLE)
    resampled = np.stack([
        np.stack([np.interp(t_new, t_old, paths[:, k, d]) for d in range(2)], axis=1)
        for k in range(len(TRACKED))
    ])  # (tracked, RESAMPLE, 2)

    shape = np.mean([normalize_landmarks(p, hand) for p in pts], axis=0)
    return np.concatenate([resampled.reshape(-1), shape]).astype(np.float32)


def speeds(points) -> np.ndarray:
    """Per-frame speed of the fastest tracked point, in hand lengths per frame."""
    pts = np.asarray(points, dtype=np.float32)
    scale = float(np.median(np.linalg.norm(pts[:, MIDDLE_MCP] - pts[:, WRIST], axis=1)))
    v = np.linalg.norm(np.diff(pts[:, TRACKED, :2], axis=0), axis=2).max(axis=1) / max(scale, 1e-6)
    v = np.concatenate([[0.0], v])
    return np.convolve(v, np.ones(3) / 3, mode="same")


def find_stroke(points) -> tuple[int, int] | None:
    """(first, last) frame of the main stroke in a J/Z clip, or None if there is none.

    Active frames closer than STROKE_GAP are merged into one run; the run with the
    most movement is the stroke.
    """
    v = speeds(points)
    active = np.flatnonzero(v > STROKE_SPEED)
    if not len(active):
        return None
    runs, start = [], active[0]
    for a, b in zip(active[:-1], active[1:]):
        if b - a > STROKE_GAP:
            runs.append((start, a))
            start = b
    runs.append((start, active[-1]))
    first, last = max(runs, key=lambda r: v[r[0]:r[1] + 1].sum())
    return int(first), int(last)


def windows(n: int, ends) -> list[tuple[int, int]]:
    """(start, end) of the window ending at each frame in `ends` (inclusive), if long enough."""
    out = []
    for e in ends:
        s = max(0, e - WINDOW + 1)
        if 0 <= e < n and e - s + 1 >= MIN_FRAMES:
            out.append((s, e))
    return out


def clip_windows(points, label: str) -> list[tuple[tuple[int, int], str]]:
    """Training windows from one recorded clip, as ((start, end), label) pairs.

    J/Z clips: windows that end just after the stroke are positives; windows that
    end before the stroke started (hand held still in the start pose) are "none".
    Windows in between see only part of the stroke and are left out.
    Other letters: every window is "none".
    """
    n = len(points)
    if label not in MOTION_LABELS:
        return [(w, NONE) for w in windows(n, range(0, n, 2))]
    stroke = find_stroke(points)
    if stroke is None:
        return []
    first, last = stroke
    if last < MIN_FRAMES:  # the stroke happened during the countdown; only its end was recorded
        return []
    pos = [(w, label) for w in windows(n, range(last - 1, last + 6))]
    neg = [(w, NONE) for w in windows(n, range(0, first - 1, 2))]
    return pos + neg


class MotionTracker:
    """Live use: feed every camera frame, get J/Z when a stroke was just drawn.

    A letter fires when the model is confident for TRIGGER_FRAMES frames in a row.
    It then stays shown for HOLD_SECONDS, and the buffer is cleared so the same
    stroke doesn't fire twice.
    """

    THRESHOLD = 0.8
    TRIGGER_FRAMES = 2
    HOLD_SECONDS = 1.2
    MAX_MISSING = 10  # camera frames without a hand before the buffer is cleared

    def __init__(self, path: Path = MOTION_MODEL_PATH, model=None):
        self.model = model if model is not None else joblib.load(path)
        self.classes = list(self.model.classes_)
        self.points = deque(maxlen=WINDOW)
        self.hands = deque(maxlen=WINDOW)
        self.missing = 0
        self.streak = 0
        self.shown: tuple[str, float] | None = None
        self.shown_until = 0.0

    def reset(self) -> None:
        self.points.clear()
        self.hands.clear()
        self.streak = 0

    def push(self, points, handedness, now: float) -> tuple[str, float] | None:
        """Add one frame (points None = no hand). Returns (letter, confidence) while one is shown."""
        if points is None:
            self.missing += 1
            if self.missing > self.MAX_MISSING:
                self.reset()
        else:
            self.missing = 0
            self.points.append(points)
            self.hands.append(handedness)
            self._classify(now)
        if self.shown and now < self.shown_until:
            return self.shown
        self.shown = None
        return None

    def _classify(self, now: float) -> None:
        if len(self.points) < MIN_FRAMES:
            return
        x = window_features(np.stack(self.points), list(self.hands))[None, :]
        proba = self.model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        label, conf = self.classes[i], float(proba[i])
        if label == NONE or conf < self.THRESHOLD:
            self.streak = 0
            return
        self.streak += 1
        if self.streak >= self.TRIGGER_FRAMES:
            self.shown, self.shown_until = (label, conf), now + self.HOLD_SECONDS
            self.reset()
