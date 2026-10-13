"""Phrase signs (HELLO, THANK YOU, ...): a movement at a place on the body.

HELLO starts at the forehead and moves out like a salute; THANK YOU starts at the
chin and moves forward and down. So, like J/Z, the classifier looks at a window of
frames, but it also needs to know where the hand is relative to the body. From the
pose we take the chin and the shoulder width (body.compute_anchors) and add the
path of the wrist and middle fingertip relative to the chin, in shoulder widths.

Features of one window:
  * motion.window_features: hand paths (centred) and the average handshape
  * the body-relative paths of the wrist and middle fingertip, resampled

Windows whose frames have no pose can't be classified.
"""

from collections import deque
from pathlib import Path

import joblib
import numpy as np

from signtrainer import body, config
from signtrainer.features import WRIST
from signtrainer.motion import NONE, RESAMPLE, majority_hand, window_features, windows

PHRASE_LABELS = tuple(config.PHRASE_WORDS)
PHRASE_MODEL_PATH = config.MODELS_DIR / "phrases.joblib"

WINDOW = 32  # frames (~1.8 s), same as the motion letters
MIN_FRAMES = 16
BODY_TRACKED = (WRIST, 12)  # wrist and middle fingertip
MIN_POSE_FRAMES = 0.5  # share of the window's frames that must have a pose


def body_frame(poses) -> tuple[np.ndarray, float] | None:
    """Median chin position and shoulder width over the window's frames with a pose."""
    chins, scales = [], []
    for pose in poses:
        if pose is None or np.isnan(pose).any():
            continue
        try:
            anchors, scale = body.compute_anchors(pose)
        except ValueError:
            continue
        chins.append(anchors["chin"])
        scales.append(scale)
    if len(chins) < MIN_POSE_FRAMES * len(poses):
        return None
    return np.median(chins, axis=0), float(np.median(scales))


def phrase_features(points, handedness, poses) -> np.ndarray | None:
    """Feature vector for one window, or None without enough pose frames.

    Args:
        points: (T, 21, 3) hand landmarks in pixel units.
        handedness: (T,) "Left" / "Right".
        poses: (T, 33, 3) pose landmarks in pixel units (NaN or None where missing).
    """
    frame = body_frame(poses)
    if frame is None:
        return None
    chin, scale = frame
    pts = np.asarray(points, dtype=np.float32)
    rel = (pts[:, BODY_TRACKED, :2] - chin) / scale
    if majority_hand(handedness) == "Left":
        rel[..., 0] *= -1  # a left-handed signer is the mirror image
    t_old = np.linspace(0.0, 1.0, len(pts))
    t_new = np.linspace(0.0, 1.0, RESAMPLE)
    resampled = np.stack([
        np.stack([np.interp(t_new, t_old, rel[:, k, d]) for d in range(2)], axis=1)
        for k in range(len(BODY_TRACKED))
    ])
    return np.concatenate([window_features(pts, handedness), resampled.reshape(-1)]).astype(np.float32)


def clip_windows(points, label: str) -> list[tuple[tuple[int, int], str]]:
    """Training windows from one recorded clip, as ((start, end), label) pairs.

    A phrase clip is the sign from start to end (the hand is lowered at the end),
    so windows that end in the second half of the clip see most of the sign and
    are positives. Every window of any other clip is "none".
    """
    n = len(points)
    if label not in PHRASE_LABELS:
        return [(w, NONE) for w in windows(n, range(0, n, 2)) if w[1] - w[0] + 1 >= MIN_FRAMES]
    return [(w, label) for w in windows(n, range(n // 2, n)) if w[1] - w[0] + 1 >= MIN_FRAMES]


class PhraseTracker:
    """Live use: feed every camera frame, get a phrase when one was just signed.

    Same firing rule as the motion letters: confident for TRIGGER_FRAMES frames in
    a row, then shown for HOLD_SECONDS and the buffer is cleared.
    """

    THRESHOLD = 0.8
    TRIGGER_FRAMES = 3
    HOLD_SECONDS = 1.5
    MAX_MISSING = 10

    def __init__(self, path: Path = PHRASE_MODEL_PATH, model=None):
        self.model = model if model is not None else joblib.load(path)
        self.classes = list(self.model.classes_)
        self.points = deque(maxlen=WINDOW)
        self.hands = deque(maxlen=WINDOW)
        self.poses = deque(maxlen=WINDOW)
        self.missing = 0
        self.streak = 0
        self.shown: tuple[str, float] | None = None
        self.shown_until = 0.0
        self.just_fired = False

    def reset(self) -> None:
        self.points.clear()
        self.hands.clear()
        self.poses.clear()
        self.streak = 0

    def push(self, points, handedness, pose, now: float) -> tuple[str, float] | None:
        """Add one frame (points None = no hand). Returns (phrase, confidence) while one is shown."""
        self.just_fired = False
        if points is None:
            self.missing += 1
            if self.missing > self.MAX_MISSING:
                self.reset()
        else:
            self.missing = 0
            self.points.append(points)
            self.hands.append(handedness)
            self.poses.append(pose)
            self._classify(now)
        if self.shown and now < self.shown_until:
            return self.shown
        self.shown = None
        return None

    def _classify(self, now: float) -> None:
        if len(self.points) < MIN_FRAMES:
            return
        x = phrase_features(np.stack(self.points), list(self.hands), list(self.poses))
        if x is None:
            self.streak = 0
            return
        proba = self.model.predict_proba(x[None, :])[0]
        i = int(np.argmax(proba))
        label, conf = self.classes[i], float(proba[i])
        if label == NONE or conf < self.THRESHOLD:
            self.streak = 0
            return
        self.streak += 1
        if self.streak >= self.TRIGGER_FRAMES:
            self.shown, self.shown_until = (label, conf), now + self.HOLD_SECONDS
            self.just_fired = True
            self.reset()
