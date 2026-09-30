"""Webcam capture and MediaPipe hand tracking, running in a background thread."""

import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from signtrainer import config
from signtrainer.model import LETTERS_MODEL_PATH, LetterClassifier

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
FINGERTIPS = {4, 8, 12, 16, 20}


@dataclass
class HandFrame:
    """Latest processing result, shared with the web server."""

    jpeg: bytes | None = None
    frame_id: int = 0
    landmarks: np.ndarray | None = None  # (21, 3) in pixel units, or None
    handedness: str | None = None
    prediction: str | None = None  # raw per-frame guess, no smoothing yet
    confidence: float = 0.0
    model_loaded: bool = False
    fps: float = 0.0
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


def load_hand_landmarker(
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
) -> vision.HandLandmarker:
    if not config.HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.HAND_MODEL_PATH} is missing; run: python scripts/download_models.py"
        )
    # Pass the model as bytes, not a path: MediaPipe on Windows fails on paths with
    # non-ASCII characters (this repo lives in "...slože").
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=config.HAND_MODEL_PATH.read_bytes()),
        running_mode=running_mode,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def first_hand(result, width: int, height: int) -> tuple[np.ndarray | None, str | None]:
    """Return (landmarks in pixel units, handedness) of the first detected hand.

    Pixel units keep x and y on the same scale (z uses width, like x). Live capture
    and dataset extraction both go through here, so their features match.
    """
    if not result.hand_landmarks:
        return None, None
    lm = result.hand_landmarks[0]
    pts = np.array([[p.x * width, p.y * height, p.z * width] for p in lm], dtype=np.float32)
    return pts, result.handedness[0][0].category_name


def draw_hand(frame: np.ndarray, pts: np.ndarray) -> None:
    for a, b in HAND_CONNECTIONS:
        pa = tuple(int(v) for v in pts[a, :2])
        pb = tuple(int(v) for v in pts[b, :2])
        cv2.line(frame, pa, pb, (255, 255, 255), 3, cv2.LINE_AA)
    for i, (x, y, _) in enumerate(pts):
        color = (44, 199, 255) if i in FINGERTIPS else (198, 196, 46)  # BGR: yellow tips, turquoise joints
        cv2.circle(frame, (int(x), int(y)), 7 if i in FINGERTIPS else 5, color, -1, cv2.LINE_AA)


class CameraWorker(threading.Thread):
    """Reads the webcam, runs hand tracking and keeps the latest result."""

    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.state = HandFrame()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def snapshot(self) -> HandFrame:
        with self.lock:
            return self.state

    def stop(self) -> None:
        self._stop_event.set()

    def _publish(self, **changes) -> None:
        with self.lock:
            self.state = HandFrame(**{**self.state.__dict__, **changes, "timestamp": time.time()})

    def run(self) -> None:
        try:
            landmarker = load_hand_landmarker()
        except Exception as exc:  # show the problem in the UI instead of crashing
            self._publish(error=str(exc))
            return

        # The letter model is optional: without it the UI still shows the hand.
        classifier = LetterClassifier() if LETTERS_MODEL_PATH.exists() else None

        # DirectShow opens much faster than the default backend on Windows.
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            self._publish(error="camera_unavailable")
            return

        start = time.monotonic()
        last_ts = -1
        fps = 0.0
        prev = time.monotonic()
        frame_id = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._publish(error="camera_unavailable")
                    time.sleep(0.5)
                    continue

                # Mirror first: it feels natural to the visitor, and MediaPipe's
                # handedness labels assume a mirrored (selfie) image.
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ts = int((time.monotonic() - start) * 1000)
                ts = max(ts, last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                )

                pts, handedness = first_hand(result, w, h)
                prediction, confidence = None, 0.0
                if pts is not None:
                    draw_hand(frame, pts)
                    if classifier is not None:
                        prediction, confidence = classifier.predict(pts, handedness)

                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now

                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_id += 1
                self._publish(
                    jpeg=jpeg.tobytes() if ok else None,
                    frame_id=frame_id,
                    landmarks=pts,
                    handedness=handedness,
                    prediction=prediction,
                    confidence=confidence,
                    model_loaded=classifier is not None,
                    fps=fps,
                    error=None,
                )
        finally:
            cap.release()
            landmarker.close()
