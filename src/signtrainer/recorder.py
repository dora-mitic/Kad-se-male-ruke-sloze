"""Records short clips of hand landmarks per letter, for building our own dataset.

Only landmarks are stored, never images. Layout:
    data/own/<person>/<session>/meta.json
    data/own/<person>/<session>/<label>_<n>.npz   (points (N, 21, 3), handedness (N,))
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from signtrainer import config
from signtrainer.capture import CameraWorker

OWN_DATA_DIR = config.DATA_DIR / "own"


class ClipRecorder:
    def __init__(self, worker: CameraWorker, person: str, session: str, lighting: str,
                 labels: list[str], rounds: int, clip_seconds: float):
        self.worker = worker
        self.labels = labels
        self.rounds = rounds
        self.clip_seconds = clip_seconds
        self.dir = OWN_DATA_DIR / person / session
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {
            "person": person,
            "session": session,
            "lighting": lighting,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        self._lock = threading.Lock()
        self._recording = False
        self._last_saved: Path | None = None
        self._last_frames = 0

    def status(self) -> dict:
        with self._lock:
            return {
                "recording": self._recording,
                "last_frames": self._last_frames,
                "last_file": self._last_saved.name if self._last_saved else None,
            }

    def start(self, label: str) -> bool:
        with self._lock:
            if self._recording:
                return False
            self._recording = True
        threading.Thread(target=self._record, args=(label,), daemon=True).start()
        return True

    def delete_last(self) -> str | None:
        with self._lock:
            path, self._last_saved = self._last_saved, None
        if path and path.exists():
            path.unlink()
            return path.name
        return None

    def _next_path(self, label: str) -> Path:
        n = 1
        while (self.dir / f"{label}_{n}.npz").exists():
            n += 1
        return self.dir / f"{label}_{n}.npz"

    def _record(self, label: str) -> None:
        points, handedness = [], []
        last_id = -1
        end = time.monotonic() + self.clip_seconds
        while time.monotonic() < end:
            snap = self.worker.snapshot()
            if snap.frame_id != last_id:
                last_id = snap.frame_id
                if snap.landmarks is not None:  # skip frames without a hand
                    points.append(snap.landmarks)
                    handedness.append(snap.handedness)
            time.sleep(0.005)

        path = None
        if points:
            path = self._next_path(label)
            np.savez_compressed(path, points=np.stack(points), handedness=np.array(handedness))
        with self._lock:
            self._recording = False
            self._last_saved = path
            self._last_frames = len(points)
