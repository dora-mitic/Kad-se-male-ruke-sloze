import json
import time

import numpy as np

from signtrainer import recorder as recorder_module
from signtrainer.capture import HandFrame
from signtrainer.recorder import ClipRecorder


class FakeWorker:
    """Produces a new frame every few ms; the hand is visible only when `hand` is set."""

    def __init__(self, hand=True):
        self.hand = hand
        self.frame_id = 0

    def snapshot(self):
        self.frame_id += 1
        pts = np.full((21, 3), self.frame_id, dtype=np.float32) if self.hand else None
        return HandFrame(frame_id=self.frame_id, landmarks=pts, handedness="Right" if self.hand else None)


def make(tmp_path, monkeypatch, worker):
    monkeypatch.setattr(recorder_module, "OWN_DATA_DIR", tmp_path)
    return ClipRecorder(worker, "p01", "s01", "dnevno", ["A", "B"], rounds=1, clip_seconds=0.1)


def wait(rec):
    while rec.status()["recording"]:
        time.sleep(0.01)


def test_writes_meta(tmp_path, monkeypatch):
    make(tmp_path, monkeypatch, FakeWorker())
    meta = json.loads((tmp_path / "p01" / "s01" / "meta.json").read_text())
    assert meta["person"] == "p01" and meta["lighting"] == "dnevno"


def test_records_clip_with_landmarks(tmp_path, monkeypatch):
    rec = make(tmp_path, monkeypatch, FakeWorker())
    assert rec.start("A")
    wait(rec)
    status = rec.status()
    assert status["last_file"] == "A_1.npz" and status["last_frames"] > 0
    data = np.load(tmp_path / "p01" / "s01" / "A_1.npz")
    assert data["points"].shape == (status["last_frames"], 21, 3)
    assert set(data["handedness"]) == {"Right"}


def test_numbering_and_undo(tmp_path, monkeypatch):
    rec = make(tmp_path, monkeypatch, FakeWorker())
    for _ in range(2):
        rec.start("A")
        wait(rec)
    assert rec.status()["last_file"] == "A_2.npz"
    assert rec.delete_last() == "A_2.npz"
    assert not (tmp_path / "p01" / "s01" / "A_2.npz").exists()
    assert (tmp_path / "p01" / "s01" / "A_1.npz").exists()


def test_no_file_without_hand(tmp_path, monkeypatch):
    rec = make(tmp_path, monkeypatch, FakeWorker(hand=False))
    rec.start("B")
    wait(rec)
    assert rec.status() == {"recording": False, "last_frames": 0, "last_file": None}
    assert not list((tmp_path / "p01" / "s01").glob("*.npz"))


def test_cannot_start_twice(tmp_path, monkeypatch):
    rec = make(tmp_path, monkeypatch, FakeWorker())
    assert rec.start("A")
    assert not rec.start("B")
    wait(rec)
