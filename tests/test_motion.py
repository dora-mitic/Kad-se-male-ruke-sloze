import numpy as np
from sklearn.linear_model import LogisticRegression

from signtrainer import dataset, motion


def hand(offset=(0.0, 0.0), scale=1.0):
    """A fixed fake hand: wrist at the origin, middle knuckle 100 px above it."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-100, 100, (21, 3)).astype(np.float32)
    pts[motion.WRIST] = 0
    pts[motion.MIDDLE_MCP] = (0, -100, 0)
    pts *= scale
    pts[:, :2] += offset
    return pts


def stroke_clip(n=40, start=12, end=24, step=60.0, base=None):
    """Hand holds still, moves right between `start` and `end`, then holds still."""
    base = hand() if base is None else base
    xs = np.concatenate([np.zeros(start), np.arange(1, end - start + 1) * step,
                         np.full(n - end, (end - start) * step)])
    return np.stack([base + np.array([x, 0, 0], np.float32) for x in xs])


def test_features_shape():
    clip = stroke_clip()
    assert motion.window_features(clip, ["Right"] * len(clip)).shape == (motion.NUM_FEATURES,)


def test_features_ignore_position_and_distance():
    clip = stroke_clip()
    moved = clip * 2.0 + np.array([300, 150, 0], np.float32)
    hands = ["Right"] * len(clip)
    np.testing.assert_allclose(motion.window_features(clip, hands),
                               motion.window_features(moved, hands), atol=1e-4)


def test_left_hand_is_mirrored():
    clip = stroke_clip()
    mirrored = clip.copy()
    mirrored[..., 0] *= -1
    np.testing.assert_allclose(motion.window_features(clip, ["Right"] * len(clip)),
                               motion.window_features(mirrored, ["Left"] * len(clip)), atol=1e-4)


def test_majority_hand_ignores_flicker():
    assert motion.majority_hand(["Left"] * 8 + ["Right"] * 2) == "Left"


def test_find_stroke():
    first, last = motion.find_stroke(stroke_clip(start=12, end=24))
    assert 10 <= first <= 14 and 22 <= last <= 26


def test_no_stroke_in_still_clip():
    assert motion.find_stroke(np.stack([hand()] * 30)) is None


def test_clip_windows_labels():
    wins = motion.clip_windows(stroke_clip(), "J")
    labels = {lab for _, lab in wins}
    assert labels == {"J"}  # stroke starts at 12: too early for a 10-frame "none" window before it
    for (s, e), _ in wins:
        assert e - s + 1 >= motion.MIN_FRAMES and e - s + 1 <= motion.WINDOW

    static = motion.clip_windows(np.stack([hand()] * 30), "A")
    assert static and {lab for _, lab in static} == {motion.NONE}


def test_clip_windows_skip_stroke_during_countdown():
    assert motion.clip_windows(stroke_clip(start=0, end=6), "Z") == []


def test_tracker_fires_once_and_holds():
    j_hand = finger_hand({"pinky"})  # passes the J handshape check
    clips = [stroke_clip(base=j_hand), np.stack([j_hand] * 40)]
    X = np.stack([motion.window_features(c[s:e + 1], ["Right"] * (e - s + 1))
                  for c in clips for (s, e) in [(0, 31), (8, 39)]])
    model = LogisticRegression().fit(X, ["none", "J", "none", "none"])

    tracker = motion.MotionTracker(model=model)
    shown = [tracker.push(p, "Right", now=i / 18) for i, p in enumerate(stroke_clip(base=j_hand))]
    fired = [s for s in shown if s]
    assert fired and {s[0] for s in fired} == {"J"}
    assert len(tracker.points) < motion.WINDOW  # buffer was cleared after firing

    assert tracker.push(None, None, now=100.0) is None  # hold time is over


def test_load_own_skips_motion_letters(tmp_path):
    d = tmp_path / "p01" / "s02"
    d.mkdir(parents=True)
    for name in ("J_1", "I_1"):
        np.savez(d / f"{name}.npz", points=np.zeros((3, 21, 3), np.float32), handedness=np.full(3, "Right"))
    assert set(dataset.load_own(tmp_path)["labels"]) == {"I"}
    clips = dataset.load_clips(tmp_path)
    assert [c["name"] for c in clips] == ["p01/s02/I_1", "p01/s02/J_1"]
    assert clips[1]["label"] == "J" and clips[1]["person"] == "p01"


def finger_hand(up):
    """A hand with the listed fingers straight up and the others curled at the knuckle."""
    pts = np.zeros((21, 3), np.float32)
    pts[motion.MIDDLE_MCP] = (0, -100, 0)
    for name, (tip, pip) in motion.FINGERS.items():
        pts[pip] = (0, -140, 0)
        pts[tip] = (0, -200, 0) if name in up else (0, -110, 0)
    return pts


def test_handshape_check():
    i_shape = np.stack([finger_hand({"pinky"})] * 12)
    v_shape = np.stack([finger_hand({"index", "middle"})] * 12)
    d_shape = np.stack([finger_hand({"index"})] * 12)
    ily_shape = np.stack([finger_hand({"index", "pinky"})] * 12)
    assert motion.handshape_ok("J", i_shape) and not motion.handshape_ok("Z", i_shape)
    assert motion.handshape_ok("Z", d_shape) and not motion.handshape_ok("J", d_shape)
    assert not motion.handshape_ok("J", v_shape) and not motion.handshape_ok("Z", v_shape)
    assert not motion.handshape_ok("J", ily_shape)
