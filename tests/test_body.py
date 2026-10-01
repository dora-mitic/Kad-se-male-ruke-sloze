import numpy as np
import pytest

from signtrainer import body


@pytest.fixture
def pose():
    """A person facing the camera: eyes at y=100, mouth at y=150, shoulders at y=250."""
    p = np.zeros((33, 3), np.float32)
    p[body.LEFT_EYE] = (180, 100, 0)
    p[body.RIGHT_EYE] = (220, 100, 0)
    p[body.MOUTH_LEFT] = (190, 150, 0)
    p[body.MOUTH_RIGHT] = (210, 150, 0)
    p[body.LEFT_SHOULDER] = (100, 250, 0)
    p[body.RIGHT_SHOULDER] = (300, 250, 0)
    return p


def hand_at(xy):
    h = np.zeros((21, 3), np.float32)
    h[:, :2] = (1000, 1000)  # far away
    h[8, :2] = xy  # index fingertip
    return h


def test_anchor_positions(pose):
    anchors, scale = body.compute_anchors(pose)
    assert scale == pytest.approx(200)
    np.testing.assert_allclose(anchors["forehead"], (200, 55))
    np.testing.assert_allclose(anchors["chin"], (200, 180))
    np.testing.assert_allclose(anchors["chest"], (200, 310))
    # top to bottom: forehead above chin above chest
    assert anchors["forehead"][1] < anchors["chin"][1] < anchors["chest"][1]


def test_scale_invariance(pose):
    a1, s1 = body.compute_anchors(pose)
    a2, s2 = body.compute_anchors(pose * 2)
    d1 = body.anchor_distances(hand_at((230, 190)), a1, s1)
    d2 = body.anchor_distances(hand_at((460, 380)), a2, s2)
    for name in body.ANCHOR_NAMES:
        assert d1[name] == pytest.approx(d2[name])


@pytest.mark.parametrize("xy, expected", [
    ((200, 60), "forehead"),
    ((205, 185), "chin"),
    ((190, 300), "chest"),
    ((600, 600), None),
])
def test_nearest_anchor(pose, xy, expected):
    anchors, scale = body.compute_anchors(pose)
    assert body.nearest_anchor(body.anchor_distances(hand_at(xy), anchors, scale)) == expected


def test_uses_closest_fingertip(pose):
    anchors, scale = body.compute_anchors(pose)
    d = body.anchor_distances(hand_at((200, 180)), anchors, scale)
    assert d["chin"] == pytest.approx(0)


def test_rejects_degenerate_pose():
    with pytest.raises(ValueError):
        body.compute_anchors(np.zeros((33, 3)))
