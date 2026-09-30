import numpy as np
import pytest

from signtrainer.features import MIDDLE_MCP, NUM_LANDMARKS, WRIST, normalize_landmarks


@pytest.fixture
def hand():
    rng = np.random.default_rng(0)
    return rng.uniform(100, 500, size=(NUM_LANDMARKS, 3)).astype(np.float32)


def test_output_is_flat_vector(hand):
    assert normalize_landmarks(hand).shape == (NUM_LANDMARKS * 3,)


def test_wrist_is_origin(hand):
    pts = normalize_landmarks(hand).reshape(NUM_LANDMARKS, 3)
    np.testing.assert_allclose(pts[WRIST], 0, atol=1e-6)


def test_middle_knuckle_is_unit_distance(hand):
    pts = normalize_landmarks(hand).reshape(NUM_LANDMARKS, 3)
    assert np.linalg.norm(pts[MIDDLE_MCP]) == pytest.approx(1.0, abs=1e-5)


def test_invariant_to_translation(hand):
    shifted = hand + np.array([250.0, -80.0, 12.0])
    np.testing.assert_allclose(normalize_landmarks(hand), normalize_landmarks(shifted), atol=1e-5)


def test_invariant_to_scale(hand):
    # Same hand, twice as close to the camera
    np.testing.assert_allclose(normalize_landmarks(hand), normalize_landmarks(hand * 2.0), atol=1e-5)


def test_left_hand_is_mirrored_onto_right(hand):
    mirrored = hand.copy()
    mirrored[:, 0] *= -1  # the same handshape made with the other hand
    np.testing.assert_allclose(
        normalize_landmarks(hand, "Right"), normalize_landmarks(mirrored, "Left"), atol=1e-5
    )


def test_input_is_not_modified(hand):
    before = hand.copy()
    normalize_landmarks(hand, "Left")
    np.testing.assert_array_equal(hand, before)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((20, 3)))


def test_rejects_degenerate_hand():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((NUM_LANDMARKS, 3)))
