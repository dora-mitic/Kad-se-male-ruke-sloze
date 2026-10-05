from signtrainer.subtitles import HoldToConfirm

FPS = 20


def feed(sub, start, seconds, letter, confidence=0.9, hand=True):
    """Feed the same guess for `seconds` at FPS; returns the last state and end time."""
    t = start
    state = None
    for _ in range(int(seconds * FPS)):
        state = sub.update(t, hand, letter, confidence)
        t += 1 / FPS
    return state, t


def make():
    return HoldToConfirm(confirm_seconds=1.0, min_confidence=0.6, grace_seconds=0.25,
                         space_seconds=1.5, clear_seconds=8.0, max_chars=24)


def test_letter_is_written_after_holding_one_second():
    sub = make()
    state, t = feed(sub, 0.0, 0.9, "A")
    assert state.text == "" and state.candidate == "A" and 0.8 < state.progress < 1.0
    state, t = feed(sub, t, 0.2, "A")
    assert state.text == "A" and state.candidate is None


def test_holding_longer_writes_the_letter_once():
    state, _ = feed(make(), 0.0, 4.0, "B")
    assert state.text == "B"


def test_low_confidence_does_not_count():
    state, _ = feed(make(), 0.0, 2.0, "C", confidence=0.4)
    assert state.text == "" and state.candidate is None


def test_short_flicker_does_not_break_the_hold():
    sub = make()
    _, t = feed(sub, 0.0, 0.5, "D")
    _, t = feed(sub, t, 0.1, "F")  # two frames of a wrong guess
    state, _ = feed(sub, t, 0.45, "D")
    assert state.text == "D"


def test_long_interruption_starts_over():
    sub = make()
    _, t = feed(sub, 0.0, 0.8, "D")
    _, t = feed(sub, t, 0.5, "F")
    state, _ = feed(sub, t, 0.5, "D")
    assert state.text == "" and state.candidate == "D"


def test_same_letter_twice_needs_a_release():
    sub = make()
    _, t = feed(sub, 0.0, 1.2, "L")
    _, t = feed(sub, t, 0.4, None)  # unclear moment between the two L's
    state, _ = feed(sub, t, 1.2, "L")
    assert state.text == "LL"


def test_hand_away_adds_space_then_clears():
    sub = make()
    _, t = feed(sub, 0.0, 1.2, "A")
    state, t = feed(sub, t, 2.0, None, hand=False)
    assert state.text == "A "
    state, t = feed(sub, t, 7.0, None, hand=False)
    assert state.text == ""


def test_motion_letter_is_written_at_once_and_blocks_its_end_shape():
    sub = make()
    _, t = feed(sub, 0.0, 0.5, "I")
    state = sub.update(t, True, "I", 0.9, motion_letter="J")
    assert state.text == "J"
    state, t = feed(sub, t + 0.05, 2.0, "I")  # hand still shows I after the hook
    assert state.text == "J" and state.candidate is None
    state, _ = feed(sub, t, 1.2, "A")
    assert state.text == "JA"


def test_double_motion_letter():
    sub = make()
    sub.update(0.0, True, "I", 0.9, motion_letter="J")
    state = sub.update(1.5, True, "I", 0.9, motion_letter="J")
    assert state.text == "JJ"


def test_text_is_capped():
    sub = HoldToConfirm(max_chars=3)
    t = 0.0
    for letter in "ABCDE":
        _, t = feed(sub, t, 1.2, letter)
    assert sub.text == "CDE"


def test_edits_from_the_ui():
    sub = make()
    t = 0.0
    for letter in "AB":
        _, t = feed(sub, t, 1.2, letter)
    sub.space()
    sub.space()  # no double spaces
    assert sub.text == "AB "
    sub.backspace()
    sub.backspace()
    assert sub.text == "A"
    sub.clear()
    sub.backspace()  # nothing left to delete
    assert sub.text == ""


def test_i_love_you_is_written_as_a_heart():
    sub = HoldToConfirm(symbols={"ILY": "❤️"})
    state, t = feed(sub, 0.0, 0.5, "ILY")
    assert state.candidate == "❤️"
    state, t = feed(sub, t, 0.7, "ILY")
    assert state.text == "❤️"
    _, t = feed(sub, t, 1.2, "A")
    sub.backspace()
    sub.backspace()  # the heart goes in one press, with its invisible selector
    assert sub.text == ""


def test_deleting_the_auto_space_sticks():
    sub = make()
    _, t = feed(sub, 0.0, 1.2, "A")
    state, t = feed(sub, t, 2.0, None, hand=False)
    assert state.text == "A "
    sub.backspace()  # clicked with the mouse, hand still away
    state, t = feed(sub, t, 1.0, None, hand=False)
    assert state.text == "A"
    _, t = feed(sub, t, 0.5, "B")  # hand back: the next absence adds a space again
    state, _ = feed(sub, t, 2.0, None, hand=False)
    assert state.text == "A "
