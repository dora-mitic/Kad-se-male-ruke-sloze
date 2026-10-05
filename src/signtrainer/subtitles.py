"""Hold-to-confirm: turns the jittery per-frame guesses into subtitle text.

A static letter is written into the subtitle once it has been recognised for
CONFIRM_SECONDS without interruption (flickers shorter than GRACE_SECONDS are
forgiven). J and Z are written at once, because the stroke itself was the hold.

After a letter is written, the same letter can't be written again until the guess
changes or the hand leaves. That also blocks the letter the hand still shows at the
end of a J/Z stroke (usually I or D), so finishing a J doesn't also write an I.

The hand leaving the camera for SPACE_SECONDS ends a word; for CLEAR_SECONDS it
clears the text (the next visitor starts fresh). The UI can also edit the text
(delete a letter, add a space, clear) from the web server's thread.

Some signs are written as a symbol, e.g. "I love you" as a heart (config.SUBTITLE_SYMBOLS).
"""

import threading
from dataclasses import dataclass

VARIATION_SELECTOR = "️"  # the invisible half of emoji like ❤️

from signtrainer import config


@dataclass
class SubtitleState:
    text: str = ""
    candidate: str | None = None  # letter (or symbol) being held right now, not yet written
    progress: float = 0.0  # 0..1, how far the candidate is towards being written


class HoldToConfirm:
    def __init__(self,
                 confirm_seconds: float = config.CONFIRM_SECONDS,
                 min_confidence: float = config.MIN_CONFIDENCE,
                 grace_seconds: float = config.GRACE_SECONDS,
                 space_seconds: float = config.SPACE_SECONDS,
                 clear_seconds: float = config.CLEAR_SECONDS,
                 max_chars: int = config.SUBTITLE_MAX_CHARS,
                 symbols: dict | None = None):
        self.confirm_seconds = confirm_seconds
        self.min_confidence = min_confidence
        self.grace_seconds = grace_seconds
        self.space_seconds = space_seconds
        self.clear_seconds = clear_seconds
        self.max_chars = max_chars
        self.symbols = config.SUBTITLE_SYMBOLS if symbols is None else symbols

        self.text = ""
        self.candidate: str | None = None
        self.since = 0.0  # when the candidate started
        self.last_seen = 0.0  # last time the candidate was the guess
        self.blocked: set[str] = set()  # letters that can't be written until released
        self.hand_gone_since: float | None = None
        self.auto_spaced = False  # the space for this absence was already added
        self._lock = threading.Lock()  # the camera thread updates, the web server edits

    def update(self, now: float, hand: bool, letter: str | None, confidence: float,
               motion_letter: str | None = None) -> SubtitleState:
        """Feed one camera frame.

        Args:
            hand: whether a hand is visible.
            letter, confidence: the per-frame static guess.
            motion_letter: J or Z on the one frame the motion model fires (else None).
        """
        with self._lock:
            return self._update(now, hand, letter, confidence, motion_letter)

    def backspace(self) -> None:
        with self._lock:
            self.text = self.text.removesuffix(VARIATION_SELECTOR)[:-1]

    def space(self) -> None:
        with self._lock:
            if self.text and not self.text.endswith(" "):
                self._write(" ")

    def clear(self) -> None:
        with self._lock:
            self.text = ""

    def _update(self, now, hand, letter, confidence, motion_letter) -> SubtitleState:
        if not hand:
            self._hand_gone(now)
            return self.state(now)
        self.hand_gone_since = None

        if motion_letter is not None:  # a new stroke is always deliberate, even JJ
            self._write(motion_letter)
            self.blocked = {motion_letter} | ({letter} if letter else set())
            self.candidate = None
            return self.state(now)

        guess = letter if letter and confidence >= self.min_confidence else None
        if guess is not None and guess not in self.blocked:
            self.blocked = set()  # a new letter releases the old ones
        elif guess is None and self.blocked and now - self.last_seen > self.grace_seconds:
            self.blocked = set()  # an unclear moment between letters also releases them

        if guess is not None and guess in self.blocked:
            self.candidate = None
            self.last_seen = now
            return self.state(now)

        if guess is not None and guess == self.candidate:
            self.last_seen = now
        elif guess is not None and (self.candidate is None or now - self.last_seen > self.grace_seconds):
            self.candidate, self.since, self.last_seen = guess, now, now
        elif self.candidate is not None and now - self.last_seen > self.grace_seconds:
            self.candidate = None  # lost it for too long: start over

        if self.candidate is not None and now - self.since >= self.confirm_seconds:
            self._write(self.candidate)
            self.blocked = {self.candidate}
            self.candidate = None
        return self.state(now)

    def state(self, now: float) -> SubtitleState:
        progress = 0.0
        if self.candidate is not None:
            progress = min(1.0, (now - self.since) / self.confirm_seconds)
        shown = self.symbols.get(self.candidate, self.candidate)
        return SubtitleState(self.text, shown, progress)

    def _write(self, label: str) -> None:
        text = self.text + self.symbols.get(label, label)
        self.text = text[-self.max_chars:].removeprefix(VARIATION_SELECTOR)

    def _hand_gone(self, now: float) -> None:
        self.candidate = None
        self.blocked = set()
        if self.hand_gone_since is None:
            self.hand_gone_since, self.auto_spaced = now, False
            return
        away = now - self.hand_gone_since
        if away >= self.clear_seconds:
            self.text = ""
        elif away >= self.space_seconds and not self.auto_spaced:
            # Only once per absence, so deleting the space with the hand away sticks.
            self.auto_spaced = True
            if self.text and not self.text.endswith(" "):
                self._write(" ")
