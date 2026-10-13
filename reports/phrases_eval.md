# Phrase signs: evaluation

Generated 2026-10-13 21:14 by `scripts/evaluate_phrases.py`.

Split: 5-fold, grouped by clip. Phrases recorded by: p01. **Only one person, so this is NOT person-independent.**

Data: 167 clips, windows: HELLO 317, THANKYOU 337, none 1409

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right phrase fired; false alarm = a phrase fired on any other clip.

| Model | Window accuracy | HELLO | THANKYOU | False alarms |
|---|---|---|---|---|
| logreg | 0.987 | 15/15 | 15/15 | 2/137 |
| random_forest (best) | 0.998 | 15/15 | 15/15 | 0/137 |
| mlp | 0.983 | 15/15 | 15/15 | 0/137 |

## Problem clips (random_forest)

- Phrase not (only) detected: none
- False alarms: none
