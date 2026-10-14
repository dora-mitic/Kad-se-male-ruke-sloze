# Phrase signs: evaluation

Generated 2026-10-14 18:41 by `scripts/evaluate_phrases.py`.

Split: 5-fold, grouped by clip. Phrases recorded by: p01, p02.

Data: 194 clips, windows: HELLO 346, THANKYOU 369, none 1431

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right phrase fired; false alarm = a phrase fired on any other clip.

| Model | Window accuracy | HELLO | THANKYOU | False alarms |
|---|---|---|---|---|
| logreg | 0.991 | 18/21 | 19/23 | 1/150 |
| random_forest (best) | 0.996 | 18/21 | 19/23 | 0/150 |
| mlp | 0.986 | 18/21 | 19/23 | 0/150 |

## Problem clips (random_forest)

- Phrase not (only) detected: p02/s03/HELLO_5 -> nothing, p02/s03/HELLO_2 -> nothing, p02/s03/HELLO_3 -> nothing, p02/s03/THANKYOU_1 -> nothing, p02/s03/THANKYOU_10 -> nothing, p02/s03/THANKYOU_2 -> nothing, p02/s03/THANKYOU_4 -> nothing
- False alarms: none
