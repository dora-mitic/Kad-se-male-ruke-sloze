# Phrase signs: evaluation

Generated 2026-10-16 19:17 by `scripts/evaluate_phrases.py`.

Split: 5-fold, grouped by clip. Phrases recorded by: p01, p02, p03, p04, p05.

Data: 394 clips, windows: HELLO 626, THANKYOU 522, none 2559

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right phrase fired; false alarm = a phrase fired on any other clip.

| Model | Window accuracy | HELLO | THANKYOU | False alarms |
|---|---|---|---|---|
| logreg | 0.977 | 32/39 | 26/38 | 5/317 |
| random_forest | 0.987 | 30/39 | 28/38 | 0/317 |
| mlp (best) | 0.984 | 31/39 | 30/38 | 2/317 |

## Problem clips (mlp)

- Phrase not (only) detected: p02/s03/HELLO_2 -> nothing, p02/s03/THANKYOU_1 -> nothing, p02/s03/THANKYOU_10 -> nothing, p03/s03/HELLO_3 -> nothing, p04/s03/HELLO_3 -> nothing, p02/s03/HELLO_3 -> nothing, p02/s03/THANKYOU_4 -> nothing, p03/s03/HELLO_1 -> nothing, p05/s03/THANKYOU_2 -> nothing, p04/s03/HELLO_4 -> nothing, p04/s03/THANKYOU_4 -> nothing, p05/s03/THANKYOU_10 -> nothing, p02/s03/HELLO_5 -> nothing, p03/s03/HELLO_2 -> nothing, p05/s03/THANKYOU_3 -> nothing, p02/s03/THANKYOU_2 -> nothing
- False alarms: p04/s01/Y_2 -> THANKYOU, p04/s02/J_6 -> THANKYOU
