# Motion letters (J, Z): evaluation

Generated 2026-10-14 18:27 by `scripts/evaluate_motion.py`.

Split: 5-fold, grouped by clip. J/Z recorded by: p01, p02.

Data: 236 clips, windows: J 168, Z 132, none 1804. Skipped (no stroke recorded): p01/s02/Z_1, p01/s02/Z_2

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right letter fired; false alarm = J or Z fired on a static-letter clip.

| Model | Window accuracy | J detected | Z detected | False alarms |
|---|---|---|---|---|
| logreg (best) | 0.988 | 27/30 | 27/28 | 0/178 |
| random_forest | 0.994 | 25/30 | 19/28 | 0/178 |
| mlp | 0.987 | 25/30 | 24/28 | 0/178 |

## Window confusion (logreg)

Rows: true, columns: predicted.

| | J | Z | none |
|---|---|---|---|
| **J** | 151 | 0 | 17 |
| **Z** | 0 | 130 | 2 |
| **none** | 5 | 1 | 1798 |

## Problem clips (logreg)

- Missed: p01/s02/J_14, p01/s02/Z_13, p01/s02/J_11, p01/s02/J_5
- Wrong letter: none
- False alarms: none
