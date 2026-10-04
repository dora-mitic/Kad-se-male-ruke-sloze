# Motion letters (J, Z): evaluation

Generated 2026-10-04 19:27 by `scripts/evaluate_motion.py`.

Split: 5-fold, grouped by clip. J/Z recorded by: p01. **Only one person, so this is NOT person-independent; expect worse results for other people.**

Data: 110 clips, windows: J 130, Z 112, none 954. Skipped (no stroke recorded): p01/s02/Z_1, p01/s02/Z_2

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right letter fired; false alarm = J or Z fired on a static-letter clip.

| Model | Window accuracy | J detected | Z detected | False alarms |
|---|---|---|---|---|
| logreg (best) | 0.992 | 18/20 | 17/18 | 0/72 |
| random_forest | 0.991 | 17/20 | 14/18 | 0/72 |
| mlp | 0.973 | 16/20 | 16/18 | 0/72 |

## Window confusion (logreg)

Rows: true, columns: predicted.

| | J | Z | none |
|---|---|---|---|
| **J** | 124 | 0 | 6 |
| **Z** | 0 | 110 | 2 |
| **none** | 1 | 0 | 953 |

## Problem clips (logreg)

- Missed: p01/s02/Z_13, p01/s02/J_5, p01/s02/J_11
- Wrong letter: none
- False alarms: none
