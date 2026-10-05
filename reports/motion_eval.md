# Motion letters (J, Z): evaluation

Generated 2026-10-04 20:48 by `scripts/evaluate_motion.py`.

Split: 5-fold, grouped by clip. J/Z recorded by: p01. **Only one person, so this is NOT person-independent; expect worse results for other people.**

Data: 125 clips, windows: J 130, Z 112, none 1089. Skipped (no stroke recorded): p01/s02/Z_1, p01/s02/Z_2

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right letter fired; false alarm = J or Z fired on a static-letter clip.

| Model | Window accuracy | J detected | Z detected | False alarms |
|---|---|---|---|---|
| logreg (best) | 0.986 | 17/20 | 17/18 | 0/87 |
| random_forest | 0.989 | 14/20 | 14/18 | 0/87 |
| mlp | 0.976 | 17/20 | 16/18 | 0/87 |

## Window confusion (logreg)

Rows: true, columns: predicted.

| | J | Z | none |
|---|---|---|---|
| **J** | 119 | 0 | 11 |
| **Z** | 0 | 110 | 2 |
| **none** | 5 | 0 | 1084 |

## Problem clips (logreg)

- Missed: p01/s02/J_11, p01/s02/J_14, p01/s02/Z_13, p01/s02/J_5
- Wrong letter: none
- False alarms: none
