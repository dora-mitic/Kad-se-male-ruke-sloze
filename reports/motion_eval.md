# Motion letters (J, Z): evaluation

Generated 2026-10-16 19:07 by `scripts/evaluate_motion.py`.

Split: 5-fold, grouped by clip. J/Z recorded by: p01, p02, p03, p04, p05.

Data: 528 clips, windows: J 272, Z 205, none 3636. Skipped (no stroke recorded): p01/s02/Z_1, p01/s02/Z_2, p05/s02/J_8, p05/s02/Z_1, p05/s02/Z_10, p05/s02/Z_3, p05/s02/Z_7

Live simulation: each held-out clip is played through the app's tracker. Detected = exactly
the right letter fired; false alarm = J or Z fired on a static-letter clip.

| Model | Window accuracy | J detected | Z detected | False alarms |
|---|---|---|---|---|
| logreg (best) | 0.991 | 50/53 | 41/47 | 0/428 |
| random_forest | 0.991 | 39/53 | 31/47 | 0/428 |
| mlp | 0.990 | 47/53 | 37/47 | 0/428 |

## Window confusion (logreg)

Rows: true, columns: predicted.

| | J | Z | none |
|---|---|---|---|
| **J** | 254 | 0 | 18 |
| **Z** | 0 | 197 | 8 |
| **none** | 7 | 4 | 3625 |

## Problem clips (logreg)

- Missed: p03/s02/Z_2, p05/s02/Z_4, p01/s02/J_5, p01/s02/Z_13, p02/s02/Z_3, p03/s02/J_1, p05/s02/Z_2, p05/s02/Z_5, p01/s02/J_11
- Wrong letter: none
- False alarms: none
