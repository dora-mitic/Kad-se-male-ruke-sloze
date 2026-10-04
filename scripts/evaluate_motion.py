"""Train and evaluate the motion-letter model (J, Z, none) on own recordings.

Training windows come from every own clip (see motion.clip_windows): J/Z clips give
positives, all other letters give "none". The split is by clip (StratifiedGroupKFold),
so windows from one clip are never in both train and test.

Honest caveat, also printed in the report: so far only one person (p01) recorded J
and Z, so this is NOT person-independent and the numbers are optimistic.

Two numbers per model:
  * window accuracy: how often a single window gets the right class
  * live simulation: each held-out clip is played frame by frame through the same
    MotionTracker the app uses. A J/Z clip counts as detected if exactly its letter
    fires; a static-letter clip counts as a false alarm if J or Z fires at all.

The best model is retrained on all windows and saved to models/motion.joblib.

Usage:
    python scripts/evaluate_motion.py
"""

import json
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from signtrainer import config, dataset
from signtrainer.model import candidates, save
from signtrainer.motion import (MOTION_LABELS, MOTION_MODEL_PATH, NONE, MotionTracker,
                                clip_windows, window_features)

FPS = 18.0  # frame rate of the recordings, used to give the simulation timestamps
FOLDS = 5


def build(clips):
    X, y, groups, skipped = [], [], [], []
    for gi, clip in enumerate(clips):
        wins = clip_windows(clip["points"], clip["label"])
        if clip["label"] in MOTION_LABELS and not any(lab != NONE for _, lab in wins):
            skipped.append(clip["name"])
            continue
        for (s, e), lab in wins:
            X.append(window_features(clip["points"][s:e + 1], clip["handedness"][s:e + 1]))
            y.append(lab)
            groups.append(gi)
    return np.array(X), np.array(y), np.array(groups), skipped


def simulate(model, clip) -> set[str]:
    """Letters that fire while the clip is played through the live tracker."""
    tracker = MotionTracker(model=model)
    fired = set()
    for i, (p, h) in enumerate(zip(clip["points"], clip["handedness"])):
        shown = tracker.push(p, h, now=i / FPS)
        if shown:
            fired.add(shown[0])
    return fired


def main() -> None:
    clips = dataset.load_clips()
    X, y, groups, skipped = build(clips)
    if not np.isin(list(MOTION_LABELS), y).all():
        raise SystemExit("Need recorded J and Z clips: python scripts/record_landmarks.py --labels JZ ...")
    used = sorted(set(groups))
    people = sorted({clips[g]["person"] for g in used if clips[g]["label"] in MOTION_LABELS})
    classes = sorted(set(y))
    print(f"{len(y)} windows from {len(used)} clips {dict(Counter(y))}; skipped {skipped}")

    # Stratify on the clip's own label, so every fold gets some J, Z and static clips.
    clip_label = np.array([clips[g]["label"] if clips[g]["label"] in MOTION_LABELS else NONE for g in groups])
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0).split(X, clip_label, groups))

    results = {}
    for name, model in candidates().items():
        y_true, y_pred = [], []
        detected = {lab: [0, 0] for lab in MOTION_LABELS}  # [hits, clips]
        wrong, false_alarms, static_clips, misses = [], [], 0, []
        for train, test in folds:
            m = clone(model).fit(X[train], y[train])
            y_true.append(y[test])
            y_pred.append(m.predict(X[test]))
            for g in sorted(set(groups[test])):
                clip = clips[g]
                fired = simulate(m, clip)
                if clip["label"] in MOTION_LABELS:
                    detected[clip["label"]][1] += 1
                    if fired == {clip["label"]}:
                        detected[clip["label"]][0] += 1
                    elif fired:
                        wrong.append(f"{clip['name']} -> {'/'.join(sorted(fired))}")
                    else:
                        misses.append(clip["name"])
                else:
                    static_clips += 1
                    if fired:
                        false_alarms.append(f"{clip['name']} -> {'/'.join(sorted(fired))}")
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        hits = sum(h for h, _ in detected.values())
        total = sum(n for _, n in detected.values())
        detection = hits / total
        false_rate = len(false_alarms) / static_clips
        results[name] = {
            "window_accuracy": float(np.mean(y_true == y_pred)),
            "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
            "detection": detection,
            "detected": {lab: {"hits": h, "clips": n} for lab, (h, n) in detected.items()},
            "false_alarm_rate": false_rate,
            "static_clips": static_clips,
            "false_alarms": false_alarms,
            "wrong_letter": wrong,
            "missed": misses,
            "score": (detection + (1 - false_rate)) / 2,
        }
        print(f"{name:>14}: windows {results[name]['window_accuracy']:.3f}  "
              f"detected {hits}/{total}  false alarms {len(false_alarms)}/{static_clips}")

    best = max(results, key=lambda n: results[n]["score"])
    print(f"\nBest: {best}. Retraining on all windows -> {MOTION_MODEL_PATH}")
    save(clone(candidates()[best]).fit(X, y), MOTION_MODEL_PATH)
    write_reports(results, best, classes, people, skipped, len(used), Counter(y))


def write_reports(results, best, classes, people, skipped, n_clips, counts) -> None:
    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    (out / "motion_metrics.json").write_text(json.dumps(
        {"best": best, "classes": classes, "people": people, "skipped": skipped, "models": results}, indent=2))

    r = results[best]
    lines = [
        "# Motion letters (J, Z): evaluation",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/evaluate_motion.py`.",
        "",
        f"Split: {FOLDS}-fold, grouped by clip. J/Z recorded by: {', '.join(people)}."
        + (" **Only one person, so this is NOT person-independent; expect worse results for"
           " other people.**" if len(people) < 2 else ""),
        "",
        f"Data: {n_clips} clips, windows: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + (f". Skipped (no stroke recorded): {', '.join(skipped)}" if skipped else ""),
        "",
        "Live simulation: each held-out clip is played through the app's tracker. Detected = exactly",
        "the right letter fired; false alarm = J or Z fired on a static-letter clip.",
        "",
        "| Model | Window accuracy | J detected | Z detected | False alarms |",
        "|---|---|---|---|---|",
    ]
    for name, res in results.items():
        d = res["detected"]
        mark = " (best)" if name == best else ""
        lines.append(
            f"| {name}{mark} | {res['window_accuracy']:.3f} | {d['J']['hits']}/{d['J']['clips']} | "
            f"{d['Z']['hits']}/{d['Z']['clips']} | {len(res['false_alarms'])}/{res['static_clips']} |")

    lines += ["", f"## Window confusion ({best})", "", "Rows: true, columns: predicted.", "",
              "| | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for c, row in zip(classes, r["confusion"]):
        lines.append(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |")

    lines += ["", f"## Problem clips ({best})", ""]
    for title, items in (("Missed", r["missed"]), ("Wrong letter", r["wrong_letter"]),
                         ("False alarms", r["false_alarms"])):
        lines.append(f"- {title}: {', '.join(items) if items else 'none'}")

    (out / "motion_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports written to {out}")


if __name__ == "__main__":
    main()
