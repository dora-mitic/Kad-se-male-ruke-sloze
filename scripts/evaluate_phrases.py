"""Train and evaluate the phrase-sign model (HELLO, THANK YOU, ..., none) on own recordings.

Like scripts/evaluate_motion.py: training windows come from every own clip with a
pose (see phrases.clip_windows), the split is by clip, and each held-out clip is
also played frame by frame through the app's PhraseTracker. A phrase clip counts
as detected if exactly its phrase fires; any other clip counts as a false alarm if
a phrase fires at all.

Only phrases that were recorded are trained. The best model is retrained on all
windows and saved to models/phrases.joblib.

Usage:
    python scripts/evaluate_phrases.py
"""

import json
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from signtrainer import config, dataset
from signtrainer.model import candidates, save
from signtrainer.motion import NONE
from signtrainer.phrases import PHRASE_LABELS, PHRASE_MODEL_PATH, PhraseTracker, clip_windows, phrase_features

FPS = 18.0
FOLDS = 5


def build(clips):
    X, y, groups = [], [], []
    for gi, clip in enumerate(clips):
        for (s, e), lab in clip_windows(clip["points"], clip["label"]):
            x = phrase_features(clip["points"][s:e + 1], clip["handedness"][s:e + 1], clip["pose"][s:e + 1])
            if x is not None:
                X.append(x)
                y.append(lab)
                groups.append(gi)
    return np.array(X), np.array(y), np.array(groups)


def simulate(model, clip) -> set[str]:
    tracker = PhraseTracker(model=model)
    fired = set()
    for i, (p, h, q) in enumerate(zip(clip["points"], clip["handedness"], clip["pose"])):
        shown = tracker.push(p, h, q, now=i / FPS)
        if shown:
            fired.add(shown[0])
    return fired


def main() -> None:
    clips = [c for c in dataset.load_clips() if c["pose"] is not None]
    X, y, groups = build(clips)
    phrases = sorted(set(y) & set(PHRASE_LABELS))
    if not phrases:
        raise SystemExit("Need recorded phrase clips: python scripts/record_landmarks.py --labels HELLO ...")
    used = sorted(set(groups))
    people = sorted({clips[g]["person"] for g in used if clips[g]["label"] in phrases})
    print(f"{len(y)} windows from {len(used)} clips {dict(Counter(y))}")

    clip_label = np.array([clips[g]["label"] if clips[g]["label"] in phrases else NONE for g in groups])
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0).split(X, clip_label, groups))

    results = {}
    for name, model in candidates().items():
        detected = {p: [0, 0] for p in phrases}
        problems, false_alarms, other_clips = [], [], 0
        correct = total = 0
        for train, test in folds:
            m = clone(model).fit(X[train], y[train])
            correct += int(np.sum(m.predict(X[test]) == y[test]))
            total += len(test)
            for g in sorted(set(groups[test])):
                clip = clips[g]
                fired = simulate(m, clip)
                if clip["label"] in phrases:
                    detected[clip["label"]][1] += 1
                    if fired == {clip["label"]}:
                        detected[clip["label"]][0] += 1
                    else:
                        problems.append(f"{clip['name']} -> {'/'.join(sorted(fired)) or 'nothing'}")
                else:
                    other_clips += 1
                    if fired:
                        false_alarms.append(f"{clip['name']} -> {'/'.join(sorted(fired))}")
        hits = sum(h for h, _ in detected.values())
        clips_n = sum(n for _, n in detected.values())
        false_rate = len(false_alarms) / max(other_clips, 1)
        results[name] = {
            "window_accuracy": correct / total,
            "detected": {p: {"hits": h, "clips": n} for p, (h, n) in detected.items()},
            "detection": hits / clips_n,
            "false_alarm_rate": false_rate,
            "other_clips": other_clips,
            "false_alarms": false_alarms,
            "problems": problems,
            "score": (hits / clips_n + (1 - false_rate)) / 2,
        }
        print(f"{name:>14}: windows {correct / total:.3f}  detected {hits}/{clips_n}  "
              f"false alarms {len(false_alarms)}/{other_clips}")

    best = max(results, key=lambda n: results[n]["score"])
    print(f"\nBest: {best}. Retraining on all windows -> {PHRASE_MODEL_PATH}")
    save(clone(candidates()[best]).fit(X, y), PHRASE_MODEL_PATH)
    write_reports(results, best, phrases, people, len(used), Counter(y))


def write_reports(results, best, phrases, people, n_clips, counts) -> None:
    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    (out / "phrases_metrics.json").write_text(json.dumps(
        {"best": best, "phrases": phrases, "people": people, "models": results}, indent=2))
    lines = [
        "# Phrase signs: evaluation",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/evaluate_phrases.py`.",
        "",
        f"Split: {FOLDS}-fold, grouped by clip. Phrases recorded by: {', '.join(people)}."
        + (" **Only one person, so this is NOT person-independent.**" if len(people) < 2 else ""),
        "",
        f"Data: {n_clips} clips, windows: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "Live simulation: each held-out clip is played through the app's tracker. Detected = exactly",
        "the right phrase fired; false alarm = a phrase fired on any other clip.",
        "",
        "| Model | Window accuracy | " + " | ".join(phrases) + " | False alarms |",
        "|---|---|" + "---|" * len(phrases) + "---|",
    ]
    for name, r in results.items():
        mark = " (best)" if name == best else ""
        cells = " | ".join(f"{r['detected'][p]['hits']}/{r['detected'][p]['clips']}" for p in phrases)
        lines.append(f"| {name}{mark} | {r['window_accuracy']:.3f} | {cells} | "
                     f"{len(r['false_alarms'])}/{r['other_clips']} |")
    r = results[best]
    lines += ["", f"## Problem clips ({best})", "",
              f"- Phrase not (only) detected: {', '.join(r['problems']) or 'none'}",
              f"- False alarms: {', '.join(r['false_alarms']) or 'none'}"]
    (out / "phrases_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports written to {out}")


if __name__ == "__main__":
    main()
