"""Train and evaluate the letter classifiers with a person-independent split.

Split:
  * Every own-recorded person (data/own/<person>/) is held out in turn
    (leave-one-person-out); training uses Kaggle plus all *other* people.
  * If there are no own recordings yet, falls back to a split inside Kaggle.
    That is NOT person-independent (same signer in train and test) and the report
    says so in bold; it is only a sanity check of the pipeline.

Signs that only the held-out person recorded (e.g. ILY, so far only p01) can't be
tested in that fold: the model never saw them. They are left out of the score and
listed as untested, instead of counting as 0%.

After evaluation, the best model is retrained on all data and saved to
models/letters.joblib. Results go to reports/.

Usage:
    python scripts/evaluate.py
"""

import json
import time
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from signtrainer import config, dataset
from signtrainer.model import LETTERS_MODEL_PATH, candidates, save


def kaggle_fallback_split(data: dict) -> list[tuple[str, np.ndarray]]:
    """Last 20% of each letter (by file order) as test. Same person => optimistic."""
    test = np.zeros(len(data["labels"]), bool)
    for label in np.unique(data["labels"]):
        idx = np.flatnonzero(data["labels"] == label)
        test[idx[int(len(idx) * 0.8):]] = True
    return [("kaggle (same-person fallback)", test)]


def main() -> None:
    kaggle, own = dataset.load_kaggle(), dataset.load_own()
    data = dataset.concat([kaggle, own])
    if not len(data["labels"]):
        raise SystemExit("No data. Run scripts/extract_landmarks.py and/or scripts/record_landmarks.py first.")

    X, y = dataset.features(data), data["labels"]
    classes = sorted(np.unique(y))
    own_people = sorted(np.unique(own["person"]))
    person_independent = bool(own_people)
    folds = [(p, data["person"] == p) for p in own_people] or kaggle_fallback_split(data)

    print(f"{len(y)} samples, {len(classes)} letters, people: {[str(p) for p in np.unique(data['person'])]}")
    if not person_independent:
        print("WARNING: no own recordings, evaluating on Kaggle itself (not person-independent).")

    # Labels with no training data when a person is held out can't be scored in that fold.
    untested = sorted({str(c) for _, test in folds for c in np.unique(y[test]) if c not in set(y[~test])})
    if untested:
        print(f"Not testable on a held-out person (nobody else recorded them): {untested}")

    results = {}
    for name, model in candidates().items():
        y_true, y_pred, per_fold = [], [], {}
        t = time.time()
        for fold_name, test in folds:
            m = clone(model).fit(X[~test], y[~test])
            scored = test & np.isin(y, m.classes_)
            pred = m.predict(X[scored])
            per_fold[fold_name] = float(np.mean(pred == y[scored]))
            y_true.append(y[scored])
            y_pred.append(pred)
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class = np.diag(cm) / cm.sum(axis=1)
        results[name] = {
            "accuracy": float(np.mean(y_true == y_pred)),
            "per_person": per_fold,
            "per_class": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, per_class)},
            "confusion": cm.tolist(),
            "seconds": round(time.time() - t, 1),
        }
        print(f"{name:>14}: accuracy {results[name]['accuracy']:.3f}  ({results[name]['seconds']} s)")

    best = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest: {best}. Retraining on all data -> {LETTERS_MODEL_PATH}")
    save(clone(candidates()[best]).fit(X, y))

    write_reports(results, best, classes, data, person_independent, untested)


def write_reports(results, best, classes, data, person_independent, untested) -> None:
    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    (out / "letters_metrics.json").write_text(json.dumps(
        {"best": best, "classes": classes, "person_independent": person_independent,
         "untested": untested, "models": results},
        indent=2,
    ))

    people, counts = np.unique(data["person"], return_counts=True)
    r = results[best]
    lines = [
        "# Letter recognition: evaluation",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/evaluate.py`.",
        "",
        "Split: " + (
            "**leave-one-person-out** over own recordings; Kaggle is always in training."
            if person_independent else
            "**NOT person-independent**: no own recordings yet, so the test set is the last 20% "
            "of Kaggle images per letter (same signer as training). These numbers are optimistic."
        ),
        "",
        "Data: " + ", ".join(f"{p} ({c} samples)" for p, c in zip(people, counts)),
    ]
    if untested:
        lines += ["", f"Not tested (only the held-out person recorded them, so they are left out of the "
                      f"score; the final model does include them): {', '.join(untested)}"]
    lines += [
        "",
        "## Models",
        "",
        "| Model | Accuracy | " + " | ".join(r["per_person"]) + " |",
        "|---|---|" + "---|" * len(r["per_person"]),
    ]
    for name, res in results.items():
        mark = " (best)" if name == best else ""
        lines.append(f"| {name}{mark} | {res['accuracy']:.3f} | "
                     + " | ".join(f"{v:.3f}" for v in res["per_person"].values()) + " |")

    lines += ["", f"## Per-letter accuracy ({best})", "", "| Letter | Accuracy |", "|---|---|"]
    for c in classes:
        a = r["per_class"][c]
        lines.append(f"| {c} | {'n/a' if a is None else f'{a:.2f}'} |")

    cm = np.array(r["confusion"])
    lines += ["", f"## Most common confusions ({best})", "", "| True | Predicted as | Count |", "|---|---|---|"]
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    for n, t, p in sorted(off, reverse=True)[:10]:
        lines.append(f"| {t} | {p} | {n} |")

    lines += ["", f"## Confusion matrix ({best})", "", "Rows: true letter, columns: predicted.", "",
              "| | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for c, row in zip(classes, cm):
        lines.append(f"| **{c}** | " + " | ".join(str(v) if v else "·" for v in row) + " |")

    (out / "letters_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports written to {out}")


if __name__ == "__main__":
    main()
