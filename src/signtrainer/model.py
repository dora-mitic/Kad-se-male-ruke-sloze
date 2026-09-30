"""Letter classifier: candidate models, training, saving and loading."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signtrainer import config
from signtrainer.features import normalize_landmarks

LETTERS_MODEL_PATH = config.MODELS_DIR / "letters.joblib"


def candidates() -> dict:
    """The models we compare. All are small and fast on a CPU."""
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300, random_state=0),
        ),
    }


def save(model, path: Path = LETTERS_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


class LetterClassifier:
    """Wraps a trained model for live use: raw landmarks in, (label, confidence) out."""

    def __init__(self, path: Path = LETTERS_MODEL_PATH):
        self.model = joblib.load(path)
        self.classes = list(self.model.classes_)

    def predict(self, points: np.ndarray, handedness: str) -> tuple[str, float]:
        x = normalize_landmarks(points, handedness)[None, :]
        proba = self.model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])
