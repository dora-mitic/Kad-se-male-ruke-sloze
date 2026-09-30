"""Project-wide settings: language, paths, and recognition thresholds."""

from pathlib import Path

# Repository root (src/signtrainer/config.py -> ../../)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets"

# Sign language to use. Only "asl" is supported for now; "hzj" is a stretch goal.
LANGUAGE = "asl"
