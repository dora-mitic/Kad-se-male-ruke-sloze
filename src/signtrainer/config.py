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

# MediaPipe hand landmark model, downloaded by scripts/download_models.py.
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe pose model (lite = fastest), used for forehead / chin / chest anchors.
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Draw the body anchors on the video (useful while developing phrase signs).
SHOW_BODY_ANCHORS = True

# Webcam
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Local web UI
HOST = "127.0.0.1"
PORT = 8000
