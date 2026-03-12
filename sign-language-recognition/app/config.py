"""
app/config.py

All configuration variables in one place.
Paths, thresholds, and Flask settings — nothing is hardcoded in service files.
"""
import os

# Root of the sign-language-recognition/ project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    # ── Model file paths ───────────────────────────────────────────────────────
    MODEL_PATH = os.path.join(MODELS_DIR, "sign_language_model.h5")
    LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
    HAND_LANDMARKER_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

    # ── MediaPipe detection settings (unchanged from realtime_recognition.py) ──
    NUM_HANDS = 1
    MIN_HAND_DETECTION_CONFIDENCE = 0.7
    MIN_HAND_PRESENCE_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5

    # ── Gesture recognition thresholds (unchanged from realtime_recognition.py)
    PREDICTION_BUFFER_SIZE = 10     # Smoothing window size
    CONFIDENCE_THRESHOLD = 0.75     # Minimum confidence to buffer a prediction
    HOLD_SECONDS = 1.5              # Seconds a sign must be held to confirm

    # ── Camera settings ────────────────────────────────────────────────────────
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    JPEG_QUALITY = 85               # MJPEG stream quality (1–100)
