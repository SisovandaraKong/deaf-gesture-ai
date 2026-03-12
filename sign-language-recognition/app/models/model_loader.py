"""
app/models/model_loader.py

Loads the trained TensorFlow .h5 model and the sklearn LabelEncoder
from disk. Called once during app startup in create_app().
"""
import pickle
import logging

from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


def load_model_and_encoder(model_path: str, encoder_path: str):
    """
    Load the TensorFlow model and label encoder from their file paths.

    Args:
        model_path:   Absolute path to sign_language_model.h5
        encoder_path: Absolute path to label_encoder.pkl

    Returns:
        (model, label_encoder) tuple
    """
    logger.info(f"Loading TensorFlow model from: {model_path}")
    model = load_model(model_path)

    logger.info(f"Loading LabelEncoder from: {encoder_path}")
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    logger.info(
        f"Model ready — {len(le.classes_)} classes: {list(le.classes_)}"
    )
    return model, le
