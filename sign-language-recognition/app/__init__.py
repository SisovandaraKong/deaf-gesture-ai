"""
app/__init__.py

Flask application factory.
Creates the app, wires all services together, and registers blueprints.
"""
import logging
from flask import Flask
from flask_cors import CORS

from app.config import Config
from app import extensions


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # ── CORS (allow browser JS to call all API routes) ────────────────────────
    CORS(app)

    # ── Initialize services ───────────────────────────────────────────────────
    from app.models.model_loader import load_model_and_encoder
    from app.services.gesture_recognizer import GestureRecognizer
    from app.services.translator import TranslatorService
    from app.services.tts_service import TTSService
    from app.services.hand_detector import HandDetectorService

    model, le = load_model_and_encoder(
        app.config["MODEL_PATH"],
        app.config["LABEL_ENCODER_PATH"],
    )

    extensions.translator = TranslatorService()
    extensions.tts = TTSService()
    extensions.recognizer = GestureRecognizer(
        model=model,
        label_encoder=le,
        config=app.config,
        translator=extensions.translator,
    )
    extensions.detector = HandDetectorService(
        config=app.config,
        recognizer=extensions.recognizer,
    )
    extensions.detector.start()

    logger.info("All services initialized successfully.")

    # ── Register blueprints ───────────────────────────────────────────────────
    from app.routes.main import main_bp
    from app.routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
