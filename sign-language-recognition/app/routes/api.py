"""
app/routes/api.py

All JSON API endpoints (registered under /api via url_prefix in __init__.py).

Routes:
  GET  /api/predict          → current sign + confidence (auto-updated by camera thread)
  GET  /api/sentence         → current sentence + Khmer translation
  GET  /api/status           → real-time status (sign, confidence, hold progress)
  POST /api/speak            → generate and stream TTS audio (MP3)
  POST /api/translate        → translate English text to Khmer
  POST /api/sentence/add     → manually add a sign to the sentence
  POST /api/sentence/clear   → clear the sentence
"""
import logging
from flask import Blueprint, jsonify, request, Response

from app import extensions

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# ── Recognition state ──────────────────────────────────────────────────────────

@api_bp.route("/predict", methods=["GET"])
def predict():
    """
    Return the full current recognition state.
    The background camera thread continuously updates this — no frame upload needed.
    """
    try:
        return jsonify(extensions.recognizer.get_full_state())
    except Exception as exc:
        logger.error(f"/api/predict error: {exc}")
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/status", methods=["GET"])
def get_status():
    """Return real-time sign detection status (current sign, confidence, hold progress)."""
    try:
        return jsonify(extensions.recognizer.get_status())
    except Exception as exc:
        logger.error(f"/api/status error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ── Sentence management ────────────────────────────────────────────────────────

@api_bp.route("/sentence", methods=["GET"])
def get_sentence():
    """Return the current English sentence and its Khmer translation."""
    try:
        return jsonify(extensions.recognizer.get_sentence_state())
    except Exception as exc:
        logger.error(f"/api/sentence GET error: {exc}")
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/sentence/add", methods=["POST"])
def add_to_sentence():
    """Manually add a sign word to the sentence."""
    try:
        data = request.get_json(silent=True) or {}
        sign = str(data.get("sign", "")).strip()
        if not sign:
            return jsonify({"error": "Field 'sign' is required."}), 400
        extensions.recognizer.add_sign(sign)
        return jsonify({"success": True, "sign": sign})
    except Exception as exc:
        logger.error(f"/api/sentence/add error: {exc}")
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/sentence/clear", methods=["POST"])
def clear_sentence():
    """Clear the accumulated sentence and Khmer translation."""
    try:
        extensions.recognizer.clear_sentence()
        return jsonify({"success": True})
    except Exception as exc:
        logger.error(f"/api/sentence/clear error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ── TTS ────────────────────────────────────────────────────────────────────────

@api_bp.route("/speak", methods=["POST"])
def speak():
    """
    Generate TTS audio and stream the MP3 back to the browser.

    Request JSON:
        { "text": "hello world", "lang": "km" }

    Response:
        audio/mpeg — the browser plays it via the HTML5 Audio API.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        lang = str(data.get("lang", "km")).strip()

        if not text:
            return jsonify({"error": "Field 'text' is required."}), 400
        if lang not in ("km", "en"):
            return jsonify({"error": "Field 'lang' must be 'km' or 'en'."}), 400

        audio_bytes = extensions.tts.generate(text, lang)
        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception as exc:
        logger.error(f"/api/speak error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ── Translation ────────────────────────────────────────────────────────────────

@api_bp.route("/translate", methods=["POST"])
def translate():
    """
    Translate English text to Khmer on demand.

    Request JSON:
        { "text": "hello" }

    Response JSON:
        { "khmer": "សួស្តី" }
    """
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "Field 'text' is required."}), 400

        khmer = extensions.translator.translate(text)
        return jsonify({"khmer": khmer})
    except Exception as exc:
        logger.error(f"/api/translate error: {exc}")
        return jsonify({"error": str(exc)}), 500
