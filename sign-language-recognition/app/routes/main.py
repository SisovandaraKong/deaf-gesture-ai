"""
app/routes/main.py

Main page routes:
  GET  /             → renders index.html (Jinja2 template)
  GET  /video_feed   → MJPEG webcam stream with MediaPipe landmarks drawn
"""
import logging
from flask import Blueprint, render_template, Response

from app import extensions

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Serve the main Sign Language Recognition UI page."""
    return render_template("index.html")


@main_bp.route("/video_feed")
def video_feed():
    """
    MJPEG stream of the webcam feed with hand–skeleton overlay.
    Consumed directly by the <img> tag in index.html as its src.
    """
    return Response(
        extensions.detector.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
