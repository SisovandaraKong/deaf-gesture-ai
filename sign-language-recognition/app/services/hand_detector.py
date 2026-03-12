"""
app/services/hand_detector.py

Manages the webcam capture loop and MediaPipe hand detection.
Runs in a background daemon thread so Flask stays non-blocking.

MediaPipe options, HAND_CONNECTIONS, drawing code (cv2.line / cv2.circle),
and the horizontal flip (cv2.flip) are preserved exactly from
realtime_recognition.py. Only the pygame/OpenCV window code is removed.
"""
import cv2
import time
import threading
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.utils.landmark_utils import normalize_landmarks

logger = logging.getLogger(__name__)

# Hand skeleton connections — identical list from realtime_recognition.py
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (0, 9),  (9, 10), (10, 11),(11, 12),
    (0, 13),(13, 14), (14, 15),(15, 16),
    (0, 17),(17, 18), (18, 19),(19, 20),
    (5, 9),  (9, 13), (13, 17),
]


class HandDetectorService:
    """
    Opens the webcam, runs MediaPipe HandLandmarker on each frame,
    draws the skeleton overlay, and feeds landmarks to GestureRecognizer.
    Exposes generate_frames() as an MJPEG generator for Flask.
    """

    def __init__(self, config, recognizer):
        self._config = config
        self._recognizer = recognizer

        # ── MediaPipe setup (exact options from realtime_recognition.py) ──────
        base_options = mp_python.BaseOptions(
            model_asset_path=config["HAND_LANDMARKER_PATH"]
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.get("NUM_HANDS", 1),
            min_hand_detection_confidence=config.get("MIN_HAND_DETECTION_CONFIDENCE", 0.7),
            min_hand_presence_confidence=config.get("MIN_HAND_PRESENCE_CONFIDENCE", 0.7),
            min_tracking_confidence=config.get("MIN_TRACKING_CONFIDENCE", 0.5),
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

        # ── Camera + thread state ─────────────────────────────────────────────
        self._cap = None
        self._running = False
        self._thread = None
        self._latest_frame = None   # Latest JPEG bytes (shared between threads)
        self._frame_lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Open the webcam and start the background processing thread."""
        logger.info("Opening camera...")
        self._cap = cv2.VideoCapture(self._config.get("CAMERA_INDEX", 0))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.get("CAMERA_WIDTH", 1280))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.get("CAMERA_HEIGHT", 720))

        if not self._cap.isOpened():
            logger.error("Failed to open camera — check CAMERA_INDEX in config.py")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="camera-thread"
        )
        self._thread.start()
        logger.info("Camera thread started.")

    def stop(self):
        """Gracefully stop the capture thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Camera released.")

    # ── Background capture loop ────────────────────────────────────────────────

    def _camera_loop(self):
        """
        Runs in a daemon thread. Continuously:
          1. Reads a frame from the webcam
          2. Mirrors it (cv2.flip)
          3. Runs MediaPipe detection
          4. Draws landmarks onto the frame
          5. Calls recognizer.process() with normalized landmarks
          6. JPEG-encodes the frame and stores it in _latest_frame
        """
        jpeg_quality = self._config.get("JPEG_QUALITY", 85)

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Mirror flip — same as original cv2.flip(raw_frame, 1)
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run MediaPipe hand detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._detector.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw skeleton connections (white lines — same as original)
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand[start].x * fw)
                    y1 = int(hand[start].y * fh)
                    x2 = int(hand[end].x * fw)
                    y2 = int(hand[end].y * fh)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Draw landmark dots (red circles — same as original)
                for lm in hand:
                    cx, cy = int(lm.x * fw), int(lm.y * fh)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # Normalize landmarks and run gesture recognition
                landmarks = normalize_landmarks(hand)
                self._recognizer.process(landmarks)
            else:
                # No hand in frame — reset detection state
                self._recognizer.clear_detection()

            # JPEG-encode the frame for the MJPEG stream
            ret_enc, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ret_enc:
                with self._frame_lock:
                    self._latest_frame = buffer.tobytes()

    # ── MJPEG stream generator ─────────────────────────────────────────────────

    def generate_frames(self):
        """
        Generator that yields MJPEG boundary-delimited JPEG frames.
        Used by the /video_feed route as a Flask streaming Response.
        """
        while True:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

            time.sleep(0.033)   # ~30 FPS cap
