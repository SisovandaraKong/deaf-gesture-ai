"""
app/services/gesture_recognizer.py

Wraps the TensorFlow model inference, prediction smoothing buffer,
hold-timer logic, and sentence accumulation.

All AI/ML logic is preserved exactly from realtime_recognition.py:
  - deque(maxlen=10) prediction buffer
  - confidence threshold 0.75
  - Counter().most_common(1) for stable label
  - HOLD_SECONDS = 1.5 hold-before-confirm timer
"""
import time
import threading
import logging
import numpy as np
from collections import deque, Counter

logger = logging.getLogger(__name__)


class GestureRecognizer:
    def __init__(self, model, label_encoder, config, translator=None):
        self.model = model
        self.le = label_encoder
        self.translator = translator  # Optional: TranslatorService instance

        # Config thresholds (from config.py, not hardcoded)
        self.hold_seconds = config.get("HOLD_SECONDS", 1.5)
        self.confidence_threshold = config.get("CONFIDENCE_THRESHOLD", 0.75)
        buffer_size = config.get("PREDICTION_BUFFER_SIZE", 10)

        # Prediction smoothing buffer (identical to original deque(maxlen=10))
        self.prediction_buffer = deque(maxlen=buffer_size)

        # ── Shared state (protected by _lock) ─────────────────────────────────
        self._lock = threading.Lock()
        self.current_sign = ""        # Sign currently being held
        self.stable_label = ""        # Most-voted sign from buffer
        self.confidence = 0.0         # Latest raw model confidence
        self.last_added_sign = ""     # Prevents double-adding same sign
        self.sign_hold_start = None   # Timestamp when current_sign first seen
        self.sentence = []            # Confirmed signs list
        self.khmer_translation = ""   # Latest Khmer translation of sentence
        self.status_message = "Show your hand to start"

    # ── Core prediction logic ──────────────────────────────────────────────────

    def process(self, landmarks: list):
        """
        Run TensorFlow model on normalized landmarks, update buffer,
        and handle the hold-timer / sentence-building logic.

        Called from the camera background thread on every detected frame.

        Args:
            landmarks: 42-float list from landmark_utils.normalize_landmarks()
        """
        # ── Model inference (unchanged from original) ──────────────────────
        prediction = self.model.predict(np.array([landmarks]), verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])

        # Only buffer predictions above confidence threshold
        if confidence > self.confidence_threshold:
            self.prediction_buffer.append(self.le.classes_[class_idx])

        # Stable label = majority vote over the buffer window
        stable_label = ""
        if self.prediction_buffer:
            stable_label = Counter(self.prediction_buffer).most_common(1)[0][0]

        # ── State update (lock protects sentence + hold timer) ─────────────
        trigger_translation = False
        sentence_snapshot = None

        with self._lock:
            self.confidence = confidence
            self.stable_label = stable_label

            if stable_label:
                if stable_label == self.current_sign:
                    # Same sign — check if held long enough to confirm
                    held = time.time() - self.sign_hold_start
                    if (held >= self.hold_seconds
                            and stable_label != self.last_added_sign):
                        self.sentence.append(stable_label)
                        self.last_added_sign = stable_label
                        self.sign_hold_start = time.time()
                        self.status_message = f'Added: "{stable_label}"'
                        trigger_translation = True
                        sentence_snapshot = " ".join(self.sentence)
                        logger.info(f"Sign confirmed: {stable_label} | sentence: {self.sentence}")
                else:
                    # New sign detected — start hold timer
                    self.current_sign = stable_label
                    self.sign_hold_start = time.time()
                    self.last_added_sign = ""
                    self.status_message = f'Holding: "{stable_label}"'

        # ── Async translation (outside lock to avoid blocking camera thread) ─
        if trigger_translation and sentence_snapshot and self.translator:
            threading.Thread(
                target=self._async_translate,
                args=(sentence_snapshot,),
                daemon=True,
            ).start()

    def clear_detection(self):
        """Reset detection state when no hand is visible in the frame."""
        self.prediction_buffer.clear()
        with self._lock:
            self.current_sign = ""
            self.stable_label = ""
            self.sign_hold_start = None
            self.status_message = "Show your hand to start"

    # ── Sentence management ────────────────────────────────────────────────────

    def add_sign(self, sign: str):
        """Manually add a sign to the sentence (called from /api/sentence/add)."""
        with self._lock:
            self.sentence.append(sign)
        logger.info(f"Sign manually added: {sign}")

    def clear_sentence(self):
        """Clear the accumulated sentence and reset translation."""
        with self._lock:
            self.sentence.clear()
            self.khmer_translation = ""
            self.status_message = "Cleared!"
        logger.info("Sentence cleared.")

    # ── State getters ──────────────────────────────────────────────────────────

    def _hold_progress_unsafe(self) -> float:
        """Compute hold progress [0.0–1.0]. Must be called with _lock held."""
        if self.current_sign and self.sign_hold_start:
            held = time.time() - self.sign_hold_start
            return min(held / self.hold_seconds, 1.0)
        return 0.0

    def get_hold_progress(self) -> float:
        """Thread-safe hold progress getter."""
        with self._lock:
            return self._hold_progress_unsafe()

    def get_status(self) -> dict:
        """Return real-time recognition status for /api/status."""
        with self._lock:
            return {
                "current_sign": self.stable_label,
                "confidence": round(self.confidence, 4),
                "hold_progress": round(self._hold_progress_unsafe(), 4),
                "status_message": self.status_message,
                "is_speaking": False,   # TTS is handled client-side (browser)
            }

    def get_sentence_state(self) -> dict:
        """Return sentence + translation for /api/sentence."""
        with self._lock:
            return {
                "sentence": list(self.sentence),
                "khmer_translation": self.khmer_translation,
            }

    def get_full_state(self) -> dict:
        """Return combined state for /api/predict."""
        with self._lock:
            return {
                "current_sign": self.stable_label,
                "confidence": round(self.confidence, 4),
                "hold_progress": round(self._hold_progress_unsafe(), 4),
                "sentence": list(self.sentence),
                "khmer_translation": self.khmer_translation,
                "status_message": self.status_message,
            }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _async_translate(self, text: str):
        """Translate the sentence to Khmer in a daemon thread."""
        try:
            khmer = self.translator.translate(text)
            with self._lock:
                self.khmer_translation = khmer
            logger.info(f"Translation: '{text}' → '{khmer}'")
        except Exception as exc:
            logger.error(f"Async translation failed: {exc}")
