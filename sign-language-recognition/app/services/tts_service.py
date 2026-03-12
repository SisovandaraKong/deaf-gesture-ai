"""
app/services/tts_service.py

Generates text-to-speech audio using gTTS (identical call to original):
    gTTS(text=text, lang=lang, slow=False)

In the original desktop app, pygame played the audio locally.
In this Flask web app the MP3 bytes are returned to the browser
and played via the HTML5 Audio API — the gTTS logic is unchanged.
"""
import os
import logging
import tempfile
from gtts import gTTS

logger = logging.getLogger(__name__)


class TTSService:
    def generate(self, text: str, lang: str = "km") -> bytes:
        """
        Generate speech from text using gTTS and return raw MP3 bytes.

        Args:
            text: Text to speak
            lang: BCP-47 language code ('km' for Khmer, 'en' for English)

        Returns:
            MP3 audio bytes ready to stream to the browser.

        Raises:
            Exception: Re-raised so the route can return a proper 500 error.
        """
        # Identical gTTS call from realtime_recognition.py
        tts = gTTS(text=text, lang=lang, slow=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name

        try:
            tts.save(tmp_path)
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            logger.info(f"Generated TTS | lang={lang} | text={repr(text)[:60]}")
            return audio_bytes
        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
