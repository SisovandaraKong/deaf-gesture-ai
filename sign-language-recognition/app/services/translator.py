"""
app/services/translator.py

Wraps googletrans for English → Khmer translation.
Logic is preserved exactly from realtime_recognition.py:
    result = translator.translate(text, src='en', dest='km')
    return result.text
"""
import logging
from googletrans import Translator

logger = logging.getLogger(__name__)


class TranslatorService:
    def __init__(self):
        # Reuse a single Translator instance (stateless between calls)
        self._translator = Translator()

    def translate(self, text: str, src: str = "en", dest: str = "km") -> str:
        """
        Translate text from src language to dest language.

        Returns the translated string, or empty string on any error.
        Preserves exact googletrans call from realtime_recognition.py.
        """
        try:
            result = self._translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as exc:
            logger.error(f"Translation error: {exc}")
            return ""
