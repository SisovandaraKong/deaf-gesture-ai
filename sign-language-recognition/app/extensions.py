"""
app/extensions.py

Module-level service registry.
Services are created inside create_app() and stored here so that
route handlers can import them without circular imports.
"""

# Set to their real instances during create_app()
recognizer = None   # GestureRecognizer  — prediction state + sentence
detector = None     # HandDetectorService — camera thread + MediaPipe
translator = None   # TranslatorService   — Google Translate (EN→KM)
tts = None          # TTSService          — gTTS audio generation
