# realtime_recognition.py - With Sentence Builder + TTS + Khmer Translation
import cv2
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from collections import deque, Counter
from googletrans import Translator
from gtts import gTTS
from PIL import ImageFont, ImageDraw, Image
import pygame
import tempfile
import os
import threading
import time

# ── Init ──────────────────────────────────────────────────────────────────────
model = load_model('sign_language_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

translator = Translator()
pygame.mixer.init()

# ── Load Khmer Fonts ──────────────────────────────────────────────────────────
khmer_font   = ImageFont.truetype("KhmerOS.ttf", 28)
english_font = ImageFont.truetype("KhmerOS.ttf", 26)
small_font   = ImageFont.truetype("KhmerOS.ttf", 20)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── State ─────────────────────────────────────────────────────────────────────
prediction_buffer = deque(maxlen=10)
sentence          = []
current_sign      = ""
last_added_sign   = ""
sign_hold_start   = None
HOLD_SECONDS      = 1.5
khmer_translation = ""
is_speaking       = False
status_message    = "Show your hand to start"

# ── Helper Functions ──────────────────────────────────────────────────────────
def put_unicode_text(frame, text, position, font, color=(255, 255, 255)):
    """Draw Unicode/Khmer text on OpenCV frame using PIL"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def speak_text(text, lang='km'):
    """Speak text using gTTS"""
    global is_speaking, status_message
    is_speaking    = True
    status_message = "Speaking..."
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tmp_path = f.name
        tts.save(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        os.unlink(tmp_path)
    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        is_speaking    = False
        status_message = "Show your hand to start"

def translate_to_khmer(text):
    """Translate English text to Khmer"""
    try:
        result = translator.translate(text, src='en', dest='km')
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Started! Controls: SPACE=speak Khmer | E=speak English | C=clear | Q=quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    h, w, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Detection ─────────────────────────────────────────────────────────────
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = detector.detect(mp_image)

    stable_label = ""
    confidence   = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw skeleton
        for start, end in HAND_CONNECTIONS:
            x1,y1 = int(hand[start].x*w), int(hand[start].y*h)
            x2,y2 = int(hand[end].x*w),   int(hand[end].y*h)
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 2)
        for lm in hand:
            cx,cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)

        # Extract features
        landmarks = []
        x_coords  = [lm.x for lm in hand]
        y_coords  = [lm.y for lm in hand]
        for lm in hand:
            landmarks.append(lm.x - min(x_coords))
            landmarks.append(lm.y - min(y_coords))

        prediction = model.predict(np.array([landmarks]), verbose=0)[0]
        class_idx  = np.argmax(prediction)
        confidence = prediction[class_idx]

        if confidence > 0.75:
            prediction_buffer.append(le.classes_[class_idx])

        if prediction_buffer:
            stable_label = Counter(prediction_buffer).most_common(1)[0][0]

        # ── Hold to confirm ───────────────────────────────────────────────────
        if stable_label:
            if stable_label == current_sign:
                held           = time.time() - sign_hold_start
                progress_angle = int((held / HOLD_SECONDS) * 360)
                cv2.ellipse(frame, (w-60, 60), (40,40), -90,
                            0, progress_angle, (0,255,100), 4)

                if held >= HOLD_SECONDS and stable_label != last_added_sign:
                    sentence.append(stable_label)
                    last_added_sign   = stable_label
                    sign_hold_start   = time.time()
                    full_text         = " ".join(sentence)
                    khmer_translation = translate_to_khmer(full_text)
                    status_message    = f'Added: "{stable_label}"'
            else:
                current_sign    = stable_label
                sign_hold_start = time.time()
                last_added_sign = ""
    else:
        prediction_buffer.clear()
        current_sign    = ""
        sign_hold_start = None

    # ── Draw UI ───────────────────────────────────────────────────────────────
    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 100), (15,15,15), -1)

    # Current sign (use cv2 for large Latin text - fast)
    sign_display = stable_label if stable_label else "..."
    cv2.putText(frame, f"Sign: {sign_display}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,100), 3)
    cv2.putText(frame, f"{confidence:.0%}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 2)

    # Status message (PIL for Unicode emoji support)
    frame = put_unicode_text(frame, status_message, (w-380, 15), small_font, (255,200,0))

    # English sentence box
    cv2.rectangle(frame, (0, h-185), (w, h-125), (30,30,30), -1)
    en_text = "EN: " + " ".join(sentence) if sentence else "EN: (empty)"
    frame = put_unicode_text(frame, en_text, (15, h-180), english_font, (255,255,255))

    # Khmer translation box
    cv2.rectangle(frame, (0, h-125), (w, h-65), (20,20,40), -1)
    kh_text = "KH: " + khmer_translation if khmer_translation else "KH: (will appear after sign)"
    frame = put_unicode_text(frame, kh_text, (15, h-120), khmer_font, (100,200,255))

    # Controls bar
    cv2.rectangle(frame, (0, h-65), (w, h), (15,15,15), -1)
    frame = put_unicode_text(frame, "SPACE: Speak Khmer | E: Speak English | C: Clear | Q: Quit",
                             (10, h-58), small_font, (180,180,180))

    cv2.imshow('Sign Language Recognition', frame)

    # ── Keys ──────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        sentence.clear()
        khmer_translation = ""
        status_message    = "Cleared!"
        print("Sentence cleared")

    elif key == ord(' ') and sentence and not is_speaking:
        text = khmer_translation if khmer_translation else " ".join(sentence)
        print(f"Speaking Khmer: {text}")
        threading.Thread(target=speak_text, args=(text, 'km'), daemon=True).start()

    elif key == ord('e') and sentence and not is_speaking:
        text = " ".join(sentence)
        print(f"Speaking English: {text}")
        threading.Thread(target=speak_text, args=(text, 'en'), daemon=True).start()

cap.release()
cv2.destroyAllWindows()