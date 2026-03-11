# realtime_recognition.py - Full App UI with resizable window
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

translator    = Translator()
pygame.mixer.init()

# ── Fonts ─────────────────────────────────────────────────────────────────────
khmer_font        = ImageFont.truetype("KhmerOS.ttf", 32)
khmer_font_large  = ImageFont.truetype("KhmerOS.ttf", 42)
english_font      = ImageFont.truetype("KhmerOS.ttf", 30)
english_font_large= ImageFont.truetype("KhmerOS.ttf", 52)
small_font        = ImageFont.truetype("KhmerOS.ttf", 22)

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
is_fullscreen     = False

# ── Window Setup ──────────────────────────────────────────────────────────────
WINDOW_NAME = 'Sign Language Recognition'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)  # default window size

# ── Helper Functions ──────────────────────────────────────────────────────────
def put_unicode_text(frame, text, position, font, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=15, thickness=-1):
    """Draw a rounded rectangle"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, thickness)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, thickness)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, thickness)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, thickness)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame

def speak_text(text, lang='km'):
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
    try:
        result = translator.translate(text, src='en', dest='km')
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def draw_ui(frame, cam_frame, stable_label, confidence):
    """Draw full app UI on a canvas"""
    global status_message

    H, W = frame.shape[:2]
    ch, cw = cam_frame.shape[:2]

    # ── Layout calculations ───────────────────────────────────────────────────
    PADDING      = 20
    SIDEBAR_W    = max(340, W // 4)
    CAM_X        = PADDING
    CAM_Y        = 70
    CAM_W        = W - SIDEBAR_W - PADDING * 3
    CAM_H        = H - CAM_Y - PADDING

    # ── Background ────────────────────────────────────────────────────────────
    frame[:] = (18, 18, 28)  # dark navy background

    # Subtle grid pattern
    for i in range(0, W, 40):
        cv2.line(frame, (i, 0), (i, H), (25, 25, 40), 1)
    for i in range(0, H, 40):
        cv2.line(frame, (0, i), (W, i), (25, 25, 40), 1)

    # ── Top bar ───────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (W, 62), (12, 12, 22), -1)
    cv2.line(frame, (0, 62), (W, 62), (0, 200, 120), 2)

    # App title
    frame = put_unicode_text(frame, "Sign Language Recognition",
                             (PADDING, 12), english_font, (0, 220, 130))

    # Fullscreen hint
    hint = "F: Fullscreen | Q: Quit"
    frame = put_unicode_text(frame, hint, (W - 280, 18), small_font, (100, 100, 140))

    # ── Camera feed ───────────────────────────────────────────────────────────
    cam_resized = cv2.resize(cam_frame, (CAM_W, CAM_H))
    frame[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cam_resized

    # Camera border glow
    color = (0, 255, 120) if stable_label else (60, 60, 100)
    cv2.rectangle(frame, (CAM_X-2, CAM_Y-2),
                  (CAM_X+CAM_W+2, CAM_Y+CAM_H+2), color, 2)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    SB_X = W - SIDEBAR_W - PADDING
    SB_Y = CAM_Y

    # --- Detected Sign Card ---
    card_h = 130
    cv2.rectangle(frame, (SB_X, SB_Y), (SB_X+SIDEBAR_W, SB_Y+card_h), (28, 28, 45), -1)
    cv2.rectangle(frame, (SB_X, SB_Y), (SB_X+SIDEBAR_W, SB_Y+card_h), (0,180,100), 2)
    frame = put_unicode_text(frame, "DETECTED", (SB_X+12, SB_Y+8), small_font, (100,100,140))

    sign_text = stable_label if stable_label else "..."
    frame = put_unicode_text(frame, sign_text, (SB_X+12, SB_Y+35),
                             english_font_large, (0,255,130))

    # Confidence bar
    bar_x, bar_y = SB_X+12, SB_Y+100
    bar_w = SIDEBAR_W - 24
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+14), (40,40,60), -1)
    fill = int(bar_w * confidence)
    bar_color = (0,220,100) if confidence > 0.8 else (220,160,0) if confidence > 0.5 else (200,60,60)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+14), bar_color, -1)
    frame = put_unicode_text(frame, f"{confidence:.0%}", (bar_x+bar_w+8, bar_y-4),
                             small_font, (180,180,180))

    # Hold progress circle
    if current_sign and sign_hold_start:
        held  = time.time() - sign_hold_start
        angle = int(min(held / HOLD_SECONDS, 1.0) * 360)
        cx_c  = SB_X + SIDEBAR_W - 30
        cy_c  = SB_Y + 55
        cv2.circle(frame, (cx_c, cy_c), 22, (40,40,60), -1)
        cv2.ellipse(frame, (cx_c, cy_c), (22,22), -90, 0, angle, (0,255,130), 3)

    # --- English Sentence Card ---
    en_y = SB_Y + card_h + 14
    en_h = 90
    cv2.rectangle(frame, (SB_X, en_y), (SB_X+SIDEBAR_W, en_y+en_h), (28,28,45), -1)
    cv2.rectangle(frame, (SB_X, en_y), (SB_X+SIDEBAR_W, en_y+en_h), (60,100,200), 2)
    frame = put_unicode_text(frame, "ENGLISH", (SB_X+12, en_y+6), small_font, (100,100,180))
    en_text = " ".join(sentence) if sentence else "(empty)"
    # Truncate if too long
    if len(en_text) > 22:
        en_text = en_text[-22:]
    frame = put_unicode_text(frame, en_text, (SB_X+12, en_y+32), english_font, (220,220,255))

    # --- Khmer Translation Card ---
    kh_y = en_y + en_h + 14
    kh_h = 90
    cv2.rectangle(frame, (SB_X, kh_y), (SB_X+SIDEBAR_W, kh_y+kh_h), (28,28,45), -1)
    cv2.rectangle(frame, (SB_X, kh_y), (SB_X+SIDEBAR_W, kh_y+kh_h), (200,100,60), 2)
    frame = put_unicode_text(frame, "KHMER", (SB_X+12, kh_y+6), small_font, (180,120,80))
    kh_text = khmer_translation if khmer_translation else "(will appear after sign)"
    if len(kh_text) > 18:
        kh_text = kh_text[-18:]
    frame = put_unicode_text(frame, kh_text, (SB_X+12, kh_y+32), khmer_font, (255,180,100))

    # --- Status Card ---
    st_y = kh_y + kh_h + 14
    st_h = 55
    cv2.rectangle(frame, (SB_X, st_y), (SB_X+SIDEBAR_W, st_y+st_h), (28,28,45), -1)
    frame = put_unicode_text(frame, status_message, (SB_X+12, st_y+12),
                             small_font, (255,200,60))

    # --- Control Buttons ---
    btn_y  = st_y + st_h + 14
    btn_h  = 48
    btn_gap = 10
    btn_w  = (SIDEBAR_W - btn_gap) // 2

    buttons = [
        ("SPACE: Khmer", (0,160,80)),
        ("E: English",   (60,100,200)),
        ("C: Clear",     (180,60,60)),
        ("F: Fullscreen",(80,80,120)),
    ]

    for i, (label, color) in enumerate(buttons):
        row = i // 2
        col = i % 2
        bx  = SB_X + col * (btn_w + btn_gap)
        by  = btn_y + row * (btn_h + btn_gap)
        cv2.rectangle(frame, (bx, by), (bx+btn_w, by+btn_h), color, -1)
        cv2.rectangle(frame, (bx, by), (bx+btn_w, by+btn_h), (255,255,255,30), 1)
        frame = put_unicode_text(frame, label, (bx+8, by+12), small_font, (255,255,255))

    return frame

# ── Main Loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Started! F=fullscreen | SPACE=speak Khmer | E=speak English | C=clear | Q=quit")

while cap.isOpened():
    ret, raw_frame = cap.read()
    if not ret:
        break

    raw_frame = cv2.flip(raw_frame, 1)
    fh, fw    = raw_frame.shape[:2]
    frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

    # ── Detection ─────────────────────────────────────────────────────────────
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = detector.detect(mp_image)

    stable_label = ""
    confidence   = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        for start, end in HAND_CONNECTIONS:
            x1,y1 = int(hand[start].x*fw), int(hand[start].y*fh)
            x2,y2 = int(hand[end].x*fw),   int(hand[end].y*fh)
            cv2.line(raw_frame, (x1,y1), (x2,y2), (255,255,255), 2)
        for lm in hand:
            cx,cy = int(lm.x*fw), int(lm.y*fh)
            cv2.circle(raw_frame, (cx,cy), 6, (0,0,255), -1)

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

        if stable_label:
            if stable_label == current_sign:
                held = time.time() - sign_hold_start
                if held >= HOLD_SECONDS and stable_label != last_added_sign:
                    sentence.append(stable_label)
                    last_added_sign   = stable_label
                    sign_hold_start   = time.time()
                    khmer_translation = translate_to_khmer(" ".join(sentence))
                    status_message    = f'Added: "{stable_label}"'
            else:
                current_sign    = stable_label
                sign_hold_start = time.time()
                last_added_sign = ""
    else:
        prediction_buffer.clear()
        current_sign    = ""
        sign_hold_start = None

    # ── Get current window size ───────────────────────────────────────────────
    try:
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        WIN_W = max(rect[2], 800)
        WIN_H = max(rect[3], 500)
    except:
        WIN_W, WIN_H = 1280, 720

    # ── Draw full UI canvas ───────────────────────────────────────────────────
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas = draw_ui(canvas, raw_frame, stable_label, confidence)

    cv2.imshow(WINDOW_NAME, canvas)

    # ── Keys ──────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    elif key == ord('c'):
        sentence.clear()
        khmer_translation = ""
        status_message    = "Cleared!"

    elif key == ord(' ') and sentence and not is_speaking:
        text = khmer_translation if khmer_translation else " ".join(sentence)
        threading.Thread(target=speak_text, args=(text, 'km'), daemon=True).start()

    elif key == ord('e') and sentence and not is_speaking:
        threading.Thread(target=speak_text,
                         args=(" ".join(sentence), 'en'), daemon=True).start()

cap.release()
cv2.destroyAllWindows()