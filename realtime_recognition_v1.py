# realtime_recognition.py - MediaPipe 0.10+ version
import cv2
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from collections import deque, Counter

# Load model and encoder
model = load_model('sign_language_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Setup MediaPipe Hand Landmarker
model_path = 'hand_landmarker.task'
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17)           # Palm
]

prediction_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)
print("Webcam started! Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    predicted_label = ""
    confidence = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw skeleton connections
        for start, end in HAND_CONNECTIONS:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Draw landmark dots
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Extract features (same as training)
        landmarks = []
        x_coords = [lm.x for lm in hand]
        y_coords = [lm.y for lm in hand]
        for lm in hand:
            landmarks.append(lm.x - min(x_coords))
            landmarks.append(lm.y - min(y_coords))

        # Predict
        input_data = np.array([landmarks])
        prediction = model.predict(input_data, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]

        if confidence > 0.75:
            predicted_label = le.classes_[class_idx]
            prediction_buffer.append(predicted_label)

    # Stable prediction
    stable_label = Counter(prediction_buffer).most_common(1)[0][0] if prediction_buffer else ""

    # ---- Draw UI ----
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)

    if stable_label:
        cv2.putText(frame, f"Sign: {stable_label}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 100), 3)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "Show your hand...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (80, 80, 80), 2)

    cv2.putText(frame, "Sign Language Recognition | Press Q to quit",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()