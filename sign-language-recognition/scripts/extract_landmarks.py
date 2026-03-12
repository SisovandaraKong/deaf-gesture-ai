# scripts/extract_landmarks.py - Fixed with border crop for MediaPipe 0.10.32
# Run from project root: python scripts/extract_landmarks.py
import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Resolve paths relative to the project root (one level up from scripts/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'dataset')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')
OUTPUT_PKL = os.path.join(BASE_DIR, 'models', 'landmarks_data.pkl')

# Setup MediaPipe
base_options = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)


def remove_pink_border(img):
    """Crop out the pink/magenta border from dataset images"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    non_pink = cv2.bitwise_not(mask)
    coords = cv2.findNonZero(non_pink)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        x = max(0, x + pad)
        y = max(0, y + pad)
        w = min(img.shape[1] - x, w - pad * 2)
        h = min(img.shape[0] - y, h - pad * 2)
        return img[y:y + h, x:x + w]
    return img


data = []
labels = []
total_success = 0
total_fail = 0

for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    img_files = os.listdir(folder_path)
    folder_success = 0

    for img_file in img_files:
        img_path = os.path.join(folder_path, img_file)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Remove pink border then resize
        img_bgr = remove_pink_border(img_bgr)
        img_bgr = cv2.resize(img_bgr, (224, 224))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            landmarks = []
            x_coords = [lm.x for lm in hand]
            y_coords = [lm.y for lm in hand]
            for lm in hand:
                landmarks.append(lm.x - min(x_coords))
                landmarks.append(lm.y - min(y_coords))
            data.append(landmarks)
            labels.append(folder)
            folder_success += 1
            total_success += 1
        else:
            total_fail += 1

    print(f"Folder '{folder}': {folder_success}/{len(img_files)} detected")

with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Total extracted: {total_success} samples from {len(set(labels))} classes")
print(f"❌ Failed to detect: {total_fail} images")
print(f"💾 Saved to: {OUTPUT_PKL}")
print("Classes found:", sorted(set(labels)))
