# collect_dataset.py - Space to Start/Pause
import cv2
import os

DATA_DIR = './dataset_new'
CLASSES = ['hello','yes','no','thank you','sorry','please',
           'help','good','bad','eat','drink','sleep',
           'I','you','we','come','go']

IMAGES_PER_CLASS = 200

os.makedirs(DATA_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

for class_name in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Skip if already has enough images
    existing = len(os.listdir(class_dir))
    if existing >= IMAGES_PER_CLASS:
        print(f'✅ Skipping "{class_name}" - already has {existing} images')
        continue

    print(f'\n📷 Class: "{class_name}" ({existing} images already)')
    print('Press SPACE to start/pause | Press Q to quit | Press N to skip to next class')

    count = existing
    capturing = False

    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Status bar
        status_color = (0, 255, 0) if capturing else (0, 165, 255)
        status_text = "● CAPTURING" if capturing else "❚❚ PAUSED"
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, f'Class: "{class_name}"', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'{status_text}  {count}/{IMAGES_PER_CLASS}', (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # Progress bar
        progress = int((count / IMAGES_PER_CLASS) * w)
        cv2.rectangle(frame, (0, 80), (progress, 90), (0, 255, 100), -1)
        cv2.rectangle(frame, (0, 80), (w, 90), (60, 60, 60), 1)

        # Bottom instructions
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, 'SPACE: start/pause  |  N: next class  |  Q: quit', (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow('Collecting Dataset', frame)

        key = cv2.waitKey(200) & 0xFF  # 200ms between captures

        if key == ord('q'):
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('n'):
            print(f'⏭ Skipping to next class (saved {count} images)')
            break
        elif key == ord(' '):
            capturing = not capturing
            print("▶ Capturing..." if capturing else "⏸ Paused")

        # Save image only when capturing
        if capturing:
            img_path = os.path.join(class_dir, f'{class_name}_{count}.jpg')
            cv2.imwrite(img_path, frame)
            count += 1

    print(f'✅ Done "{class_name}": {count} images saved')

cap.release()
cv2.destroyAllWindows()
print('\n🎉 Dataset collection complete!')