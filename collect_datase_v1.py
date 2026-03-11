# collect_dataset.py - Capture your own hand images via webcam
import cv2
import os
import time

DATA_DIR = './dataset_new'
CLASSES = ['0','1','2','3','4','5','6','7','8','9',
           'A','B','C','D','E','F','G','H','I','J','K',
           'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
IMAGES_PER_CLASS = 200  # capture 100 images per gesture

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for class_name in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f'\n📷 Ready to capture: "{class_name}"')
    print('Show your hand gesture and press SPACE to start capturing...')

    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Get ready for: {class_name}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, 'Press SPACE to start', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow('Collecting Data', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Capture images
    count = 0
    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f'Capturing "{class_name}": {count}/{IMAGES_PER_CLASS}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('Collecting Data', frame)

        img_path = os.path.join(class_dir, f'{class_name}_{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1

        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    print(f'✅ Captured {count} images for "{class_name}"')

cap.release()
cv2.destroyAllWindows()
print('\n🎉 Dataset collection complete!')
print(f'New dataset saved in: {DATA_DIR}')