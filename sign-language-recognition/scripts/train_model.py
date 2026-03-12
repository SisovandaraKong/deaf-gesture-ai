# scripts/train_model.py
# Run from project root: python scripts/train_model.py
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Resolve paths relative to the project root (one level up from scripts/)
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKS_PKL   = os.path.join(BASE_DIR, 'models', 'landmarks_data.pkl')
ENCODER_PKL     = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'sign_language_model.h5')

# Load data
with open(LANDMARKS_PKL, 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])

# Encode labels (A, B, C... 0, 1, 2...)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Save label encoder
with open(ENCODER_PKL, 'wb') as f:
    pickle.dump(le, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_onehot, test_size=0.2, random_state=42, stratify=labels_encoded
)

num_classes = len(le.classes_)
print(f"Training with {num_classes} classes: {list(le.classes_)}")

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(data.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(patience=15, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Plot accuracy/loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'training_results.png'))
plt.show()
