<div align="center">

# 🤟 Deaf Gesture AI — Sign Language Recognition System

### ប្រព័ន្ធទទួលស្គាល់ភាសាសញ្ញាដៃ សម្រាប់សហគមន៍ (ពិការ) នៅកម្ពុជា
*A real-time AI system that translates hand gestures into text and speech — bridging communication between the Deaf community and the hearing world.*

---

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00897B?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![University](https://img.shields.io/badge/Final%20Year%20Project-Cambodia%202026-red?style=for-the-badge)

</div>

---

## ការពិពណ៌នាខ្លី (Khmer Short Description)

> **ប្រព័ន្ធ Deaf Gesture AI** គឺជាកម្មវិធីដែលប្រើបញ្ញាសិប្បនិម្មិត (AI) ដើម្បីទទួលស្គាល់ **ភាសាសញ្ញាដៃ** ក្នុងពេលវេលាជាក់ស្តែង (Real-time)។ ប្រព័ន្ធនេះប្រើ **MediaPipe** ដើម្បីទំនាញចំណុចកំណត់ (Landmarks) នៃដៃ ហើយប្រើ **Neural Network** ដើម្បីស្គាល់ប្រភេទសញ្ញា។ អត្ថបទដែលស្គាល់បានត្រូវបានបកប្រែជា **ភាសាខ្មែរ** ហើយអានចេញជាសំឡេង ដើម្បីជួយផ្លាស់ប្តូរការទំនាក់ទំនងរវាងសហគមន៍ **ពិការ** និងអ្នកជុំវិញ។

---

## 📖 About The Project

**Deaf Gesture AI** is a real-time sign language recognition system developed as a **Final Year Project** at a university in Cambodia. The system uses a webcam to detect and interpret hand gestures, build a sentence word by word, translate it into Khmer, and read it aloud — enabling deaf and hard-of-hearing individuals to communicate more effectively.

The pipeline is straightforward:

```
📷 Webcam Input → 🖐 Hand Detection (MediaPipe) → 🧠 Neural Network → 📝 Sentence Builder → 🌐 Khmer Translation → 🔊 Text-to-Speech
```

### 🎯 Who It Helps

- **Deaf and hard-of-hearing individuals** in Cambodia
- **Teachers and students** in schools for the Deaf
- **Families** communicating with their Deaf members
- **Public service workers** who interact with the Deaf community

---

## 📸 Screenshots

> *Add screenshots of your application here*

| Dataset Collection | Real-time Recognition UI |
|---|---|
| ![collect](screenshots/collect_dataset.png) | ![recognition](screenshots/realtime_ui.png) |

| Hand Landmark Detection | Training Results |
|---|---|
| ![landmarks](screenshots/landmarks.png) | ![training](screenshots/training_results.png) |

---

## ✨ Features

- 🖐 **Real-time Hand Detection** — Uses MediaPipe Hand Landmarker to detect 21 hand keypoints at 30+ FPS
- 🧠 **Deep Learning Classification** — Feed-forward neural network trained on normalized landmark coordinates
- 📝 **Sentence Builder** — Accumulates recognized signs into a full sentence with a hold-to-confirm mechanism (1.5 seconds)
- 🌐 **Khmer Translation** — Automatically translates the English sentence into Khmer using Google Translate
- 🔊 **Text-to-Speech** — Speaks the sentence aloud in **Khmer** (`SPACE`) or **English** (`E`) using gTTS
- 📊 **Confidence Meter** — Live confidence bar showing prediction certainty (green/yellow/red)
- ⏱ **Prediction Smoothing** — Sliding window buffer (10 frames) reduces flickering predictions
- 🎨 **Professional UI** — Dark-themed resizable window with sidebar, Khmer Unicode font rendering, hand skeleton overlay
- 🖥 **Fullscreen Mode** — Toggle fullscreen with `F` key
- 🔤 **Supports 43+ Classes** — ASL static alphabet (A–Z), digits (0–9), and common words (hello, yes, no, thank you, sorry, please, help, eat, drink, sleep, I, you, we, come, go, good, bad)
- 📸 **Smart Dataset Collector** — Webcam-based collector with pause/resume, skip-class, and auto-skip if class is already full
- 🩷 **Pre-processing Pipeline** — Automatic pink border removal from dataset images before landmark extraction

---

## 🗂 Project Structure

```
deaf-gesture-ai/
│
├── 📄 collect_dataset.py           # Step 1 — Webcam-based dataset image collector
├── 📄 extract_landmarks.py         # Step 2 — Extract MediaPipe hand landmarks from images
├── 📄 train_model.py               # Step 3 — Train the neural network classifier
├── 📄 realtime_recognition.py      # Step 4 — Full real-time recognition app with UI
│
├── 📄 collect_datase_v1.py         # (Legacy) Earlier version of dataset collector
├── 📄 realtime_recognition_v1.py   # (Legacy) Earlier version of recognition
├── 📄 realtime_recognition_v2_show_sentence.py  # (Legacy) Sentence-display version
│
├── 🤖 hand_landmarker.task         # MediaPipe Hand Landmarker model file (required)
├── 🤖 sign_language_model.h5       # Trained Keras model (generated after training)
├── 📦 landmarks_data.pkl           # Extracted landmark dataset (generated after Step 2)
├── 📦 label_encoder.pkl            # Sklearn LabelEncoder (generated after training)
├── 🖼 training_results.png         # Accuracy/loss plot (generated after training)
├── 🔤 KhmerOS.ttf                  # Khmer Unicode font file (required)
│
├── 📁 dataset/                     # Static gesture dataset (A-Z, 0-9)
│   ├── A/                          # ~200 images per letter class
│   ├── B/
│   └── ... (26 letters + 10 digits)
│
└── 📁 dataset_new/                 # Word/phrase gesture dataset
    ├── hello/
    ├── yes/
    ├── no/
    ├── thank you/
    └── ... (custom word classes)
```

---

## ⚙️ Requirements

### Python Version
```
Python 3.8 — 3.11 (recommended: 3.10)
```

> ⚠️ **Python 3.12+ is not recommended** due to compatibility issues with some TensorFlow versions.

### Python Packages

```txt
opencv-python>=4.8.0
mediapipe>=0.10.0
tensorflow>=2.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
Pillow>=10.0.0
googletrans==4.0.0-rc1
gTTS>=2.3.2
pygame>=2.5.0
matplotlib>=3.7.0
```

> 💡 **Note on `googletrans`**: Use version `4.0.0-rc1` specifically. Newer versions may have issues.

### System Dependencies

- **Webcam** — Any USB or built-in webcam
- **Speaker / Audio output** — Required for text-to-speech playback
- **Internet connection** — Required for Google Translate API and gTTS (text-to-speech)
- **KhmerOS.ttf font** — Must be placed in the project root directory

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/deaf-gesture-ai.git
cd deaf-gesture-ai
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install opencv-python mediapipe tensorflow scikit-learn numpy Pillow pygame matplotlib gTTS
pip install googletrans==4.0.0-rc1
```

Or if a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe Hand Landmarker Model

```bash
# Download hand_landmarker.task from Google's MediaPipe models
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Or download manually from: [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place `hand_landmarker.task` in the project root.

### 5. Add Khmer Font

Download `KhmerOS.ttf` and place it in the project root directory:
```bash
# On Ubuntu/Debian, you can copy from system fonts
cp /usr/share/fonts/truetype/khmeros/KhmerOS.ttf ./KhmerOS.ttf

# Or download from: https://fonts.google.com/noto/specimen/Noto+Sans+Khmer
```

---

## 🎮 How to Use

The project follows a **4-step pipeline**. Follow each step in order.

---

### 📸 Step 1 — Collect Dataset

Run the dataset collector to capture training images via your webcam:

```bash
python collect_dataset.py
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Start / Pause capturing |
| `N` | Skip to next class |
| `Q` | Quit the program |

**What it does:**
- Loops through each class defined in `CLASSES` list
- Saves **200 images per class** to `./dataset_new/<class_name>/`
- Automatically skips classes that already have 200+ images
- Shows a live progress bar and capture status overlay

> 💡 **Tip:** Make sure your hand is well-lit and in good contrast with the background. Vary your hand position slightly between captures for better generalization.

---

### 🖐 Step 2 — Extract Landmarks

Process all collected images through MediaPipe to extract hand keypoints:

```bash
python extract_landmarks.py
```

**What it does:**
- Reads every image in `./dataset_new/`
- Removes pink/magenta border artifacts (if any)
- Resizes images to 224×224
- Detects 21 hand landmarks using MediaPipe
- Normalizes coordinates relative to the hand bounding box
- Saves a 42-feature vector (21 landmarks × x,y) per image
- Outputs `landmarks_data.pkl` with all data and labels

> ✅ A summary of detected vs. failed images is printed at the end.

---

### 🧠 Step 3 — Train the Model

Train the neural network classifier on the extracted landmarks:

```bash
python train_model.py
```

**What it does:**
- Loads `landmarks_data.pkl`
- Encodes string labels to integers (saved as `label_encoder.pkl`)
- Splits data: 80% training / 20% validation
- Trains a 4-layer dense neural network (see architecture below)
- Uses early stopping (patience = 15 epochs, max 100 epochs)
- Saves the trained model as `sign_language_model.h5`
- Generates `training_results.png` with accuracy/loss curves

> ✅ Typically converges in 30–60 epochs. Expect **90–98% validation accuracy** with a clean dataset.

---

### 🎥 Step 4 — Run Real-time Recognition

Launch the full recognition application:

```bash
python realtime_recognition.py
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Speak current sentence in **Khmer** 🔊 |
| `E` | Speak current sentence in **English** 🔊 |
| `C` | Clear the sentence |
| `F` | Toggle fullscreen mode |
| `Q` | Quit the application |

**How signs are added to the sentence:**
1. Hold your hand sign steady in front of the camera
2. A circular **hold timer** (1.5 seconds) counts down in the sidebar
3. When the timer completes, the sign is added to the sentence
4. The sentence is automatically translated into Khmer
5. Press `SPACE` to hear it spoken aloud

---

## 📁 Dataset Structure

```
dataset_new/
├── hello/
│   ├── hello_0.jpg
│   ├── hello_1.jpg
│   └── ... (200 images recommended)
├── yes/
├── no/
├── thank you/
└── your_custom_class/
```

**Guidelines for best results:**

| Parameter | Recommendation |
|-----------|---------------|
| Images per class | **100 minimum**, 200 recommended |
| Image format | JPG |
| Lighting | Natural or well-lit room |
| Background | Plain, contrasting with skin tone |
| Hand position | Vary slightly — don't all be identical |
| Class name | Folder name = class label shown in the app |

To add new gesture classes, simply add the class name to the `CLASSES` list in `collect_dataset.py` and re-run Steps 1–3.

---

## 🧠 Model Architecture

The classifier uses a **fully connected feed-forward neural network** trained on normalized hand landmark coordinates.

```
Input: 42 features (21 landmarks × [x, y] normalized to bounding box)

┌─────────────────────────────────────────┐
│  Dense(256, activation='relu')          │  ← Feature expansion
│  BatchNormalization()                   │  ← Stable training
│  Dropout(0.4)                           │  ← Prevent overfitting
├─────────────────────────────────────────┤
│  Dense(128, activation='relu')          │
│  BatchNormalization()                   │
│  Dropout(0.3)                           │
├─────────────────────────────────────────┤
│  Dense(64, activation='relu')           │
│  Dropout(0.2)                           │
├─────────────────────────────────────────┤
│  Dense(num_classes, activation='softmax')│  ← Probability per class
└─────────────────────────────────────────┘

Optimizer : Adam
Loss      : Categorical Cross-entropy
Metric    : Accuracy
Early Stop: Patience = 15 epochs (restores best weights)
```

**Why landmarks, not raw images?**
Using 42 landmark coordinates instead of raw pixels makes the model:
- **Much smaller** (~few KB vs. MB)
- **Robust to lighting changes** — colors don't matter, only shape
- **Scale and position invariant** — coordinates are normalized
- **Faster** — no CNN feature extraction needed at runtime

---

## 🛠 Technologies Used

<div align="center">

| Technology | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Core programming language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Neural network training & inference |
| ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white) | High-level model API |
| ![MediaPipe](https://img.shields.io/badge/MediaPipe-00897B?logo=google&logoColor=white) | Hand landmark detection |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) | Webcam capture & image processing |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Label encoding & data splitting |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical computation |
| ![Pillow](https://img.shields.io/badge/Pillow-FFD700?logo=python&logoColor=black) | Khmer Unicode text rendering |
| ![gTTS](https://img.shields.io/badge/gTTS-4285F4?logo=google&logoColor=white) | Google Text-to-Speech |
| ![Google Translate](https://img.shields.io/badge/Google%20Translate-4285F4?logo=google-translate&logoColor=white) | English → Khmer translation |
| ![Pygame](https://img.shields.io/badge/Pygame-green?logo=python&logoColor=white) | Audio playback |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white) | Training visualization |

</div>

---

## 🔮 Future Improvements

- [ ] 📱 **Mobile App** — Port to Android/iOS using TensorFlow Lite for on-device inference
- [ ] 🤝 **Two-hand support** — Extend MediaPipe to detect both hands for more complex signs
- [ ] 🎬 **Dynamic gestures** — Support motion-based signs (e.g., J, Z in ASL) using LSTM/temporal models
- [ ] 🇰🇭 **Cambodian Sign Language (CSL)** — Train on native CSL dataset instead of ASL
- [ ] 🌐 **Offline translation** — Replace Google Translate with an offline Khmer translation model
- [ ] 📊 **Larger dataset** — Collect data from multiple users for better generalization
- [ ] 🎨 **GUI settings panel** — Allow users to adjust hold time, confidence threshold, font size
- [ ] 💬 **Chat history** — Save recognized sentences to a log file or database
- [ ] 🔁 **Feedback loop** — Allow users to correct misrecognitions to improve the model over time
- [ ] 🚀 **Web interface** — Serve the model via Flask/FastAPI for browser-based access

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Input features | 42 (21 landmarks × x,y) |
| Confidence threshold | 75% |
| Prediction smoothing | 10-frame buffer |
| Hold-to-confirm duration | 1.5 seconds |
| Supported classes | 43+ (A–Z, 0–9, words) |
| Typical validation accuracy | ~90–98% |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

<div align="center">

| | |
|---|---|
| **Name** | Kong Sisovandara |
| **University** | RUPP |
| **Faculty** | Engineer |
| **Degree** | Bachelor of Science in ITE |
| **Country** | 🇰🇭 Cambodia |
| **Year** | 2026 |
| **Project Type** | Final Year Project / Project Practicum |
| **GitHub** | [@kong-sisovandara](https://github.com/kong-sisovandara) |

</div>

---

## 🙏 Acknowledgments

- [Google MediaPipe](https://developers.google.com/mediapipe) — for the incredible hand landmark detection model
- [gTTS (Google Text-to-Speech)](https://gtts.readthedocs.io/) — for Khmer TTS support
- [KhmerOS Font Project](https://khmeros.info/) — for the open-source Khmer Unicode font
- The **Deaf community in Cambodia** — for the inspiration behind this project
