# Sign Language Recognition — Flask Web App

Real-time hand gesture recognition with a Flask API backend and Jinja2 web UI.  
Detects ASL letters and common words, builds sentences, translates to Khmer, and speaks aloud.

---

## Folder Structure

```
sign-language-recognition/
├── app/
│   ├── __init__.py               ← Flask app factory
│   ├── config.py                 ← All config variables (paths, thresholds)
│   ├── extensions.py             ← Module-level service registry
│   ├── models/
│   │   └── model_loader.py       ← Loads .h5 model + label encoder
│   ├── routes/
│   │   ├── main.py               ← GET / and GET /video_feed
│   │   └── api.py                ← All /api/* JSON endpoints
│   ├── services/
│   │   ├── hand_detector.py      ← MediaPipe detection + camera thread
│   │   ├── gesture_recognizer.py ← TensorFlow prediction + hold timer + sentence
│   │   ├── translator.py         ← Google Translate EN→KM
│   │   └── tts_service.py        ← gTTS audio generation
│   ├── utils/
│   │   └── landmark_utils.py     ← Landmark normalization helper
│   ├── static/
│   │   ├── css/style.css         ← Dark navy theme
│   │   ├── js/app.js             ← Polling, TTS playback, UI updates
│   │   └── fonts/KhmerOS.ttf     ← Khmer font
│   └── templates/
│       ├── base.html             ← Jinja2 base template
│       └── index.html            ← Main UI page
├── scripts/
│   ├── collect_dataset.py        ← Webcam dataset collection
│   ├── extract_landmarks.py      ← MediaPipe landmark extraction
│   └── train_model.py            ← TensorFlow model training
├── models/                       ← Saved model files (copy here)
│   ├── sign_language_model.h5
│   ├── label_encoder.pkl
│   └── hand_landmarker.task
├── dataset/                      ← Training images
├── requirements.txt
├── run.py                        ← Entry point
└── README.md
```

---

## Setup

### 1. Copy model files

Before running, copy your trained model files into the `models/` directory:

```bash
cp ../sign_language_model.h5  models/
cp ../label_encoder.pkl       models/
cp ../hand_landmarker.task    models/
```

Copy the Khmer font into the static fonts directory:

```bash
cp ../KhmerOS.ttf app/static/fonts/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python run.py
```

Open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Route                  | Description                                      |
|--------|------------------------|--------------------------------------------------|
| GET    | `/`                    | Main UI page                                     |
| GET    | `/video_feed`          | MJPEG webcam stream with landmark overlay        |
| GET    | `/api/predict`         | Full recognition state (sign, confidence, etc.)  |
| GET    | `/api/status`          | Real-time sign + hold progress                   |
| GET    | `/api/sentence`        | Current sentence + Khmer translation             |
| POST   | `/api/speak`           | Generate + stream TTS audio (MP3)                |
| POST   | `/api/translate`       | Translate English text to Khmer                  |
| POST   | `/api/sentence/add`    | Manually add a sign to the sentence              |
| POST   | `/api/sentence/clear`  | Clear the sentence                               |

### Example requests

```bash
# Get current status
curl http://localhost:5000/api/status

# Speak Khmer
curl -X POST http://localhost:5000/api/speak \
     -H "Content-Type: application/json" \
     -d '{"text": "ជំរាបសួរ", "lang": "km"}'

# Translate
curl -X POST http://localhost:5000/api/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "hello world"}'
```

---

## Keyboard Shortcuts (in browser)

| Key     | Action         |
|---------|----------------|
| `Space` | Speak Khmer    |
| `E`     | Speak English  |
| `C`     | Clear sentence |

---

## Configuration

All tunable values live in `app/config.py`:

| Variable                       | Default | Description                           |
|--------------------------------|---------|---------------------------------------|
| `HOLD_SECONDS`                 | `1.5`   | Seconds to hold a sign before confirm |
| `CONFIDENCE_THRESHOLD`         | `0.75`  | Min model confidence to buffer sign   |
| `PREDICTION_BUFFER_SIZE`       | `10`    | Smoothing window (majority vote)      |
| `MIN_HAND_DETECTION_CONFIDENCE`| `0.7`   | MediaPipe detection threshold         |
| `CAMERA_INDEX`                 | `0`     | Webcam device index                   |
| `JPEG_QUALITY`                 | `85`    | MJPEG stream quality (1–100)          |

---

## Retrain the model

```bash
# 1. Collect new dataset images
python scripts/collect_dataset.py

# 2. Extract landmarks from images
python scripts/extract_landmarks.py

# 3. Train the model
python scripts/train_model.py

# 4. Copy new model files to models/
cp models/sign_language_model.h5 models/
cp models/label_encoder.pkl      models/
```
