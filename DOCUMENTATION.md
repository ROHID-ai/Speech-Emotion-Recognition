# Cross-Speaker Speech Emotion Recognition (SER)

## 1. Project Overview

This project is a full-stack Speech Emotion Recognition (SER) system that predicts the emotional state expressed in a human voice recording. A user uploads a short audio clip through a web interface. The backend processes the audio with a deep learning pipeline and returns the most likely emotion, along with confidence scores for all supported classes.

The system is designed for **cross-speaker** recognition. Instead of relying on speaker-specific voice traits, it uses intermediate acoustic representations from Facebook’s pre-trained **HuBERT** model. Those features are then classified by a custom Temporal Convolutional–Bidirectional LSTM network. This approach reduces speaker-identity overfitting (voice profiling) and improves generalization to unseen speakers. The reported validation accuracy on held-out voices is **78.75%**.

Supported emotion classes:

| Index | Emotion    |
|-------|------------|
| 0     | Neutral   |
| 1     | Calm       |
| 2     | Happy      |
| 3     | Sad        |
| 4     | Angry      |
| 5     | Fearful    |
| 6     | Disgust    |
| 7     | Surprised  |

Supported audio formats: **WAV, MP3, FLAC, OGG, M4A**.

---

## 2. Project Structure

```
Emotion_Detector/
├── server.py                          # FastAPI backend and inference pipeline
├── requirements.txt                   # Python dependencies
├── best_hubert_emotion.weights.h5     # Trained classifier weights
├── extract_features.py.save           # Offline HuBERT feature extraction script (training prep)
├── DOCUMENTATION.md                   # This document
└── Emotion_Detector/                  # React + Vite frontend
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        └── Components/
            └── EmotionUploader.jsx    # Upload UI and result visualization
```

---

## 3. System Architecture

The application has three logical layers:

1. **Presentation layer (Frontend)**  
   A React single-page application built with Vite. The main component, `EmotionUploader`, lets the user select or drag-and-drop an audio file, sends it to the API, and displays the predicted emotion with a probability bar chart.

2. **Application layer (Backend API)**  
   A FastAPI server (`server.py`) exposes REST endpoints. It accepts multipart file uploads, runs inference, and returns JSON results. CORS is enabled so the local frontend can call the API during development.

3. **Model / intelligence layer**  
   Two models work together at inference time:
   - **HuBERT Base (facebook/hubert-base-ls960)** — extracts frame-level speech representations from raw audio (PyTorch / Hugging Face Transformers).
   - **HubertEmotionClassifier** — a TensorFlow/Keras classifier that maps HuBERT features to one of eight emotion labels. Weights are loaded from `best_hubert_emotion.weights.h5`.

```
Audio file
    │
    ▼
[Frontend] EmotionUploader  ──POST /predict──►  [FastAPI] server.py
                                                    │
                                                    ▼
                                            Load & normalize audio (librosa, 16 kHz)
                                                    │
                                                    ▼
                                            HuBERT feature extractor + model
                                            (hidden state layer 7 → shape [T, 768])
                                                    │
                                                    ▼
                                            Pad / truncate to [199, 768]
                                                    │
                                                    ▼
                                            HubertEmotionClassifier (Keras)
                                                    │
                                                    ▼
                                            Softmax → emotion + confidence
                                                    │
                                                    ▼
[Frontend] displays result ◄────────────── JSON response
```

---

## 4. How the System Works (End-to-End)

### 4.1 User interaction

1. The user opens the frontend (typically `http://localhost:5173`).
2. The user drops or selects an audio file.
3. The client validates the file extension client-side.
4. On **Analyze voice**, the browser builds a `FormData` payload with the file and sends `POST http://127.0.0.1:8000/predict`.
5. While waiting, the UI shows an analyzing state.
6. On success, the UI shows:
   - Predicted emotion label
   - Confidence percentage
   - Eight probability bars (one per emotion class)
7. On failure, an error message is shown (unsupported format, server error, or network failure).

### 4.2 Backend request handling

1. `POST /predict` receives the uploaded file.
2. The server checks that the filename ends with an allowed extension.
3. The file bytes are written to a temporary file so librosa can read them.
4. `predict_emotion()` runs the full inference pipeline.
5. The temporary file is deleted after processing.
6. A JSON object is returned to the client.

Example successful response:

```json
{
  "emotion": "Happy",
  "confidence": 0.8721,
  "all_probabilities": {
    "Neutral": 0.01,
    "Calm": 0.02,
    "Happy": 0.8721,
    "Sad": 0.03,
    "Angry": 0.02,
    "Fearful": 0.01,
    "Disgust": 0.01,
    "Surprised": 0.0279
  }
}
```

### 4.3 Inference pipeline (detailed)

**Step A — Audio loading**  
The audio is loaded with librosa at a target sample rate of **16,000 Hz** and amplitude-normalized. This matches HuBERT’s expected input format.

**Step B — HuBERT feature extraction**  
`Wav2Vec2FeatureExtractor` prepares the waveform tensor. The HuBERT model runs in evaluation mode with gradients disabled. The pipeline requests all hidden states and uses **layer 7** (index 7).  

Layer 7 is used because mid-depth HuBERT layers tend to capture **acoustic and prosodic** information useful for emotion, while deeper layers become more phonetic/speaker-oriented. The output at this stage is a matrix of shape `[num_frames, 768]`.

**Step C — Sequence length normalization**  
The classifier expects a fixed input of **199 frames × 768 features**:
- If the clip is longer, features are truncated to the first 199 frames.
- If the clip is shorter, zero-padding is applied to reach 199 frames.

**Step D — Emotion classification**  
The padded/truncated tensor is passed through `HubertEmotionClassifier`:

1. Layer normalization  
2. 1D convolution (64 filters, kernel size 5) with ReLU and L2 regularization  
3. Batch normalization  
4. Spatial dropout (0.3)  
5. Max pooling (pool size 4)  
6. Bidirectional LSTM (64 units per direction, dropout 0.3)  
7. Global average pooling  
8. Dense layer (64 units, ReLU, L2)  
9. Dropout (0.4)  
10. Dense output (8 logits)

Softmax converts logits to class probabilities. The class with the highest probability is selected as the predicted emotion.

---

## 5. Backend Components

### 5.1 `server.py`

| Element | Purpose |
|---------|---------|
| `WEIGHTS_PATH` | Path to trained Keras weights |
| `TARGET_SR` | Audio resample rate (16000) |
| `MAX_FRAMES` | Fixed sequence length (199) |
| `INV_EMOTION_MAP` | Maps class indices to emotion names |
| `HubertEmotionClassifier` | Custom Keras model definition |
| `feature_extractor` / `hubert_model` | Loaded once at startup |
| `predict_emotion()` | Core inference function |
| `GET /` | Health / status check |
| `POST /predict` | Emotion prediction endpoint |

Models are loaded when the server starts. The first startup may take longer because HuBERT weights are downloaded from Hugging Face if not already cached.

### 5.2 `best_hubert_emotion.weights.h5`

Contains the trained parameters of `HubertEmotionClassifier`. This file must remain in the same working directory used when launching uvicorn, because the server loads it via a relative path.

### 5.3 `extract_features.py.save`

Offline training-preparation script (not required for running the app). It walks a dataset of WAV files, extracts HuBERT layer-7 features for each clip, normalizes length to `[199, 768]`, and saves `.npy` files. Those features can be used to train or retrain the emotion classifier separately from the live API.

### 5.4 `requirements.txt`

Main Python dependencies:

- **TensorFlow** — classifier model  
- **PyTorch + Transformers** — HuBERT model  
- **librosa / soundfile / audioread** — audio I/O and preprocessing  
- **NumPy / tqdm** — numerics and progress utilities  
- **FastAPI + uvicorn + python-multipart** — HTTP API and file uploads  

Note: The pinned Torch line references a CUDA wheel index. On macOS or CPU-only machines, install a CPU build of PyTorch separately if that install step fails.

---

## 6. Frontend Components

### 6.1 Technology stack

- React 19  
- Vite 8  
- Lucide React (icons)  
- Tailwind-style utility classes (used in component markup)

### 6.2 `App.jsx`

Root component that renders `EmotionUploader`.

### 6.3 `EmotionUploader.jsx`

Primary user interface. Responsibilities:

- File selection via click or drag-and-drop  
- Client-side format validation  
- Upload to `http://127.0.0.1:8000/predict`  
- UI states: idle, analyzing, done, error  
- Visualization of emotion probabilities as animated bars  
- Display of predicted emotion and confidence  
- Reset flow to try another clip  

The emotion color order in the UI matches `INV_EMOTION_MAP` on the backend so bars always correspond to the same eight classes.

---

## 7. API Reference

### `GET /`

Returns a simple status message confirming the API is running.

**Response**

```json
{ "message": "Emotion Recognition API is running" }
```

### `POST /predict`

Accepts a multipart form upload with field name `file`.

**Request**

- Content-Type: `multipart/form-data`  
- Field: `file` (audio)

**Success (200)**  
JSON with `emotion`, `confidence`, and `all_probabilities` (see Section 4.2).

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Unsupported audio format |
| 500 | Inference failure (corrupt audio, model error, etc.) |

---

## 8. How to Run the Project

### Prerequisites

- Python 3.10+ recommended  
- Node.js 18+ and npm  
- Sufficient disk space for HuBERT model download on first run  

### 8.1 Backend

```bash
cd Emotion_Detector

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# If Torch CUDA install fails (e.g. on Mac), use CPU Torch first:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt

uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Confirm the API at: `http://127.0.0.1:8000`

### 8.2 Frontend

Open a second terminal:

```bash
cd Emotion_Detector/Emotion_Detector

npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

### 8.3 Verify

1. Backend root endpoint responds.  
2. Frontend loads without console errors.  
3. Upload a short speech clip and click **Analyze voice**.  
4. Emotion label and probability bars appear.

---

## 9. Design Rationale

**Why HuBERT?**  
HuBERT is a self-supervised speech model trained on large-scale unlabeled audio. Its hidden states provide rich acoustic features without requiring handcrafted spectrogram engineering for every experiment.

**Why middle layer (layer 7)?**  
Emotion is strongly tied to prosody (pitch, energy, rhythm) and mid-level acoustics. Using an intermediate layer balances emotion-relevant cues against speaker-identity or pure phonetic content that deeper layers emphasize.

**Why a separate Keras classifier?**  
HuBERT acts as a frozen feature extractor. The lighter Conv1D + BiLSTM head is trained specifically for emotion classification, with dropout and L2 regularization to improve robustness on unseen speakers.

**Why fixed 199 frames?**  
A fixed-length input simplifies batching and model architecture. Truncation/padding is a practical compromise for variable-length speech utterances.

**Why decoupled frontend and backend?**  
Separating the React UI from the Python ML service keeps the heavy model stack independent of the browser, allows API reuse, and simplifies local development (two processes, CORS-enabled).

---

## 10. Limitations and Notes

- First backend startup downloads HuBERT and loads TensorFlow/PyTorch models; this can take noticeable time and memory.  
- Very short or silent clips may produce unreliable predictions.  
- The classifier was trained under a specific feature-extraction setup; changing sample rate, layer index, or frame length without retraining will degrade results.  
- CORS currently allows all origins (`*`), which is convenient for local development but should be restricted for production.  
- The frontend backend URL is hardcoded to `127.0.0.1:8000`; update `BACKEND_URL` in `EmotionUploader.jsx` if the API host or port changes.

---

## 11. Summary

This project delivers a complete speech emotion recognition application: a React frontend for audio upload and visualization, and a FastAPI backend that converts speech into HuBERT representations and classifies them into eight emotion categories. The hybrid HuBERT + temporal classifier design targets cross-speaker generalization, and the provided weights enable immediate local inference without retraining.
