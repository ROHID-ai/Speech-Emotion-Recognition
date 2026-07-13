"""
FastAPI Speech Emotion Recognition API.

IMPORTANT (Render): Heavy ML imports and HuBERT weights are NOT loaded at
module import time. Uvicorn must bind to $PORT quickly so Render's deploy
health check can succeed. Models load lazily on the first /predict request.
"""

from __future__ import annotations

import os
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

# Quiet TensorFlow once it is eventually imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "best_hubert_emotion.weights.h5"
TARGET_SR = 16000
MAX_FRAMES = 199
HUBERT_MODEL_ID = "facebook/hubert-base-ls960"

INV_EMOTION_MAP = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised",
}

# Lazy-loaded globals (filled by ensure_models_loaded)
_feature_extractor = None
_hubert_model = None
_classifier = None
_tf = None
_models_ready = False
_models_error: str | None = None
_load_lock = threading.Lock()


class HubertEmotionClassifier:
    """Keras model wrapper; real class is built after TensorFlow import."""

    pass


def _build_classifier(tf_module: Any, num_classes: int = 8):
    class _HubertEmotionClassifier(tf_module.keras.Model):
        def __init__(self, num_classes: int = 8):
            super().__init__()
            reg = tf_module.keras.regularizers.l2(1e-4)
            self.input_norm = tf_module.keras.layers.LayerNormalization()
            self.conv1d = tf_module.keras.layers.Conv1D(
                64,
                kernel_size=5,
                padding="same",
                activation="relu",
                kernel_regularizer=reg,
            )
            self.bn1 = tf_module.keras.layers.BatchNormalization()
            self.spatial_dropout = tf_module.keras.layers.SpatialDropout1D(0.3)
            self.pool1 = tf_module.keras.layers.MaxPooling1D(pool_size=4)
            self.bilstm = tf_module.keras.layers.Bidirectional(
                tf_module.keras.layers.LSTM(64, return_sequences=True, dropout=0.3)
            )
            self.global_pool = tf_module.keras.layers.GlobalAveragePooling1D()
            self.dense1 = tf_module.keras.layers.Dense(
                64, activation="relu", kernel_regularizer=reg
            )
            self.final_dropout = tf_module.keras.layers.Dropout(0.4)
            self.classifier = tf_module.keras.layers.Dense(num_classes)

        def call(self, inputs, training=False):
            x = self.input_norm(inputs)
            x = self.conv1d(x)
            x = self.bn1(x, training=training)
            x = self.spatial_dropout(x, training=training)
            x = self.pool1(x)
            x = self.bilstm(x, training=training)
            x = self.global_pool(x)
            x = self.dense1(x)
            x = self.final_dropout(x, training=training)
            return self.classifier(x)

    return _HubertEmotionClassifier(num_classes=num_classes)


def ensure_models_loaded() -> None:
    """Load Torch/TF/HuBERT/classifier once. Safe to call from multiple threads."""
    global _feature_extractor, _hubert_model, _classifier, _tf
    global _models_ready, _models_error

    if _models_ready:
        return
    if _models_error:
        raise RuntimeError(_models_error)

    with _load_lock:
        if _models_ready:
            return
        if _models_error:
            raise RuntimeError(_models_error)

        try:
            print("Loading ML dependencies and models (first request only)...")

            if not WEIGHTS_PATH.is_file():
                raise FileNotFoundError(
                    f"Classifier weights not found at {WEIGHTS_PATH}. "
                    "Ensure best_hubert_emotion.weights.h5 is deployed with the service."
                )

            import numpy as np  # noqa: F401
            import torch
            import tensorflow as tf
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            _tf = tf

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_ID)
            hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_ID)
            hubert_model.eval()

            classifier = _build_classifier(tf, num_classes=8)
            # Build variables, then load trained weights.
            classifier(tf.zeros((1, MAX_FRAMES, 768)))
            classifier.load_weights(str(WEIGHTS_PATH))

            _feature_extractor = feature_extractor
            _hubert_model = hubert_model
            _classifier = classifier
            _models_ready = True
            print("Models loaded. Inference ready.")
        except Exception as exc:
            _models_error = f"Model initialization failed: {exc}"
            print(_models_error)
            raise RuntimeError(_models_error) from exc


def predict_emotion(audio_path: str) -> dict:
    ensure_models_loaded()

    import librosa
    import numpy as np
    import torch
    import tensorflow as tf

    speech, _sr = librosa.load(audio_path, sr=TARGET_SR)
    speech = librosa.util.normalize(speech)

    input_values = _feature_extractor(
        speech, return_tensors="pt", sampling_rate=TARGET_SR
    ).input_values

    with torch.no_grad():
        outputs = _hubert_model(input_values, output_hidden_states=True)
        layer_features = outputs.hidden_states[7].squeeze(0).numpy()

    num_frames = layer_features.shape[0]
    if num_frames >= MAX_FRAMES:
        final_features = layer_features[:MAX_FRAMES, :]
    else:
        pad_width = MAX_FRAMES - num_frames
        final_features = np.pad(
            layer_features, ((0, pad_width), (0, 0)), mode="constant"
        )

    final_features = np.expand_dims(final_features, axis=0)

    logits = _classifier(final_features, training=False)
    probabilities = tf.nn.softmax(logits).numpy()[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "emotion": INV_EMOTION_MAP[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "all_probabilities": {
            INV_EMOTION_MAP[i]: float(p) for i, p in enumerate(probabilities)
        },
    }


# ==========================================
# APP (binds immediately — no model load here)
# ==========================================
app = FastAPI(title="Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health endpoint used by Render. Must stay lightweight."""
    return {
        "message": "Emotion Recognition API is running",
        "models_ready": _models_ready,
        "models_error": _models_error,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": _models_ready}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(
        (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    ):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    suffix = os.path.splitext(file.filename)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            return predict_emotion(tmp_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {exc}"
            ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
