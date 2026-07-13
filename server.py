"""
FastAPI Speech Emotion Recognition API.

Models are loaded during application lifespan startup (not at import time,
and not deferred until the first /predict). GET / reports models_ready /
models_error so Render logs and health checks expose real load failures.
"""

from __future__ import annotations

import os
import tempfile
import traceback
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# PATHS — server.py and weights live in the same directory (repo root).
# Do NOT use parent.parent; that points outside the project on Render.
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = (BASE_DIR / "best_hubert_emotion.weights.h5").resolve()
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

# Runtime model state (set by load_models_at_startup)
feature_extractor = None
hubert_model = None
classifier = None
models_ready = False
models_error: str | None = None


def _build_classifier(tf_module: Any, num_classes: int = 8):
    """Build the Keras emotion head used with HuBERT layer-7 features."""

    class HubertEmotionClassifier(tf_module.keras.Model):
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

    return HubertEmotionClassifier(num_classes=num_classes)


def verify_required_files() -> None:
    """Fail fast with clear messages if local artifacts are missing."""
    print(f"[startup] BASE_DIR (absolute)     = {BASE_DIR}")
    print(f"[startup] WEIGHTS_PATH (absolute) = {WEIGHTS_PATH}")
    print(f"[startup] WEIGHTS_PATH exists     = {WEIGHTS_PATH.is_file()}")
    print(f"[startup] cwd                     = {Path.cwd().resolve()}")
    print(f"[startup] repo listing            = {sorted(p.name for p in BASE_DIR.iterdir())}")

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            "Missing required model file: best_hubert_emotion.weights.h5\n"
            f"Expected absolute path: {WEIGHTS_PATH}\n"
            f"BASE_DIR contents: {sorted(p.name for p in BASE_DIR.iterdir())}\n"
            "HuBERT itself is downloaded from Hugging Face "
            f"({HUBERT_MODEL_ID}) and is not a local file."
        )


def load_models_at_startup() -> None:
    """
    Load all models used by this API:

    1) Wav2Vec2FeatureExtractor + HubertModel (PyTorch / transformers)
    2) HubertEmotionClassifier weights (TensorFlow / Keras .h5)

    There is no separate torch.load() emotion checkpoint in this project.
    """
    global feature_extractor, hubert_model, classifier
    global models_ready, models_error

    models_ready = False
    models_error = None

    try:
        verify_required_files()

        print("Loading PyTorch / Transformers stack...")
        import torch
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        print(f"Loading HuBERT... ({HUBERT_MODEL_ID})")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_ID)
        hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_ID)
        hubert_model.eval()
        print(
            f"HuBERT loaded. torch={torch.__version__}, "
            f"device=cpu, eval={not hubert_model.training}"
        )

        print("Loading TensorFlow / Keras emotion classifier...")
        import tensorflow as tf

        classifier = _build_classifier(tf, num_classes=8)
        # Build variables before loading trained weights.
        classifier(tf.zeros((1, MAX_FRAMES, 768)))
        print(f"Loading TensorFlow model weights from: {WEIGHTS_PATH}")
        classifier.load_weights(str(WEIGHTS_PATH))
        print(f"TensorFlow loaded. tf={tf.__version__}")

        models_ready = True
        models_error = None
        print("All models loaded successfully.")
    except Exception as e:
        traceback.print_exc()
        models_ready = False
        models_error = f"{type(e).__name__}: {e}"
        print(f"[startup] MODEL LOAD FAILED: {models_error}")
        # Keep the HTTP server up so GET / can expose models_error on Render.


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[startup] lifespan begin — loading models")
    load_models_at_startup()
    if models_ready:
        print("[startup] lifespan complete — models_ready=True")
    else:
        print(f"[startup] lifespan complete — models_ready=False error={models_error}")
    yield
    print("[shutdown] lifespan end")


app = FastAPI(title="Emotion Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_emotion(audio_path: str) -> dict:
    if not models_ready:
        raise RuntimeError(
            models_error or "Models are not loaded. Check / for models_error."
        )

    import librosa
    import numpy as np
    import torch
    import tensorflow as tf

    speech, _sr = librosa.load(audio_path, sr=TARGET_SR)
    speech = librosa.util.normalize(speech)

    input_values = feature_extractor(
        speech, return_tensors="pt", sampling_rate=TARGET_SR
    ).input_values

    with torch.no_grad():
        outputs = hubert_model(input_values, output_hidden_states=True)
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

    logits = classifier(final_features, training=False)
    probabilities = tf.nn.softmax(logits).numpy()[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "emotion": INV_EMOTION_MAP[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "all_probabilities": {
            INV_EMOTION_MAP[i]: float(p) for i, p in enumerate(probabilities)
        },
    }


@app.get("/")
def root():
    return {
        "message": (
            "Emotion Recognition API is running"
            if models_ready
            else "Emotion Recognition API is running, but models failed to load"
        ),
        "models_ready": models_ready,
        "models_error": models_error,
        "weights_path": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.is_file(),
        "base_dir": str(BASE_DIR),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if models_ready else "degraded",
        "models_ready": models_ready,
        "models_error": models_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not models_ready:
        raise HTTPException(
            status_code=503,
            detail=models_error or "Models are not loaded yet",
        )

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
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {type(e).__name__}: {e}"
            ) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
