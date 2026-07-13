# Speech Emotion Recognition API

FastAPI backend that classifies emotion in speech audio using HuBERT features and a Keras classifier.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Health check: `GET http://127.0.0.1:8000/`  
Predict: `POST http://127.0.0.1:8000/predict` (multipart field `file`)

## Render deployment

### Exact settings

| Setting | Value |
|--------|--------|
| Runtime | Python 3 |
| Python version | `3.11.9` (from `.python-version`) |
| Root Directory | *(leave empty)* — `server.py` is at repo root |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn server:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/` |
| Instance | **Standard (2GB+) required** — Free 512MB will OOM when models load |

### Environment variables

```
WEB_CONCURRENCY=1
TF_CPP_MIN_LOG_LEVEL=3
TOKENIZERS_PARALLELISM=false
HF_HOME=/opt/render/project/src/.cache/huggingface
TRANSFORMERS_CACHE=/opt/render/project/src/.cache/huggingface
```

### Why deploys used to hang at "Deploying..."

`server.py` previously loaded TensorFlow, PyTorch, and downloaded HuBERT **at import time**, before Uvicorn could bind `$PORT`. Render waits for an open port after `WEB_CONCURRENCY=1`, so the service sat on "Deploying..." until timeout / OOM.

Models now load **lazily on the first `/predict` request**. The process binds immediately and `/` returns healthy.

### Entry point

- File: `server.py`
- App object: `app = FastAPI(...)`
- Uvicorn module path: `server:app`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

You can also deploy from `render.yaml` in this repository.
