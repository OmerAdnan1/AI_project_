# Anime Genre Classifier — Python Backend

FastAPI backend that serves predictions from the ViT-B/16 + CLIP anime genre classifier.

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — model loading, TTA inference, `/classify` endpoint |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image |
| `docker-compose.yml` | Local dev helper |

---

## Quick start (local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Put your model files in the same directory

```
anime-backend/
├── main.py
├── best_phase2.pt          ← your Kaggle checkpoint
└── best_thresholds_cv.npy  ← CV-tuned thresholds (optional)
```

### 3. Run

```bash
MODEL_CHECKPOINT_PATH=best_phase2.pt \
THRESHOLDS_PATH=best_thresholds_cv.npy \
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000`.

---

## API reference

### `GET /health`
Returns `{"status": "ok", "device": "cuda|cpu", "model_loaded": true}`.

### `POST /classify`
Accepts `multipart/form-data` with a single field `file` (JPG or PNG, max 5 MB).

**Response**
```json
{
  "genres": [
    {"name": "Action",  "confidence": 0.82},
    {"name": "Fantasy", "confidence": 0.61}
  ]
}
```
Genres are sorted by confidence descending.  
Only genres above the per-genre threshold are returned (minimum 1).

---

## Docker

```bash
# Build
docker build -t anime-backend .

# Run — mount your checkpoint files
docker run -p 8000:8000 \
  -v $(pwd)/best_phase2.pt:/app/best_phase2.pt:ro \
  -v $(pwd)/best_thresholds_cv.npy:/app/best_thresholds_cv.npy:ro \
  anime-backend
```

Or with Compose:
```bash
# Edit docker-compose.yml to uncomment the volume mounts, then:
docker compose up --build
```

---

## Deploying to a GPU server (Railway / Render / RunPod / Modal)

The backend is stateless — one instance, one loaded model.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CHECKPOINT_PATH` | `best_phase2.pt` | Path to `.pt` checkpoint inside container |
| `THRESHOLDS_PATH` | `best_thresholds_cv.npy` | Path to `.npy` thresholds (optional) |
| `PORT` | `8000` | Port uvicorn listens on |

### Vercel → Backend connection

In your Vercel project settings, set:
```
PYTHON_BACKEND_URL=https://your-backend-domain.com
```

The Next.js API route at `/api/classify` will POST to `${PYTHON_BACKEND_URL}/classify`.

### CORS

The backend currently allows all origins (`*`).  
For production, restrict `allow_origins` in `main.py` to your Vercel domain:

```python
allow_origins=["https://your-app.vercel.app"],
```

---

## Checkpoint note

The checkpoint (`best_phase2.pt`) is saved by Kaggle Cell 15 as:
```python
torch.save({
    'epoch'      : epoch,
    'model_state': raw_state,   # stripped of 'module.' prefix
    'optim_state': optimizer.state_dict(),
    'val_f1'     : val_f1,
    'val_loss'   : val_loss,
}, save_path)
```

The backend loads `ckpt['model_state']` directly — no manual stripping needed.

The thresholds file (`best_thresholds_cv.npy`) is saved by Cell 16 and is optional;  
if absent, a uniform threshold of 0.30 is used for all genres.
