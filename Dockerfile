# ── Stage 1: base image with CUDA support ────────────────────────────────────
# Use slim CPU image by default; swap for nvidia/cuda if you have a GPU host.
FROM python:3.11-slim

# System deps (PIL needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy model checkpoint and thresholds if they exist locally
# (they can also be mounted as volumes or downloaded at startup)
# COPY best_phase2.pt .
# COPY best_thresholds_cv.npy .

# Environment defaults (override at runtime)
ENV MODEL_CHECKPOINT_PATH=best_phase2.pt
ENV THRESHOLDS_PATH=best_thresholds_cv.npy
ENV PORT=8000

EXPOSE 8000

# Use --workers 1 — model is loaded once in lifespan, don't fork
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
