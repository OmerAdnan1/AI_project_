"""
Anime Genre Classifier — FastAPI Backend
Loads ViT-B/16 + CLIP model weights and serves predictions.
"""

import os
import io
import logging
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager
from typing import List

import torch
import torch.nn as nn
import clip

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
KEEP_GENRES = [
    'Action', 'Adventure', 'Avant Garde', 'Comedy', 'Drama',
    'Ecchi', 'Fantasy', 'Horror', 'Mecha', 'Mystery', 'Romance',
    'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural'
]
NUM_GENRES  = len(KEEP_GENRES)
IMG_SIZE    = 224
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Default thresholds (overridden by best_thresholds_cv.npy if present)
DEFAULT_THRESHOLD = 0.30

GENRE_PROMPTS = {
    'Action'       : ["an anime cover of an action series",
                      "a fast-paced action anime with fighting"],
    'Adventure'    : ["an anime cover of an adventure series",
                      "an adventure anime with journeys and exploration"],
    'Avant Garde'  : ["an anime cover of an avant garde or experimental series",
                      "an abstract or artistic avant-garde anime"],
    'Comedy'       : ["an anime cover of a comedy series",
                      "a funny comedy anime with humorous characters"],
    'Drama'        : ["an anime cover of a drama series",
                      "a dramatic and emotional anime"],
    'Ecchi'        : ["an anime cover of an ecchi series",
                      "a suggestive ecchi anime"],
    'Fantasy'      : ["an anime cover of a fantasy series",
                      "a magical fantasy anime with supernatural worlds"],
    'Horror'       : ["an anime cover of a horror series",
                      "a dark and scary horror anime"],
    'Mecha'        : ["an anime cover of a mecha series",
                      "a robot mecha anime with giant machines"],
    'Mystery'      : ["an anime cover of a mystery series",
                      "a mystery detective anime"],
    'Romance'      : ["an anime cover of a romance series",
                      "a romantic love story anime"],
    'Sci-Fi'       : ["an anime cover of a sci-fi series",
                      "a science fiction anime with futuristic technology"],
    'Slice of Life': ["an anime cover of a slice of life series",
                      "an everyday life anime with ordinary characters"],
    'Sports'       : ["an anime cover of a sports series",
                      "a sports competition anime"],
    'Supernatural' : ["an anime cover of a supernatural series",
                      "an anime with supernatural powers and phenomena"],
}


# ── Model Definition (must match training exactly) ───────────────────────────
class AnimeGenreClassifier(nn.Module):
    def __init__(self, clip_visual, text_emb, num_genres=15,
                 dropout=0.4, patch_weight=0.3):
        super().__init__()
        self.encoder      = clip_visual
        self.patch_weight = patch_weight
        self.register_buffer('text_emb', text_emb)
        self.alpha         = nn.Parameter(torch.tensor(0.8))

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_genres)
        )
        self.residual_proj = nn.Linear(512, num_genres, bias=False)

    def _encode_image(self, x):
        vt  = self.encoder.transformer
        x   = self.encoder.conv1(x)
        cls = self.encoder.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x.flatten(2).transpose(1, 2)], dim=1)
        x   = x + self.encoder.positional_embedding
        x   = self.encoder.ln_pre(x)
        x   = x.permute(1, 0, 2)
        x   = vt(x)
        x   = x.permute(1, 0, 2)

        cls_feat   = x[:, 0, :]
        patch_feat = x[:, 1:, :].mean(dim=1)
        feat       = (1 - self.patch_weight) * cls_feat + self.patch_weight * patch_feat
        feat       = self.encoder.ln_post(feat)
        if self.encoder.proj is not None:
            feat = feat @ self.encoder.proj
        return feat.float()

    def forward(self, x):
        img_feat = self._encode_image(x)
        img_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim  = img_norm @ self.text_emb.T
        mlp_out  = self.head(img_feat) + self.residual_proj(img_feat)
        alpha    = self.alpha.clamp(0.5, 0.95)
        return alpha * mlp_out + (1 - alpha) * cos_sim


# ── Global state ─────────────────────────────────────────────────────────────
model_state: dict = {}


def load_model():
    """Load CLIP + fine-tuned weights. Called once at startup."""
    logger.info(f"Loading model on device: {DEVICE}")

    clip_model, _ = clip.load('ViT-B/16', device=DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Build text embeddings
    with torch.no_grad():
        all_embeddings = []
        for genre in KEEP_GENRES:
            tokens  = clip.tokenize(GENRE_PROMPTS[genre]).to(DEVICE)
            embs    = clip_model.encode_text(tokens).float()
            embs    = embs / embs.norm(dim=-1, keepdim=True)
            avg_emb = embs.mean(dim=0)
            avg_emb = avg_emb / avg_emb.norm()
            all_embeddings.append(avg_emb)
        text_embeddings = torch.stack(all_embeddings)

    model = AnimeGenreClassifier(
        clip_visual  = clip_model.visual.float(),
        text_emb     = text_embeddings.to(DEVICE).float(),
        num_genres   = NUM_GENRES,
        dropout      = 0.4,
        patch_weight = 0.3,
    ).to(DEVICE)

    # Load fine-tuned weights if checkpoint exists
    ckpt_path = os.environ.get('MODEL_CHECKPOINT_PATH', 'best_phase2.pt')
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get('model_state', ckpt)
        # Strip DataParallel prefix if present
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        logger.info(f"Checkpoint loaded (Val F1: {ckpt.get('val_f1', 'N/A')})")
    else:
        logger.warning(
            f"No checkpoint found at '{ckpt_path}'. "
            "Running with random head weights (CLIP encoder is intact). "
            "Set MODEL_CHECKPOINT_PATH env var to your .pt file."
        )

    model.eval()
    model.float()

    # Load CV-tuned thresholds if available
    thresh_path = os.environ.get('THRESHOLDS_PATH', 'best_thresholds_cv.npy')
    if os.path.exists(thresh_path):
        thresholds = np.load(thresh_path)
        logger.info(f"Loaded CV thresholds from {thresh_path}")
    else:
        thresholds = np.full(NUM_GENRES, DEFAULT_THRESHOLD)
        logger.info(f"Using default threshold {DEFAULT_THRESHOLD} for all genres")

    model_state['model']      = model
    model_state['thresholds'] = thresholds
    logger.info("Model ready.")


# ── TTA transforms ────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ]),
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.25)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ]),
    transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ]),
]


@torch.no_grad()
def predict(image: Image.Image) -> list[dict]:
    """Run TTA inference, return list of {name, confidence} dicts."""
    model      = model_state['model']
    thresholds = model_state['thresholds']

    avg_probs = np.zeros(NUM_GENRES, dtype=np.float32)
    for tfm in tta_transforms:
        tensor = tfm(image).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]
        avg_probs += probs
    avg_probs /= len(tta_transforms)

    results = []
    for i, genre in enumerate(KEEP_GENRES):
        results.append({
            "name"      : genre,
            "confidence": float(round(float(avg_probs[i]), 4)),
            "threshold" : float(thresholds[i]),
        })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    model_state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Anime Genre Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your Vercel domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class GenrePrediction(BaseModel):
    name      : str
    confidence: float

class ClassifyResponse(BaseModel):
    genres: List[GenrePrediction]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": bool(model_state)}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPG or PNG image."
        )

    contents = await file.read()

    # 5 MB guard
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        raw_results = predict(image)
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Return only genres above per-genre threshold; always return at least top 1
    above = [r for r in raw_results if r["confidence"] >= r["threshold"]]
    if not above:
        above = raw_results[:1]

    genres = [GenrePrediction(name=r["name"], confidence=r["confidence"])
              for r in above]

    return ClassifyResponse(genres=genres)
