"""Minimal inference server for DeepLOB predictions."""

import logging
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# Ensure the DeepLOB repo root is importable regardless of invocation method.
sys.path.insert(0, str(Path(__file__).parent.parent))

from deeplob.model import DeepLOB  # noqa: E402
from deeplob.utils import get_device  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Global inference state (set before uvicorn.run() in __main__)
# ---------------------------------------------------------------------------
_model: Optional[DeepLOB] = None
_scaler = None
_device: Optional[torch.device] = None
_k: int = 10
_checkpoint_dir: str = "outputs/"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    lob_snapshot: list[float]

    @field_validator("lob_snapshot")
    @classmethod
    def validate_length(cls, v: list[float]) -> list[float]:
        if len(v) != 40:
            raise ValueError(f"lob_snapshot must contain exactly 40 features, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    direction: int
    confidence: float
    probabilities: list[float]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_from_dir(k: int, checkpoint_dir: str) -> None:
    global _model, _scaler, _device

    ckpt_path = Path(checkpoint_dir) / f"best_model_k{k}.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    _device = get_device()

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Infer hidden_size from the saved FC weight (num_classes × hidden_size).
    hidden_size = ckpt["model_state"]["fc.weight"].shape[1]
    _model = DeepLOB(hidden_size=hidden_size)
    _model.load_state_dict(ckpt["model_state"])
    _model.to(_device)
    _model.eval()

    logger.info(
        "Loaded DeepLOB k=%d from %s (epoch=%d val_f1=%.4f) on %s",
        k,
        ckpt_path,
        ckpt.get("epoch", -1),
        ckpt.get("val_f1", 0.0),
        _device,
    )

    scaler_path = Path(checkpoint_dir) / f"scaler_k{k}.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        logger.info("Loaded StandardScaler from %s", scaler_path)
    else:
        logger.warning(
            "No StandardScaler found at %s — inputs expected pre-normalized (FI-2010 z-score)",
            scaler_path,
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_from_dir(_k, _checkpoint_dir)
    yield


app = FastAPI(title="DeepLOB Inference Server", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Get mid-price direction prediction for a single LOB snapshot.

    Accepts 40 LOB features in FI-2010 order. Tiles the snapshot into
    the 100-step window the model expects, then returns the predicted
    direction class and softmax confidence.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.monotonic()

    features = np.array(request.lob_snapshot, dtype=np.float32)  # (40,)

    if _scaler is not None:
        features = _scaler.transform(features.reshape(1, -1)).reshape(-1).astype(np.float32)

    # Tile single snapshot to fill the 100-event window the model requires.
    window = np.tile(features, (100, 1))  # (100, 40)
    x = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).to(_device)  # (1, 1, 100, 40)

    with torch.no_grad():
        logits = _model(x)  # (1, 3)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (3,)

    probs_list: list[float] = probs.cpu().tolist()
    direction = int(torch.argmax(probs).item())
    confidence = float(probs[direction].item())
    latency_ms = (time.monotonic() - t0) * 1000.0

    logger.info(
        "predict: direction=%d confidence=%.4f latency_ms=%.2f",
        direction,
        confidence,
        latency_ms,
    )

    return PredictResponse(
        direction=direction,
        confidence=confidence,
        probabilities=probs_list,
    )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": f"deeplob_k{_k}",
        "device": str(_device) if _device is not None else "unknown",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepLOB inference server")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="outputs/")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    _k = args.k
    _checkpoint_dir = args.checkpoint_dir

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
