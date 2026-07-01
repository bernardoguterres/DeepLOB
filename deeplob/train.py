"""Training loop and experiment management for DeepLOB.

Entry point::

    python -m deeplob.train --config configs/default.yaml --k 1

Full pipeline: load config → seed → get_dataloaders → DeepLOB → Adam
→ epoch loop (train + validate + early stopping) → checkpoint best model.
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplob.dataset import get_dataloaders
from deeplob.model import DeepLOB
from deeplob.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed

logger = logging.getLogger(__name__)

__all__ = ["train_one_epoch", "validate", "train"]


def _reconstruct_no_improve(log_path: Path, best_val_f1: float) -> int:
    """Count consecutive epochs without improvement at the end of a training log.

    Replays the JSONL log in order, tracking the running best val_f1.  Used
    when resuming a checkpoint so that the early-stopping counter is restored
    correctly rather than reset to zero.

    Args:
        log_path: Path to the ``training_log_k{k}.jsonl`` file.
        best_val_f1: The best val_f1 stored in the checkpoint (used as the
            floor so the counter cannot be under-counted).

    Returns:
        Number of consecutive epochs at the tail of the log where val_f1
        did not exceed the running best.  Returns 0 on any read/parse error.
    """
    no_improve = 0
    running_best = -1.0
    try:
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["val_f1"] > running_best:
                    running_best = entry["val_f1"]
                    no_improve = 0
                else:
                    no_improve += 1
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning(
            "Could not reconstruct no_improve from %s: %s — resetting to 0", log_path, exc
        )
        no_improve = 0
    return no_improve


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Iterates over all batches in *loader*, computes loss, back-propagates,
    and steps the optimiser. A tqdm progress bar shows per-batch loss.

    Args:
        model: DeepLOB model (will be set to train mode).
        loader: Training DataLoader.
        optimizer: Optimiser.
        criterion: Loss function (e.g. ``CrossEntropyLoss`` with class weights).
        device: Compute device.

    Returns:
        Mean loss averaged over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
            # Brief inter-batch pause every 50 batches to break up sustained GPU load
            if i % 50 == 0:
                time.sleep(0.05)
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a validation or test set.

    Args:
        model: DeepLOB model (will be set to eval mode).
        loader: DataLoader (test or validation split).
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of ``(mean_loss, macro_f1)`` where *macro_f1* is computed with
        :func:`sklearn.metrics.f1_score` (``average="macro"``).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    mean_loss = total_loss / n_batches if n_batches > 0 else 0.0
    macro_f1: float = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return mean_loss, macro_f1


def train(
    config_path: str,
    data_dir: str,
    k: int,
    output_dir: str = "outputs/",
) -> None:
    """Full training pipeline for a single prediction horizon.

    Loads config, data, and model; runs the training loop with early stopping
    on validation macro F1; saves the best checkpoint to *output_dir*.

    Progress is logged as JSON lines to ``outputs/training_log_k{k}.jsonl``::

        {"epoch": 1, "train_loss": 0.92, "val_loss": 0.89, "val_f1": 0.41}

    Args:
        config_path: Path to ``default.yaml`` (or compatible config).
        data_dir: Path to FI-2010 ``.npy`` files.
        k: Prediction horizon. One of [1, 2, 3, 5, 10].
        output_dir: Directory for checkpoints and log files (created if absent).
    """
    # ── 1. Bootstrap ────────────────────────────────────────────────────────
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device()

    # ── 2. Data ──────────────────────────────────────────────────────────────
    training_cfg = config["training"]
    train_loader, test_loader, class_weights = get_dataloaders(
        data_dir=data_dir,
        k=k,
        batch_size=training_cfg["batch_size"],
        window=training_cfg.get("window", 100),
        train_days=training_cfg.get("train_days", 7),
    )

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model_cfg = config.get("model", {})
    model = DeepLOB(
        hidden_size=model_cfg.get("hidden_size", 256),
        num_lstm_layers=model_cfg.get("lstm_layers", 1),
    ).to(device)

    # ── 4. Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── 5. Optimiser ─────────────────────────────────────────────────────────
    adam_eps = training_cfg.get("adam_eps", 1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"], eps=adam_eps)

    # ── 6. Output paths ──────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"training_log_k{k}.jsonl"
    ckpt_path = str(out / f"best_model_k{k}.pt")

    n_epochs = training_cfg.get("epochs", 50)
    patience = training_cfg.get("patience", 10)
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    start_epoch = 1

    # ── 6b. Resume from checkpoint if one already exists ─────────────────────
    if Path(ckpt_path).exists():
        resumed_epoch, best_val_f1 = load_checkpoint(ckpt_path, model, optimizer)
        best_epoch = resumed_epoch
        no_improve = _reconstruct_no_improve(log_path, best_val_f1)
        start_epoch = resumed_epoch + 1
        logger.info(
            "Resuming k=%d from epoch %d (best val_f1=%.4f, no_improve=%d/%d)",
            k,
            resumed_epoch,
            best_val_f1,
            no_improve,
            patience,
        )
    else:
        logger.info("Starting k=%d from scratch.", k)

    # ── 7. Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = validate(model, test_loader, criterion, device)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f",
            epoch,
            n_epochs,
            train_loss,
            val_loss,
            val_f1,
        )

        # JSONL log
        try:
            with log_path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": round(train_loss, 6),
                            "val_loss": round(val_loss, 6),
                            "val_f1": round(val_f1, 6),
                        }
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.warning("Failed to write training log to %s: %s", log_path, exc)

        if device.type == "mps":
            time.sleep(2)  # thermal recovery between epochs on Apple Silicon

        # Checkpoint best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_f1, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience
                )
                break

    # ── 8. Summary ───────────────────────────────────────────────────────────
    logger.info("Training complete. Best val F1: %.4f at epoch %d", best_val_f1, best_epoch)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train DeepLOB for a single horizon.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data_dir", default="data/raw/", help="Path to FI-2010 .npy files")
    parser.add_argument("--k", type=int, required=True, help="Prediction horizon: 1,2,3,5,10")
    parser.add_argument("--output_dir", default="outputs/", help="Output directory")
    args = parser.parse_args()
    train(args.config, args.data_dir, args.k, args.output_dir)
