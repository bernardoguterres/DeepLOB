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
from typing import NamedTuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplob.dataset import get_dataloaders
from deeplob.model import DeepLOB
from deeplob.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed

logger = logging.getLogger(__name__)

__all__ = ["train_one_epoch", "validate", "run_epoch", "EpochResult", "train"]


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


class EpochResult(NamedTuple):
    """Outcome of one call to :func:`run_epoch`."""

    train_loss: float
    val_loss: float
    val_f1: float
    best_val_f1: float
    best_epoch: int
    no_improve: int
    should_stop: bool


def run_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    best_val_f1: float,
    best_epoch: int,
    no_improve: int,
    patience: int,
    ckpt_path: str,
) -> EpochResult:
    """Run one train+validate epoch, checkpoint on improvement, and check early-stopping.

    Shared by :func:`train` and :func:`~deeplob.ablation.run_ablation` so the
    checkpoint/early-stopping/thermal-pause logic exists in exactly one place.

    Args:
        model: DeepLOB (or ablation variant) model.
        train_loader: Training DataLoader.
        test_loader: Validation/test DataLoader.
        optimizer: Optimiser.
        criterion: Loss function.
        device: Compute device.
        epoch: Current epoch number (1-indexed).
        best_val_f1: Best macro F1 seen so far.
        best_epoch: Epoch at which ``best_val_f1`` was achieved.
        no_improve: Consecutive epochs without improvement so far.
        patience: Epochs to tolerate without improvement before stopping.
        ckpt_path: Path to write the checkpoint to on improvement.

    Returns:
        :class:`EpochResult` with the epoch's metrics and updated early-stopping state.
    """
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_f1 = validate(model, test_loader, criterion, device)

    if device.type == "mps":
        time.sleep(2)  # thermal recovery between epochs on Apple Silicon

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        no_improve = 0
        save_checkpoint(model, optimizer, epoch, val_f1, ckpt_path)
        should_stop = False
    else:
        no_improve += 1
        should_stop = no_improve >= patience

    return EpochResult(
        train_loss=train_loss,
        val_loss=val_loss,
        val_f1=val_f1,
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        no_improve=no_improve,
        should_stop=should_stop,
    )


def _build_training_setup(
    config: dict,
    data_dir: str,
    k: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, nn.Module, torch.optim.Optimizer, nn.Module]:
    """Build dataloaders, model, criterion, and optimiser from a loaded config.

    Args:
        config: Parsed YAML config (``load_config`` output).
        data_dir: Path to FI-2010 ``.npy`` files.
        k: Prediction horizon.
        device: Compute device the model is moved to.

    Returns:
        Tuple of ``(train_loader, test_loader, model, optimizer, criterion)``.
    """
    training_cfg = config["training"]
    train_loader, test_loader, class_weights = get_dataloaders(
        data_dir=data_dir,
        k=k,
        batch_size=training_cfg["batch_size"],
        window=training_cfg.get("window", 100),
        train_days=training_cfg.get("train_days", 7),
    )

    model_cfg = config.get("model", {})
    model = DeepLOB(
        hidden_size=model_cfg.get("hidden_size", 256),
        num_lstm_layers=model_cfg.get("lstm_layers", 1),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Default of 1.0 matches the paper's epsilon (Zhang et al. 2019), not
    # PyTorch's own Adam default of 1e-8 — configs should still set this
    # explicitly, but the fallback shouldn't silently diverge from the paper.
    adam_eps = training_cfg.get("adam_eps", 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"], eps=adam_eps)

    return train_loader, test_loader, model, optimizer, criterion


class _ResumeState(NamedTuple):
    """Early-stopping state to resume from, or fresh-start defaults."""

    start_epoch: int
    best_val_f1: float
    best_epoch: int
    no_improve: int


def _resume_or_start(
    ckpt_path: str,
    log_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    k: int,
    patience: int,
) -> _ResumeState:
    """Resume training state from an existing checkpoint, or start fresh.

    Args:
        ckpt_path: Path to ``best_model_k{k}.pt``.
        log_path: Path to ``training_log_k{k}.jsonl`` (read to rebuild the
            early-stopping counter when resuming).
        model: Model to load the checkpoint's weights into, if resuming.
        optimizer: Optimiser to load the checkpoint's state into, if resuming.
        k: Prediction horizon (for logging only).
        patience: Early-stopping patience (for logging only).

    Returns:
        :class:`_ResumeState` with the epoch/F1/counter to continue from.
    """
    if not Path(ckpt_path).exists():
        logger.info("Starting k=%d from scratch.", k)
        return _ResumeState(start_epoch=1, best_val_f1=-1.0, best_epoch=0, no_improve=0)

    resumed_epoch, best_val_f1 = load_checkpoint(ckpt_path, model, optimizer)
    no_improve = _reconstruct_no_improve(log_path, best_val_f1)
    logger.info(
        "Resuming k=%d from epoch %d (best val_f1=%.4f, no_improve=%d/%d)",
        k,
        resumed_epoch,
        best_val_f1,
        no_improve,
        patience,
    )
    return _ResumeState(
        start_epoch=resumed_epoch + 1,
        best_val_f1=best_val_f1,
        best_epoch=resumed_epoch,
        no_improve=no_improve,
    )


def _append_jsonl_log(log_path: Path, epoch: int, result: EpochResult) -> None:
    """Append one epoch's metrics as a JSON line, warning (not raising) on I/O failure."""
    try:
        with log_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": round(result.train_loss, 6),
                        "val_loss": round(result.val_loss, 6),
                        "val_f1": round(result.val_f1, 6),
                    }
                )
                + "\n"
            )
    except OSError as exc:
        logger.warning("Failed to write training log to %s: %s", log_path, exc)


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
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device()

    train_loader, test_loader, model, optimizer, criterion = _build_training_setup(
        config, data_dir, k, device
    )

    training_cfg = config["training"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"training_log_k{k}.jsonl"
    ckpt_path = str(out / f"best_model_k{k}.pt")

    n_epochs = training_cfg.get("epochs", 50)
    patience = training_cfg.get("patience", 10)
    resume = _resume_or_start(ckpt_path, log_path, model, optimizer, k, patience)
    best_val_f1, best_epoch, no_improve = resume.best_val_f1, resume.best_epoch, resume.no_improve

    for epoch in range(resume.start_epoch, n_epochs + 1):
        result = run_epoch(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            device,
            epoch,
            best_val_f1,
            best_epoch,
            no_improve,
            patience,
            ckpt_path,
        )
        best_val_f1, best_epoch, no_improve = (
            result.best_val_f1,
            result.best_epoch,
            result.no_improve,
        )

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f",
            epoch,
            n_epochs,
            result.train_loss,
            result.val_loss,
            result.val_f1,
        )
        _append_jsonl_log(log_path, epoch, result)

        if result.should_stop:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience
            )
            break

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
