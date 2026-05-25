"""Shared utilities: config loading, seeding, device selection, checkpointing."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

__all__ = [
    "load_config",
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
]


def load_config(path: str) -> dict:
    """Load YAML configuration file.

    Args:
        path: Path to YAML config file.

    Returns:
        Config as nested dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as valid YAML.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with p.open() as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse config file {path}: {exc}") from exc


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for: Python ``random``, NumPy, PyTorch CPU, and PyTorch CUDA
    (all devices). Also sets ``torch.backends.cudnn.deterministic = True``.

    Args:
        seed: Random seed (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


def get_device() -> torch.device:
    """Select best available compute device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.
    Prints the selected device name.

    Returns:
        :class:`torch.device` instance.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    path: str,
) -> None:
    """Save model checkpoint to disk.

    Creates parent directories automatically if they do not exist.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        val_f1: Validation macro F1 at this checkpoint.
        path: File path for the checkpoint (``.pt`` file).

    Raises:
        OSError: If the checkpoint cannot be written to *path*.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
            },
            path,
        )
    except OSError as exc:
        raise OSError(f"Failed to save checkpoint to {path}: {exc}") from exc


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    """Load checkpoint and restore model and optimizer state in-place.

    The checkpoint is always mapped to CPU first so it can be loaded on any
    device; the caller is responsible for moving the model afterwards.

    Args:
        path: Path to checkpoint file.
        model: Model to restore state into.
        optimizer: Optimizer to restore state into.

    Returns:
        Tuple of ``(epoch, val_f1)`` from the checkpoint.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the checkpoint file is corrupt or incompatible.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["epoch"]), float(ckpt["val_f1"])
