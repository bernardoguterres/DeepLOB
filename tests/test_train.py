"""Tests for deeplob.train and deeplob.utils.

All tests use synthetic data and small models — the FI-2010 dataset is not
required. Training tests are designed to complete in seconds on CPU.

Tests are flat functions following the ``test_<what>_<condition>`` convention.
"""

import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from deeplob.model import DeepLOB
from deeplob.train import train, train_one_epoch
from deeplob.utils import get_device, load_checkpoint, save_checkpoint, set_seed

# ---------------------------------------------------------------------------
# 1. Training loop — loss decreases with gradient descent
# ---------------------------------------------------------------------------


def test_loss_decreases_over_epochs(tiny_loaders, tiny_model):
    """Training loss at epoch 5 must be strictly lower than at epoch 1.

    Uses a learning rate of 0.01 so the over-parameterised network (62k params,
    200 samples) overfits within a few epochs, regardless of label randomness.
    """
    train_loader, _, class_weights = tiny_loaders
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.01)

    losses = []
    for _ in range(5):
        loss = train_one_epoch(tiny_model, train_loader, optimizer, criterion, device)
        losses.append(loss)

    assert losses[4] < losses[0], (
        f"Loss did not decrease over 5 epochs: " f"epoch 1={losses[0]:.4f}, epoch 5={losses[4]:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Checkpointing — round-trip save → load preserves all state
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load(tmp_path):
    """Loaded model must have identical weights to the saved model."""
    model = DeepLOB(hidden_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = str(tmp_path / "test.pt")

    save_checkpoint(model, optimizer, epoch=5, val_f1=0.75, path=ckpt_path)

    fresh_model = DeepLOB(hidden_size=4)
    fresh_optim = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    epoch_loaded, f1_loaded = load_checkpoint(ckpt_path, fresh_model, fresh_optim)

    # Metadata round-trips exactly
    assert epoch_loaded == 5
    assert f1_loaded == 0.75

    # All state-dict keys are present in the loaded model
    assert set(model.state_dict().keys()) == set(fresh_model.state_dict().keys())

    # All tensors match after round-trip (float: allclose; int: equal)
    for key, param in model.state_dict().items():
        fresh_param = fresh_model.state_dict()[key]
        if param.is_floating_point():
            assert torch.allclose(param, fresh_param), f"Float mismatch after load: {key}"
        else:
            assert torch.equal(param, fresh_param), f"Int tensor mismatch after load: {key}"


# ---------------------------------------------------------------------------
# 3. Device selection — returns a valid torch.device
# ---------------------------------------------------------------------------


def test_get_device_returns_valid_device():
    """get_device() must return a torch.device with a recognised type string."""
    device = get_device()

    assert isinstance(device, torch.device), f"Expected torch.device, got {type(device)}"
    assert str(device) in {
        "cpu",
        "cuda",
        "mps",
    }, f"Device type '{device}' is not one of {{'cpu', 'cuda', 'mps'}}"


# ---------------------------------------------------------------------------
# 4. Training log — JSONL file is written with one entry per epoch
# ---------------------------------------------------------------------------


def test_training_log_written_to_jsonl(tmp_path, tiny_loaders):
    """train() must create a JSONL log with one valid JSON object per epoch.

    get_dataloaders is patched so the test runs without FI-2010 files.
    hidden_size=4 keeps the model tiny for speed.
    """
    config_content = """\
seed: 42
model:
  hidden_size: 4
  lstm_layers: 1
training:
  lr: 0.001
  batch_size: 32
  window: 100
  train_days: 7
  epochs: 2
  patience: 10
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    output_dir = str(tmp_path / "outputs")
    train_loader, test_loader, class_weights = tiny_loaders

    with patch(
        "deeplob.train.get_dataloaders",
        return_value=(train_loader, test_loader, class_weights),
    ):
        train(str(config_path), "unused_data_dir", k=1, output_dir=output_dir)

    log_path = Path(output_dir) / "training_log_k1.jsonl"
    assert log_path.exists(), "training_log_k1.jsonl was not created in output_dir"

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2, f"Expected 2 log lines (one per epoch), got {len(lines)}"

    required_keys = {"epoch", "train_loss", "val_loss", "val_f1"}
    for i, line in enumerate(lines, 1):
        record = json.loads(line)
        missing = required_keys - record.keys()
        assert not missing, f"Log line {i} is missing keys: {missing}  line={line}"


# ---------------------------------------------------------------------------
# 5. Early stopping — counter logic fires at the right epoch
# ---------------------------------------------------------------------------


def test_early_stopping_triggers():
    """Early stopping must fire at epoch 5 for patience=3.

    Sequence: [0.60, 0.65, 0.65, 0.65, 0.65]
    Best improves at epoch 2 (0.65), then no improvement for epochs 3, 4, 5.
    With patience=3 the loop breaks at epoch 5.

    Tests the early-stopping *logic* directly (mirrors the implementation in
    deeplob/train.py) without invoking the full train() pipeline.
    """
    val_f1_seq = [0.60, 0.65, 0.65, 0.65, 0.65]
    patience = 3

    best_val_f1 = -1.0
    no_improve = 0
    stopped_at = None

    for epoch, val_f1 in enumerate(val_f1_seq, 1):
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stopped_at = epoch
                break

    assert stopped_at == 5, (
        f"Expected early stopping at epoch 5 "
        f"(patience=3, last improvement at epoch 2), "
        f"got stopped_at={stopped_at}"
    )


# ---------------------------------------------------------------------------
# 6. Seeding — same seed produces identical tensors
# ---------------------------------------------------------------------------


def test_set_seed_reproducibility():
    """Generating tensors with the same seed twice must yield identical results."""
    set_seed(42)
    A = torch.randn(10, 10)

    set_seed(42)
    B = torch.randn(10, 10)

    assert torch.equal(A, B), (
        "Tensors A and B differ despite being generated with the same seed — "
        "set_seed() is not fully reproducible."
    )
