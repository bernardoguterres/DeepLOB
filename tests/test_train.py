"""Tests for deeplob.train and deeplob.utils.

All tests use synthetic data and small models — the FI-2010 dataset is not
required. Training tests are designed to complete in seconds on CPU.

Tests are flat functions following the ``test_<what>_<condition>`` convention.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from deeplob.model import DeepLOB
from deeplob.train import (
    _append_jsonl_log,
    _reconstruct_no_improve,
    _resume_or_start,
    EpochResult,
    train,
    train_one_epoch,
)
from deeplob.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed

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


# ---------------------------------------------------------------------------
# 7. load_config — missing file raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_load_config_missing_file():
    """load_config must raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# 8. load_checkpoint — missing file raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_load_checkpoint_missing_file():
    """load_checkpoint must raise FileNotFoundError for a non-existent path."""
    model = DeepLOB(hidden_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint("/nonexistent/path/model.pt", model, optimizer)


# ---------------------------------------------------------------------------
# 9. get_device — falls back to CPU when no accelerators are available
# ---------------------------------------------------------------------------


def test_get_device_falls_back_to_cpu():
    """get_device returns torch.device('cpu') when MPS and CUDA are both unavailable."""
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        device = get_device()
    assert str(device) == "cpu", f"Expected 'cpu', got '{device}'"


# ---------------------------------------------------------------------------
# 10. get_device — returns CUDA when CUDA is available and MPS is not
# ---------------------------------------------------------------------------


def test_get_device_uses_cuda_when_available():
    """get_device returns torch.device('cuda') when CUDA is available and MPS is not."""
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=True),
    ):
        device = get_device()
    assert str(device) == "cuda", f"Expected 'cuda', got '{device}'"


# ---------------------------------------------------------------------------
# 11. _reconstruct_no_improve — replays a JSONL log to rebuild the counter
# ---------------------------------------------------------------------------


def test_reconstruct_no_improve_counts_tail_epochs(tmp_path):
    """Counter must count consecutive epochs at the tail that didn't beat the running best."""
    log_path = tmp_path / "log.jsonl"
    # val_f1 improves at epoch 1 and 2, then plateaus for epochs 3 and 4.
    entries = [
        {"epoch": 1, "val_f1": 0.50},
        {"epoch": 2, "val_f1": 0.60},
        {"epoch": 3, "val_f1": 0.55},
        {"epoch": 4, "val_f1": 0.60},
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    no_improve = _reconstruct_no_improve(log_path, best_val_f1=0.60)

    assert no_improve == 2, f"Expected 2 consecutive non-improving epochs, got {no_improve}"


def test_reconstruct_no_improve_skips_blank_lines(tmp_path):
    """Blank lines in the log must be skipped rather than raising."""
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        json.dumps({"epoch": 1, "val_f1": 0.5})
        + "\n\n"
        + json.dumps({"epoch": 2, "val_f1": 0.4})
        + "\n"
    )

    no_improve = _reconstruct_no_improve(log_path, best_val_f1=0.5)

    assert no_improve == 1, f"Expected 1 non-improving epoch, got {no_improve}"


def test_reconstruct_no_improve_returns_zero_on_corrupt_log(tmp_path):
    """A corrupt (non-JSON) log line must cause the function to return 0, not raise."""
    log_path = tmp_path / "log.jsonl"
    log_path.write_text("{not valid json\n")

    no_improve = _reconstruct_no_improve(log_path, best_val_f1=0.5)

    assert no_improve == 0


def test_reconstruct_no_improve_returns_zero_when_missing_file(tmp_path):
    """A log path that doesn't exist must cause the function to return 0, not raise."""
    no_improve = _reconstruct_no_improve(tmp_path / "does_not_exist.jsonl", best_val_f1=0.5)
    assert no_improve == 0


# ---------------------------------------------------------------------------
# 12. _resume_or_start — fresh start vs resuming from checkpoint
# ---------------------------------------------------------------------------


def test_resume_or_start_fresh_when_no_checkpoint(tmp_path):
    """When no checkpoint exists, resume state must be fresh-start defaults."""
    model = DeepLOB(hidden_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_path = str(tmp_path / "missing.pt")
    log_path = tmp_path / "log.jsonl"

    resume = _resume_or_start(ckpt_path, log_path, model, optimizer, k=1, patience=10)

    assert resume.start_epoch == 1
    assert resume.best_val_f1 == -1.0
    assert resume.best_epoch == 0
    assert resume.no_improve == 0


def test_resume_or_start_resumes_from_existing_checkpoint(tmp_path):
    """When a checkpoint exists, resume state must reflect its saved epoch/F1 plus the log."""
    model = DeepLOB(hidden_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, optimizer, epoch=3, val_f1=0.70, path=ckpt_path)

    log_path = tmp_path / "log.jsonl"
    entries = [
        {"epoch": 1, "val_f1": 0.60},
        {"epoch": 2, "val_f1": 0.65},
        {"epoch": 3, "val_f1": 0.70},
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    fresh_model = DeepLOB(hidden_size=4)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters())

    resume = _resume_or_start(ckpt_path, log_path, fresh_model, fresh_optimizer, k=1, patience=10)

    assert resume.start_epoch == 4, f"Expected start_epoch=4 (resumed_epoch+1), got {resume.start_epoch}"
    assert resume.best_val_f1 == pytest.approx(0.70)
    assert resume.best_epoch == 3
    assert resume.no_improve == 0, "Best epoch is the last logged epoch → no_improve must be 0"


# ---------------------------------------------------------------------------
# 13. _append_jsonl_log — swallows I/O errors instead of raising
# ---------------------------------------------------------------------------


def test_append_jsonl_log_does_not_raise_on_oserror(tmp_path):
    """A write failure must be logged as a warning, not propagated as an exception."""
    log_path = tmp_path / "log.jsonl"
    result = EpochResult(
        train_loss=0.5,
        val_loss=0.4,
        val_f1=0.6,
        best_val_f1=0.6,
        best_epoch=1,
        no_improve=0,
        should_stop=False,
    )

    with patch.object(Path, "open", side_effect=OSError("disk full")):
        _append_jsonl_log(log_path, epoch=1, result=result)  # must not raise


# ---------------------------------------------------------------------------
# 14. train — early stopping halts the epoch loop before n_epochs
# ---------------------------------------------------------------------------


def test_train_stops_early_when_patience_exhausted(tmp_path, tiny_loaders):
    """With patience=1 and a val_f1 that never improves after epoch 1, training
    must stop well before the configured 10 epochs.
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
  epochs: 10
  patience: 1
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    output_dir = str(tmp_path / "outputs")
    train_loader, test_loader, class_weights = tiny_loaders

    with (
        patch(
            "deeplob.train.get_dataloaders",
            return_value=(train_loader, test_loader, class_weights),
        ),
        # Constant val_f1: epoch 1 "improves" over the initial -1.0 floor,
        # every epoch after that fails to improve → early stop after epoch 2.
        patch("deeplob.train.validate", return_value=(0.3, 0.55)),
    ):
        train(str(config_path), "unused_data_dir", k=1, output_dir=output_dir)

    log_path = Path(output_dir) / "training_log_k1.jsonl"
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2, (
        f"Expected exactly 2 epochs before early stopping (patience=1), got {len(lines)}"
    )


# ---------------------------------------------------------------------------
# 15. load_config — invalid YAML raises ValueError
# ---------------------------------------------------------------------------


def test_load_config_invalid_yaml_raises_valueerror(tmp_path):
    """Malformed YAML must be wrapped in a ValueError, not raise a raw YAMLError."""
    config_path = tmp_path / "bad.yaml"
    # Unbalanced flow mapping brace — triggers yaml.YAMLError on parse.
    config_path.write_text("training: {lr: 0.01, batch_size: 32\n")

    with pytest.raises(ValueError, match="Failed to parse config file"):
        load_config(str(config_path))


# ---------------------------------------------------------------------------
# 16. save_checkpoint — write failure raises OSError with context
# ---------------------------------------------------------------------------


def test_save_checkpoint_raises_oserror_on_write_failure(tmp_path):
    """A torch.save failure must be wrapped in an OSError naming the target path."""
    model = DeepLOB(hidden_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_path = str(tmp_path / "ckpt.pt")

    with patch("deeplob.utils.torch.save", side_effect=OSError("no space left on device")):
        with pytest.raises(OSError, match="Failed to save checkpoint"):
            save_checkpoint(model, optimizer, epoch=1, val_f1=0.5, path=ckpt_path)


# ---------------------------------------------------------------------------
# 17. load_checkpoint — corrupt/incompatible checkpoint raises RuntimeError
# ---------------------------------------------------------------------------


def test_load_checkpoint_raises_runtimeerror_on_corrupt_file(tmp_path):
    """A torch.load failure on an existing file must be wrapped in a RuntimeError."""
    model = DeepLOB(hidden_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_path = tmp_path / "corrupt.pt"
    ckpt_path.write_bytes(b"not a real checkpoint")

    with patch(
        "deeplob.utils.torch.load", side_effect=RuntimeError("PytorchStreamReader failed")
    ):
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_checkpoint(str(ckpt_path), model, optimizer)
