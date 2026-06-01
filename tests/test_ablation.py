"""Tests for deeplob.ablation.

All tests use synthetic tensors — no FI-2010 data required.
"""

import json
from pathlib import Path
from unittest.mock import patch

import torch

from deeplob.ablation import CNNInceptionModel, CNNOnlyModel, _ablation_table, run_ablation
from deeplob.model import DeepLOB

# ---------------------------------------------------------------------------
# 1. All three ablation models produce correct output shape
# ---------------------------------------------------------------------------


def test_ablation_models_output_shape(batch):
    """CNNOnlyModel, CNNInceptionModel, and DeepLOB must all output (B, 3) logits.

    Uses the standard ``batch`` fixture: shape ``(8, 1, 100, 40)``.
    """
    models = [
        ("CNNOnlyModel", CNNOnlyModel()),
        ("CNNInceptionModel", CNNInceptionModel()),
        ("DeepLOB", DeepLOB(hidden_size=16, num_lstm_layers=1)),
    ]

    for name, model in models:
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (8, 3), f"{name}: expected output shape (8, 3), got {out.shape}"
        assert out.dtype == torch.float32, f"{name}: expected float32 output, got {out.dtype}"


# ---------------------------------------------------------------------------
# 2. _ablation_table — correct Markdown and delta computation
# ---------------------------------------------------------------------------


def test_ablation_table_all_variants():
    """_ablation_table renders correct rows and percentage deltas for all variants."""
    results = {"CNN only": 0.60, "CNN + Inception": 0.70, "Full DeepLOB": 0.75}
    table = _ablation_table(results)

    assert "CNN only" in table
    assert "CNN + Inception" in table
    assert "Full DeepLOB" in table
    # Full DeepLOB row must use "—" as the delta
    assert "—" in table
    # CNN only: (0.60 - 0.75) / 0.75 * 100 = -20.0%
    assert "-20.0%" in table
    # CNN + Inception: (0.70 - 0.75) / 0.75 * 100 ≈ -6.7%
    assert "-6.7%" in table


def test_ablation_table_missing_full_model():
    """_ablation_table handles a missing 'Full DeepLOB' key (full_f1 defaults to 0)."""
    results = {"CNN only": 0.60}
    table = _ablation_table(results)
    assert "CNN only" in table
    # full_f1 = 0.0 → delta computed as 0.0%
    assert "+0.0%" in table


def test_ablation_table_positive_delta():
    """_ablation_table formats positive delta with a leading '+' sign."""
    # If an ablation variant somehow beats Full DeepLOB
    results = {"CNN only": 0.80, "Full DeepLOB": 0.75}
    table = _ablation_table(results)
    # (0.80 - 0.75) / 0.75 * 100 ≈ +6.7%
    assert "+6.7%" in table


# ---------------------------------------------------------------------------
# 3. run_ablation — integration test with mocked training functions
# ---------------------------------------------------------------------------


def test_run_ablation_saves_json(tmp_path, tiny_loaders):
    """run_ablation saves ablation_results.json with correct structure.

    Training and validation are mocked so the test completes instantly.
    With epochs=2 and patience=1, each variant runs 2 epochs then early-stops.
    """
    train_loader, test_loader, class_weights = tiny_loaders

    # Minimal YAML config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 42\n"
        "model:\n"
        "  hidden_size: 16\n"
        "  lstm_layers: 1\n"
        "training:\n"
        "  lr: 0.001\n"
        "  batch_size: 16\n"
        "  window: 100\n"
        "  train_days: 7\n"
        "  epochs: 2\n"
        "  patience: 1\n"
        "data:\n"
        "  horizons: [10]\n"
    )

    output_dir = str(tmp_path / "ablation")

    with (
        patch(
            "deeplob.ablation.get_dataloaders",
            return_value=(train_loader, test_loader, class_weights),
        ),
        patch("deeplob.ablation.train_one_epoch", return_value=0.5),
        patch("deeplob.ablation.validate", return_value=(0.5, 0.70)),
    ):
        run_ablation(str(config_path), "data/raw/", k=10, output_dir=output_dir, pretrained_dir="")

    results_path = Path(output_dir) / "ablation_results.json"
    assert results_path.exists(), "ablation_results.json was not created"

    with results_path.open() as fh:
        data = json.load(fh)

    assert data["k"] == 10, f"Expected k=10, got {data['k']}"
    assert set(data["macro_f1"].keys()) == {
        "CNN only",
        "CNN + Inception",
        "Full DeepLOB",
    }, f"Unexpected model keys: {set(data['macro_f1'].keys())}"
    # Each variant ran for 2 epochs, first epoch improved → best_val_f1 = 0.70
    for name, f1 in data["macro_f1"].items():
        assert f1 == 0.70, f"{name}: expected best_val_f1=0.70, got {f1}"
