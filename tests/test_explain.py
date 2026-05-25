"""Tests for deeplob.explain.

All tests use random tensors — the FI-2010 dataset is not required.
Model fixtures are defined in conftest.py (trained_model, single_sample).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from deeplob.explain import (
    FEATURE_NAMES,
    _use_agg_backend,
    batch_integrated_gradients,
    integrated_gradients,
    plot_class_attribution_heatmap,
    plot_feature_importance,
    run_explanation,
)
from deeplob.utils import save_checkpoint

# ---------------------------------------------------------------------------
# 1. IG output shape and dtype
# ---------------------------------------------------------------------------


def test_ig_output_shape(trained_model, single_sample):
    """integrated_gradients must return a (40,) float32 tensor."""
    device = torch.device("cpu")
    attrs = integrated_gradients(trained_model, single_sample, target_class=2, device=device)

    assert attrs.shape == (40,), f"Expected shape (40,), got {attrs.shape}"
    assert attrs.dtype == torch.float32, f"Expected float32, got {attrs.dtype}"


# ---------------------------------------------------------------------------
# 2. IG completeness axiom — the critical correctness test
# ---------------------------------------------------------------------------


def test_ig_completeness_axiom(trained_model, single_sample):
    """IG completeness: attrs.sum() * 100 ≈ F(x)[target] - F(baseline)[target].

    Uses n_steps=200 for high integration accuracy (error ≈ O(1/n_steps)).
    Tolerance of 0.05 is ~1% of a typical [-5, 5] output range.

    If this test fails, the IG implementation has a bug — do NOT relax
    the tolerance; fix the source instead.  The most common causes are:
    - Forgetting to multiply by (x − baseline) at the end.
    - Using sum() instead of mean() when averaging gradients across steps.
    - Shape mismatch causing incorrect window averaging.
    """
    device = torch.device("cpu")
    target_class = 2
    baseline = torch.zeros_like(single_sample)

    attrs = integrated_gradients(
        trained_model,
        single_sample,
        target_class=target_class,
        baseline=baseline,
        n_steps=200,
        device=device,
    )

    # Ground-truth difference F(x)[target] - F(baseline)[target]
    trained_model.eval()
    with torch.no_grad():
        f_x = trained_model(single_sample.to(device))[0, target_class].item()
        f_base = trained_model(baseline.to(device))[0, target_class].item()
    expected = f_x - f_base

    # attrs has shape (40,) = mean over 100 window timesteps
    # so attrs.sum() = (1/100) × Σ all raw IG values ≈ (1/100) × (F(x) - F(base))
    actual = attrs.sum().item() * 100

    assert abs(actual - expected) < 0.05, (
        f"IG completeness violated: actual={actual:.4f}, expected={expected:.4f}, "
        f"diff={abs(actual - expected):.4f} (tolerance=0.05). "
        "Check that avg_grads is multiplied by (x - baseline)."
    )


# ---------------------------------------------------------------------------
# 3. IG uses zeros baseline when none is supplied
# ---------------------------------------------------------------------------


def test_ig_baseline_zeros_by_default(trained_model, single_sample):
    """integrated_gradients must not raise when baseline is omitted."""
    device = torch.device("cpu")
    # Should complete without error; zeros baseline is applied internally
    attrs = integrated_gradients(trained_model, single_sample, target_class=0, device=device)
    assert attrs.shape == (40,)


# ---------------------------------------------------------------------------
# 4. IG infers device from the model when device=None
# ---------------------------------------------------------------------------


def test_ig_device_inferred_from_model(trained_model, single_sample):
    """integrated_gradients infers compute device from model.parameters() when device=None."""
    # No device= kwarg — line 101 infers it from next(model.parameters()).device
    attrs = integrated_gradients(trained_model, single_sample, target_class=1)
    assert attrs.shape == (40,)
    assert attrs.dtype == torch.float32


# ---------------------------------------------------------------------------
# 5. _use_agg_backend silently ignores locked-backend errors
# ---------------------------------------------------------------------------


def test_use_agg_backend_ignores_runtime_error():
    """_use_agg_backend must not propagate AttributeError/RuntimeError from matplotlib.use."""
    # Simulate matplotlib.use raising RuntimeError (backend already locked)
    with patch("matplotlib.use", side_effect=RuntimeError("backend locked")):
        _use_agg_backend()  # must not raise


def test_use_agg_backend_ignores_attribute_error():
    """_use_agg_backend must not propagate AttributeError from matplotlib.use."""
    with patch("matplotlib.use", side_effect=AttributeError("no use")):
        _use_agg_backend()  # must not raise


# ---------------------------------------------------------------------------
# 6. batch_integrated_gradients — output dict keys and shapes
# ---------------------------------------------------------------------------


def test_batch_ig_output_keys(trained_model, tiny_loaders):
    """batch_integrated_gradients must return a dict with correct keys and shapes."""
    _, test_loader, _ = tiny_loaders
    device = torch.device("cpu")

    result = batch_integrated_gradients(
        trained_model, test_loader, device, n_samples=10, n_steps=10
    )

    required_keys = {"attributions", "labels", "predictions", "feature_names"}
    missing = required_keys - result.keys()
    assert not missing, f"Result is missing keys: {missing}"

    assert result["attributions"].shape == (
        10,
        40,
    ), f"Expected attributions shape (10, 40), got {result['attributions'].shape}"
    assert (
        len(result["feature_names"]) == 40
    ), f"Expected 40 feature names, got {len(result['feature_names'])}"


# ---------------------------------------------------------------------------
# 7. FEATURE_NAMES — length and expected values
# ---------------------------------------------------------------------------


def test_feature_names_length_and_format():
    """FEATURE_NAMES must have exactly 40 entries in the correct LOB format."""
    assert len(FEATURE_NAMES) == 40, f"Expected 40 feature names, got {len(FEATURE_NAMES)}"
    assert (
        FEATURE_NAMES[0] == "ask_price_L1"
    ), f"Expected FEATURE_NAMES[0]='ask_price_L1', got '{FEATURE_NAMES[0]}'"
    assert (
        FEATURE_NAMES[1] == "ask_vol_L1"
    ), f"Expected FEATURE_NAMES[1]='ask_vol_L1', got '{FEATURE_NAMES[1]}'"
    assert "bid_price_L1" in FEATURE_NAMES, "'bid_price_L1' not found in FEATURE_NAMES"
    assert "bid_vol_L10" in FEATURE_NAMES, "'bid_vol_L10' not found in FEATURE_NAMES"


# ---------------------------------------------------------------------------
# 8. plot_feature_importance — saves a non-empty PNG file
# ---------------------------------------------------------------------------


def test_plot_feature_importance_saves_file(tmp_path):
    """plot_feature_importance must write a non-empty PNG to save_path."""
    rng = np.random.default_rng(42)
    attributions = rng.standard_normal((50, 40)).astype(np.float32)
    save_path = str(tmp_path / "test_importance.png")

    plot_feature_importance(attributions, FEATURE_NAMES, save_path=save_path)

    png = tmp_path / "test_importance.png"
    assert png.exists(), f"Expected PNG at {save_path} but file was not created"
    assert png.stat().st_size > 0, "Saved PNG file is empty"


def test_plot_feature_importance_default_save_path():
    """plot_feature_importance uses the default path 'outputs/plots/feature_importance.png'.

    Patches plt.savefig and Path.mkdir so no filesystem writes occur.
    Exercises the ``if save_path is None:`` branch (line 331).
    """
    rng = np.random.default_rng(42)
    attributions = rng.standard_normal((10, 40)).astype(np.float32)

    with patch("matplotlib.pyplot.savefig"), patch("pathlib.Path.mkdir"):
        plot_feature_importance(attributions, FEATURE_NAMES)
        # If save_path is None, line 331 assigns the default string — no raise expected


def test_feature_color_grey_fallback(tmp_path):
    """plot_feature_importance assigns 'grey' to features with unrecognised prefixes."""
    rng = np.random.default_rng(0)
    attributions = rng.standard_normal((10, 40)).astype(np.float32)
    # Inflate the first feature so it appears in the top-k selection
    attributions[:, 0] = 100.0
    # Replace the first feature name with one that matches no colour prefix
    custom_names = ["unknown_level_L1"] + list(FEATURE_NAMES[1:])
    save_path = str(tmp_path / "test_grey.png")

    # Must not raise; the 'grey' colour branch is exercised for 'unknown_level_L1'
    plot_feature_importance(attributions, custom_names, save_path=save_path)

    assert (tmp_path / "test_grey.png").exists()


# ---------------------------------------------------------------------------
# 9. plot_class_attribution_heatmap — saves a non-empty PNG file
# ---------------------------------------------------------------------------


def test_plot_class_attribution_heatmap_saves_file(tmp_path):
    """plot_class_attribution_heatmap must write a non-empty PNG to save_path."""
    rng = np.random.default_rng(42)
    attributions = rng.standard_normal((60, 40)).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)
    save_path = str(tmp_path / "test_heatmap.png")

    plot_class_attribution_heatmap(attributions, labels, FEATURE_NAMES, save_path=save_path)

    png = tmp_path / "test_heatmap.png"
    assert png.exists(), f"Expected PNG at {save_path} but file was not created"
    assert png.stat().st_size > 0, "Saved heatmap PNG file is empty"


def test_plot_class_attribution_heatmap_missing_class(tmp_path):
    """plot_class_attribution_heatmap handles a label set with only two classes present."""
    rng = np.random.default_rng(1)
    attributions = rng.standard_normal((40, 40)).astype(np.float32)
    # Only classes 0 and 2 — class 1 is absent
    labels = np.array([0] * 20 + [2] * 20, dtype=np.int64)
    save_path = str(tmp_path / "test_heatmap_2class.png")

    plot_class_attribution_heatmap(attributions, labels, FEATURE_NAMES, save_path=save_path)

    assert (tmp_path / "test_heatmap_2class.png").exists()


# ---------------------------------------------------------------------------
# 10. shap_summary — mocked shap import
# ---------------------------------------------------------------------------


def test_shap_summary_returns_attributions(trained_model, tiny_loaders):
    """shap_summary returns a dict with correct keys when shap is mocked.

    shap.GradientExplainer is replaced by a MagicMock so no real gradient
    computation occurs and the test completes instantly.
    """
    from deeplob.explain import shap_summary

    _, test_loader, _ = tiny_loaders
    device = torch.device("cpu")

    n_background = 5
    n_explain = 10
    rng = np.random.default_rng(42)

    # Fake shap_values: list[3 arrays], each (n_explain, 1, 100, 40)
    fake_shap_vals = [
        rng.standard_normal((n_explain, 1, 100, 40)).astype(np.float32) for _ in range(3)
    ]

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = fake_shap_vals

    mock_shap = MagicMock()
    mock_shap.GradientExplainer.return_value = mock_explainer

    # Temporarily replace the 'shap' entry in sys.modules so that
    # `import shap` inside shap_summary picks up the mock
    with patch.dict(sys.modules, {"shap": mock_shap}):
        result = shap_summary(
            trained_model,
            test_loader,
            device,
            n_background=n_background,
            n_explain=n_explain,
        )

    required_keys = {"attributions", "labels", "predictions", "feature_names"}
    assert required_keys <= result.keys(), f"Missing keys: {required_keys - result.keys()}"
    assert result["attributions"].ndim == 2
    assert result["attributions"].shape[1] == 40
    assert len(result["feature_names"]) == 40


# ---------------------------------------------------------------------------
# 11. run_explanation — ValueError for unsupported method
# ---------------------------------------------------------------------------


def test_run_explanation_invalid_method():
    """run_explanation raises ValueError before touching the config for invalid method."""
    with pytest.raises(ValueError, match="method must be"):
        run_explanation("nonexistent.yaml", "data/raw/", method="gradient_tape")


# ---------------------------------------------------------------------------
# 12. run_explanation — full pipeline with mocked I/O
# ---------------------------------------------------------------------------


def test_run_explanation_saves_outputs(tmp_path, trained_model, tiny_loaders):
    """run_explanation saves PNGs and NPZ for method='ig'.

    get_dataloaders and batch_integrated_gradients are mocked to skip real
    data loading and IG computation.  A real checkpoint is written so that
    load_checkpoint exercises the actual torch.load path.
    """
    train_loader, test_loader, class_weights = tiny_loaders

    # --- config ----------------------------------------------------------
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

    # --- checkpoint ------------------------------------------------------
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = str(ckpt_dir / "best_model_k10.pt")
    optimizer = torch.optim.Adam(trained_model.parameters())
    save_checkpoint(trained_model, optimizer, epoch=1, val_f1=0.70, path=ckpt_path)

    # --- mock IG results -------------------------------------------------
    rng = np.random.default_rng(0)
    mock_result = {
        "attributions": rng.standard_normal((20, 40)).astype(np.float32),
        "labels": np.array([0] * 7 + [1] * 6 + [2] * 7, dtype=np.int64),
        "predictions": rng.integers(0, 3, size=20).astype(np.int64),
        "feature_names": FEATURE_NAMES,
    }

    output_dir = str(tmp_path / "plots")

    with (
        patch(
            "deeplob.explain.get_dataloaders",
            return_value=(train_loader, test_loader, class_weights),
        ),
        patch("deeplob.explain.batch_integrated_gradients", return_value=mock_result),
    ):
        run_explanation(
            str(config_path),
            "data/raw/",
            checkpoint_dir=str(ckpt_dir),
            output_dir=output_dir,
            k=10,
            method="ig",
        )

    out = Path(output_dir)
    assert (out / "feature_importance_k10.png").exists(), "feature_importance PNG not created"
    assert (out / "heatmap_k10.png").exists(), "heatmap PNG not created"
    assert (out / "attributions_k10.npz").exists(), "attributions NPZ not created"

    # Verify NPZ content
    npz = np.load(out / "attributions_k10.npz", allow_pickle=True)
    assert "attributions" in npz
    assert "labels" in npz
    assert "predictions" in npz
