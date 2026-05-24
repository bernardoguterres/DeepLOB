"""Tests for deeplob.explain.

All tests use random tensors — the FI-2010 dataset is not required.
Model fixtures are defined in conftest.py (trained_model, single_sample).
"""

import numpy as np
import torch

from deeplob.explain import (
    FEATURE_NAMES,
    batch_integrated_gradients,
    integrated_gradients,
    plot_feature_importance,
)

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
# 4. batch_integrated_gradients — output dict keys and shapes
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
# 5. FEATURE_NAMES — length and expected values
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
# 6. plot_feature_importance — saves a non-empty PNG file
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
