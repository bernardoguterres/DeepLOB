"""Tests for deeplob.dataset.

All tests use synthetic data generated with np.random — the FI-2010 dataset
is not required to run this suite.

Tests are flat functions (no classes) following the
``test_<what>_<condition>`` naming convention.
"""

import numpy as np
import pytest
import torch

from deeplob.dataset import (
    LOBDataset,
    load_fi2010_with_boundaries,
    make_windows,
    normalise,
    time_split,
)

# ---------------------------------------------------------------------------
# 1. load_fi2010_with_boundaries — feature column selection
# ---------------------------------------------------------------------------


def test_load_returns_correct_feature_columns(synthetic_day_files):
    """X must have exactly 40 columns and y must contain only classes 0, 1, 2."""
    X, y, _ = load_fi2010_with_boundaries(synthetic_day_files, k=1)

    assert X.shape[1] == 40, f"Expected 40 feature columns, got {X.shape[1]}"
    assert set(np.unique(y)) == {
        0,
        1,
        2,
    }, f"Expected labels {{0, 1, 2}} after remapping, got {set(np.unique(y))}"


# ---------------------------------------------------------------------------
# 2. load_fi2010_with_boundaries — day boundaries
# ---------------------------------------------------------------------------


def test_load_boundaries_sum_to_total(synthetic_day_files):
    """boundaries[-1] must equal len(X), which is 10 days × 5000 events = 50000."""
    X, y, boundaries = load_fi2010_with_boundaries(synthetic_day_files, k=1)

    assert boundaries[-1] == 50_000, f"Expected boundaries[-1] == 50000, got {boundaries[-1]}"
    assert len(X) == 50_000, f"Expected len(X) == 50000, got {len(X)}"
    assert len(y) == len(X), "len(y) must equal len(X)"


# ---------------------------------------------------------------------------
# 3. time_split — correct sizes
# ---------------------------------------------------------------------------


def test_time_split_no_overlap(synthetic_day_files):
    """Train and test sets must have the right sizes and partition all rows."""
    X, y, boundaries = load_fi2010_with_boundaries(synthetic_day_files, k=1)
    X_train, y_train, X_test, y_test = time_split(X, y, boundaries, train_days=7)

    assert len(X_train) == 35_000, f"Expected 35000 train rows, got {len(X_train)}"
    assert len(X_test) == 15_000, f"Expected 15000 test rows, got {len(X_test)}"
    assert len(X_train) + len(X_test) == 50_000


# ---------------------------------------------------------------------------
# 4. time_split — ordering preserved (no shuffle)
# ---------------------------------------------------------------------------


def test_time_split_never_shuffles(synthetic_day_files):
    """Last train row and first test row must be adjacent in the original array."""
    X, y, boundaries = load_fi2010_with_boundaries(synthetic_day_files, k=1)
    X_train, _, X_test, _ = time_split(X, y, boundaries, train_days=7)

    split = boundaries[6]  # = 35000

    # No gap: X_train[-1] is index split-1; X_test[0] is index split
    np.testing.assert_array_equal(
        X_train[-1],
        X[split - 1],
        err_msg="Last train row does not match X[split-1] — possible shuffle",
    )
    np.testing.assert_array_equal(
        X_test[0],
        X[split],
        err_msg="First test row does not match X[split] — possible gap or overlap",
    )


# ---------------------------------------------------------------------------
# 5. normalise — scaler fit on train only (no leakage)
# ---------------------------------------------------------------------------


def test_normalise_scaler_fit_on_train_only():
    """Train must be centred; test must NOT be centred (different distribution)."""
    rng = np.random.default_rng(0)
    # Train: values in [10, 11], mean ≈ 10.5
    X_train_raw = rng.random((1000, 40)) + 10.0
    # Test: values in [100, 101], mean ≈ 100.5 — very different scale
    X_test_raw = rng.random((1000, 40)) + 100.0

    X_train_norm, X_test_norm, _ = normalise(X_train_raw, X_test_raw)

    # Scaler was fit on train → train is centred
    assert (
        abs(X_train_norm.mean()) < 0.1
    ), f"Train mean after normalisation should be ≈0, got {X_train_norm.mean():.4f}"
    # Scaler was NOT refit on test → test is far from centred
    assert (
        abs(X_test_norm.mean()) > 1.0
    ), f"Test mean should be >> 0 (scaler not refit on test), got {X_test_norm.mean():.4f}"


# ---------------------------------------------------------------------------
# 6. make_windows — output shapes
# ---------------------------------------------------------------------------


def test_make_windows_output_shape(small_X_y):
    """Windowed arrays must have the correct shapes for N=1000, window=100."""
    X, y = small_X_y
    window = 100

    X_w, y_w = make_windows(X, y, window=window)

    assert X_w.shape == (901, 100, 40), f"Expected (901, 100, 40), got {X_w.shape}"
    assert y_w.shape == (901,), f"Expected (901,), got {y_w.shape}"


# ---------------------------------------------------------------------------
# 7. make_windows — no label look-ahead
# ---------------------------------------------------------------------------


def test_make_windows_no_lookahead(small_X_y):
    """y_windowed[i] must equal y_original[i + window - 1] for every window."""
    X, y = small_X_y
    window = 100

    X_w, y_w = make_windows(X, y, window=window)

    # Vectorised check: y_w should be exactly y shifted by window-1
    np.testing.assert_array_equal(
        y_w,
        y[window - 1 :],
        err_msg="y_windowed does not match y[window-1:] — possible label look-ahead",
    )

    # No label index may fall outside the original array
    max_label_idx = (len(y_w) - 1) + (window - 1)
    assert max_label_idx < len(
        y
    ), f"Label index {max_label_idx} is out of bounds for y of length {len(y)}"


# ---------------------------------------------------------------------------
# 8. LOBDataset — item shapes and dtypes
# ---------------------------------------------------------------------------


def test_lob_dataset_item_shapes(small_X_y):
    """Dataset[0] must return (1, window, 40) float32 tensor and int64 scalar."""
    X, y = small_X_y
    window = 100
    X_w, y_w = make_windows(X, y, window=window)
    dataset = LOBDataset(X_w, y_w)

    x, label = dataset[0]

    assert x.shape == (1, window, 40), f"Expected shape (1, 100, 40), got {x.shape}"
    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
    assert label.dtype == torch.int64, f"Expected int64, got {label.dtype}"
    assert label.ndim == 0, f"Label should be a 0-d tensor (scalar), got ndim={label.ndim}"


# ---------------------------------------------------------------------------
# 9. class weights — correct inverse-frequency formula
# ---------------------------------------------------------------------------


def test_class_weights_sum_correctly():
    """Class with fewer samples must receive a higher weight."""
    # Construct y_train with known class counts: 300 × 0, 500 × 1, 200 × 2
    y_train = np.array([0] * 300 + [1] * 500 + [2] * 200, dtype=np.int64)

    n = len(y_train)
    counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor(n / (3.0 * counts), dtype=torch.float32)

    assert class_weights.shape == (3,), f"Expected shape (3,), got {class_weights.shape}"
    assert class_weights.dtype == torch.float32

    # Class 2 (200 samples) must outweigh class 1 (500 samples)
    assert class_weights[2] > class_weights[1], (
        f"Rarer class should have higher weight: w[2]={class_weights[2]:.4f}, "
        f"w[1]={class_weights[1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 10. load_fi2010_with_boundaries — invalid k
# ---------------------------------------------------------------------------


def test_invalid_k_raises_valueerror(synthetic_day_files):
    """k=7 is not a valid horizon and must raise ValueError immediately."""
    with pytest.raises(ValueError, match="k must be one of"):
        load_fi2010_with_boundaries(synthetic_day_files, k=7)


# ---------------------------------------------------------------------------
# 11. load_fi2010_with_boundaries — missing data directory
# ---------------------------------------------------------------------------


def test_load_fi2010_data_dir_not_found():
    """load_fi2010_with_boundaries raises FileNotFoundError for a non-existent directory."""
    with pytest.raises(FileNotFoundError, match="data_dir does not exist"):
        load_fi2010_with_boundaries("/nonexistent/path/to/data", k=1)


# ---------------------------------------------------------------------------
# 12. load_fi2010_with_boundaries — empty directory (no .npy files)
# ---------------------------------------------------------------------------


def test_load_fi2010_no_npy_files(tmp_path):
    """load_fi2010_with_boundaries raises FileNotFoundError when directory has no .npy files."""
    with pytest.raises(FileNotFoundError, match="No .npy files found"):
        load_fi2010_with_boundaries(str(tmp_path), k=1)


# ---------------------------------------------------------------------------
# 13. time_split — train_days too large
# ---------------------------------------------------------------------------


def test_time_split_too_many_train_days(synthetic_day_files):
    """time_split raises ValueError when train_days >= len(boundaries)."""
    X, y, boundaries = load_fi2010_with_boundaries(synthetic_day_files, k=1)
    with pytest.raises(ValueError, match="train_days"):
        time_split(X, y, boundaries, train_days=len(boundaries))


# ---------------------------------------------------------------------------
# 14. make_windows — window larger than data
# ---------------------------------------------------------------------------


def test_make_windows_window_too_large(small_X_y):
    """make_windows raises ValueError when window > len(X)."""
    X, y = small_X_y
    with pytest.raises(ValueError, match="window"):
        make_windows(X, y, window=len(X) + 1)


# ---------------------------------------------------------------------------
# 15. LOBDataset — __len__ returns correct count
# ---------------------------------------------------------------------------


def test_lob_dataset_len(small_X_y):
    """LOBDataset.__len__ must equal the number of windowed samples."""
    X, y = small_X_y
    X_w, y_w = make_windows(X, y, window=100)
    dataset = LOBDataset(X_w, y_w)
    assert len(dataset) == len(y_w), f"Expected {len(y_w)}, got {len(dataset)}"


# ---------------------------------------------------------------------------
# 16. get_dataloaders — integration (synthetic data, real pipeline)
# ---------------------------------------------------------------------------


def test_get_dataloaders_integration(synthetic_day_files):
    """get_dataloaders returns two DataLoaders and class_weights tensor of shape (3,)."""
    from deeplob.dataset import get_dataloaders

    train_loader, test_loader, class_weights = get_dataloaders(
        data_dir=synthetic_day_files,
        k=1,
        batch_size=32,
        window=100,
        train_days=7,
    )

    assert class_weights.shape == (
        3,
    ), f"Expected class_weights shape (3,), got {class_weights.shape}"
    assert class_weights.dtype == torch.float32

    # Verify train batch shapes
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape[1:] == (
        1,
        100,
        40,
    ), f"Expected batch shape (*, 1, 100, 40), got {x_batch.shape}"
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype == torch.int64
