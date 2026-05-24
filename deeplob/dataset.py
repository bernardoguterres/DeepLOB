"""LOB data loading, normalisation, windowing, and PyTorch Dataset/DataLoader construction.

Implements the full preprocessing pipeline for the FI-2010 benchmark dataset:

    load_fi2010_with_boundaries → time_split → normalise → make_windows
        → LOBDataset → DataLoader

The convenience wrapper ``get_dataloaders`` runs the entire chain in one call.
"""

from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maps prediction horizon k → column index of the corresponding label in the
#: FI-2010 .npy files (columns 0-39 are raw LOB features; 40-44 are labels).
_K_MAP: dict[int, int] = {1: 40, 2: 41, 3: 42, 5: 43, 10: 44}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "load_fi2010_with_boundaries",
    "time_split",
    "normalise",
    "make_windows",
    "LOBDataset",
    "get_dataloaders",
]


def load_fi2010_with_boundaries(data_dir: str, k: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load FI-2010 dataset from .npy files and return data with day boundaries.

    Each .npy file contains one trading day with shape ``(N_day, 144)``:

    * Columns 0–39: raw LOB features (10 bid + 10 ask price/volume pairs).
    * Columns 40–44: integer labels (1/2/3) for horizons k ∈ {1, 2, 3, 5, 10}.
    * Columns 45–143: hand-crafted features (ignored here).

    Files are loaded in alphabetical order to preserve chronological day order.

    Args:
        data_dir: Path to directory containing FI-2010 .npy files (one per day).
        k: Prediction horizon. One of [1, 2, 3, 5, 10].

    Returns:
        Tuple of:
            X: Raw LOB features, shape ``(N_total, 40)``. Columns 0–39 only.
            y: Integer class labels (0=down, 1=stationary, 2=up), shape ``(N_total,)``.
            boundaries: Cumulative row counts per day, e.g. ``[50231, 102847, ...]``.
                Used by :func:`time_split` to cut at exact day boundaries.

    Raises:
        FileNotFoundError: If *data_dir* does not exist or contains no .npy files.
        ValueError: If *k* is not in [1, 2, 3, 5, 10].
    """
    if k not in _K_MAP:
        raise ValueError(f"k must be one of {sorted(_K_MAP)}, got {k}")

    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    npy_files = sorted(path.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    label_col = _K_MAP[k]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    boundaries: list[int] = []
    cumulative = 0

    for fp in npy_files:
        arr = np.load(fp)  # (N_day, 144)
        x_parts.append(arr[:, :40].astype(np.float64))
        # Raw labels are 1/2/3 → subtract 1 → 0/1/2
        y_parts.append(arr[:, label_col].astype(np.int64) - 1)
        cumulative += arr.shape[0]
        boundaries.append(cumulative)

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y, boundaries


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    boundaries: list[int],
    train_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data at a day boundary into train and test sets.

    Never shuffles. Split point is ``boundaries[train_days - 1]``.

    Args:
        X: Feature array, shape ``(N, 40)``.
        y: Label array, shape ``(N,)``.
        boundaries: Cumulative day boundary indices from
            :func:`load_fi2010_with_boundaries`.
        train_days: Number of days to use for training (default 7).

    Returns:
        Tuple of ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If *train_days* >= ``len(boundaries)``.
        AssertionError: If train + test sizes do not sum to total N.
    """
    if train_days >= len(boundaries):
        raise ValueError(f"train_days ({train_days}) must be < len(boundaries) ({len(boundaries)})")

    split = boundaries[train_days - 1]
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    assert len(X_train) + len(X_test) == len(
        X
    ), f"Train ({len(X_train)}) + test ({len(X_test)}) != total ({len(X)})"

    return X_train, y_train, X_test, y_test


def normalise(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train set only and transform both splits.

    Fitting on the test set would constitute data leakage; the scaler is
    returned so it can be saved alongside model checkpoints for inference.

    Args:
        X_train: Training features, shape ``(N_train, 40)``.
        X_test: Test features, shape ``(N_test, 40)``.

    Returns:
        Tuple of ``(X_train_normalised, X_test_normalised, fitted_scaler)``.
        Scaler is returned for saving alongside checkpoints.
    """
    scaler = StandardScaler()
    X_train_norm: np.ndarray = scaler.fit_transform(X_train)
    X_test_norm: np.ndarray = scaler.transform(X_test)
    return X_train_norm, X_test_norm, scaler


def make_windows(X: np.ndarray, y: np.ndarray, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows over LOB event sequence.

    Each sample covers events ``[i : i+window]``. The label is ``y[i+window-1]``
    (the last event in the window). No label references beyond the window end.

    Uses :func:`numpy.lib.stride_tricks.sliding_window_view` to return a
    zero-copy view of *X*, keeping memory overhead low for large datasets.

    Args:
        X: Feature array, shape ``(N, 40)``.
        y: Label array, shape ``(N,)``.
        window: Number of events per sample (default 100).

    Returns:
        Tuple of:
            X_windowed: shape ``(N - window + 1, window, 40)``.
            y_windowed: shape ``(N - window + 1,)``.

    Raises:
        ValueError: If *window* > ``len(X)``.
    """
    n_total = len(X)
    if window > n_total:
        raise ValueError(f"window ({window}) > len(X) ({n_total})")

    # sliding_window_view on (N, F) with shape (window, F)
    # returns (N-window+1, 1, window, F); squeeze axis 1 → (N-window+1, window, F)
    X_windowed: np.ndarray = sliding_window_view(X, (window, X.shape[1])).squeeze(axis=1)
    y_windowed: np.ndarray = y[window - 1 :]

    return X_windowed, y_windowed


class LOBDataset(Dataset):
    """PyTorch Dataset wrapping windowed LOB arrays.

    Tensors are built lazily per sample to avoid materialising the full
    ``(N, window, 40)`` float32 array in memory when *X* is a stride view.

    Args:
        X: Windowed feature array, shape ``(N, window, 40)``.
        y: Label array, shape ``(N,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = X  # kept as numpy (may be a zero-copy stride view)
        self._y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self) -> int:
        """Return total number of windowed samples."""
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample.

        Args:
            idx: Sample index.

        Returns:
            x: Float tensor, shape ``(1, window, 40)``. Channel dim prepended for CNN.
            y: Long tensor scalar (class index).
        """
        # np.array always allocates a new writable C-contiguous buffer.
        # np.ascontiguousarray is not enough here: if the stride view already has
        # the right dtype it returns the read-only source unchanged, which makes
        # torch.from_numpy emit a non-writable-tensor warning.
        x = torch.from_numpy(np.array(self._X[idx], dtype=np.float32)).unsqueeze(0)
        return x, self._y[idx]


def get_dataloaders(
    data_dir: str,
    k: int,
    batch_size: int,
    window: int = 100,
    train_days: int = 7,
) -> tuple[DataLoader, DataLoader, Tensor]:
    """Full pipeline from raw files to DataLoaders.

    Runs: ``load → split → normalise → window → LOBDataset → DataLoader``.

    Class weights are computed from the training labels only, using the
    inverse-frequency formula ``weight[c] = N / (3 * count[c])`` where
    *N* = number of training windows. Suitable for use with
    :class:`torch.nn.CrossEntropyLoss(weight=...)`.

    Args:
        data_dir: Path to FI-2010 .npy files.
        k: Prediction horizon. One of [1, 2, 3, 5, 10].
        batch_size: Batch size for both loaders.
        window: Sliding window length (default 100).
        train_days: Days used for training (default 7).

    Returns:
        Tuple of ``(train_loader, test_loader, class_weights)``.
        class_weights: Float tensor of length 3, computed from *y_train* only.
            ``weight[c] = N / (3 * count[c])`` where N = ``len(y_train)``.
    """
    X, y, boundaries = load_fi2010_with_boundaries(data_dir, k)
    X_train, y_train, X_test, y_test = time_split(X, y, boundaries, train_days)
    X_train_n, X_test_n, _ = normalise(X_train, X_test)

    X_train_w, y_train_w = make_windows(X_train_n, y_train, window)
    X_test_w, y_test_w = make_windows(X_test_n, y_test, window)

    train_loader = DataLoader(
        LOBDataset(X_train_w, y_train_w),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        LOBDataset(X_test_w, y_test_w),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Inverse-frequency class weights from training windows only
    n_train = len(y_train_w)
    counts = np.bincount(y_train_w, minlength=3)
    class_weights = torch.tensor(n_train / (3.0 * counts), dtype=torch.float32)

    return train_loader, test_loader, class_weights
