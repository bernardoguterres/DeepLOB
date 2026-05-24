"""Shared pytest fixtures for the DeepLOB test suite."""

import numpy as np
import pytest
import torch

from deeplob.model import DeepLOB


@pytest.fixture
def model():
    """Instantiate a DeepLOB model in eval mode."""
    m = DeepLOB()
    m.eval()
    return m


@pytest.fixture
def batch():
    """Standard batch: (8, 1, 100, 40) random tensor."""
    torch.manual_seed(42)
    return torch.randn(8, 1, 100, 40)


@pytest.fixture
def synthetic_day_files(tmp_path):
    """Create 10 synthetic FI-2010-format .npy files in a temp directory.

    Each file contains 5000 events × 144 columns:
    * Columns 0–39:  random floats (LOB features).
    * Columns 40–44: random ints 1–3 (labels for k=1,2,3,5,10).
    * Columns 45–143: random floats (hand-crafted features, should be ignored).
    """
    rng = np.random.default_rng(42)
    for i in range(10):
        data = rng.random((5000, 144))
        data[:, 40:45] = rng.integers(1, 4, size=(5000, 5))
        np.save(tmp_path / f"day_{i + 1:02d}.npy", data)
    return str(tmp_path)


@pytest.fixture
def small_X_y():
    """Small X, y arrays for windowing and Dataset tests.

    Returns:
        Tuple of (X, y) where X has shape (1000, 40) and y has shape (1000,).
    """
    rng = np.random.default_rng(42)
    X = rng.random((1000, 40)).astype(np.float32)
    y = rng.integers(0, 3, size=1000).astype(np.int64)
    return X, y
