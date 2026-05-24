"""Tests for deeplob.ablation.

All tests use synthetic tensors — no FI-2010 data required.
"""

import torch

from deeplob.ablation import CNNInceptionModel, CNNOnlyModel
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
