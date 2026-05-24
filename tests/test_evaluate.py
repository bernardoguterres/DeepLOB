"""Tests for deeplob.evaluate.

All tests use synthetic data — the FI-2010 dataset is not required.
A ``StaticPredictor`` helper returns fixed logits so that predictions are
fully deterministic without needing a trained model.
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from deeplob.evaluate import PAPER_BENCHMARKS, benchmark_table, evaluate

# ---------------------------------------------------------------------------
# Helper: deterministic model that always returns pre-set logits
# ---------------------------------------------------------------------------


class StaticPredictor(nn.Module):
    """Return a fixed logit tensor regardless of input.

    Args:
        logits: Tensor of shape ``(N, num_classes)`` to return in one shot.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits
        self._call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        batch_size = x.shape[0]
        start = self._call_count
        end = start + batch_size
        self._call_count = end
        return self._logits[start:end]


def _make_loader(y_true: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Wrap labels as a single-batch loader: [(x_dummy, y_tensor)]."""
    n = len(y_true)
    x = torch.zeros(n, 1, 100, 40)  # shape expected by DeepLOB — dummy values
    y = torch.tensor(y_true, dtype=torch.long)
    return [(x, y)]


def _make_logits(y_pred: list[int], num_classes: int = 3) -> torch.Tensor:
    """One-hot logits: class *c* gets score 10.0, others get 0.0."""
    n = len(y_pred)
    logits = torch.zeros(n, num_classes)
    for i, c in enumerate(y_pred):
        logits[i, c] = 10.0
    return logits


# ---------------------------------------------------------------------------
# 1. Perfect predictions → accuracy 1.0 and macro F1 1.0
# ---------------------------------------------------------------------------


def test_evaluate_perfect_predictions():
    """With all-correct predictions, accuracy and macro F1 must both be 1.0."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = y_true  # perfect

    logits = _make_logits(y_pred)
    model = StaticPredictor(logits)
    loader = _make_loader(y_true)
    device = torch.device("cpu")

    result = evaluate(model, loader, device)

    assert result["accuracy"] == 1.0, f"Expected accuracy 1.0, got {result['accuracy']}"
    assert result["macro_f1"] == 1.0, f"Expected macro_f1 1.0, got {result['macro_f1']}"


# ---------------------------------------------------------------------------
# 2. Known case — verify all metrics against sklearn reference values
# ---------------------------------------------------------------------------


def test_evaluate_known_case():
    """Metrics must match sklearn reference within tolerance 0.001.

    Ground truth: [0, 0, 1, 1, 2, 2]
    Predictions:  [0, 1, 1, 1, 2, 0]
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    logits = _make_logits(y_pred)
    model = StaticPredictor(logits)
    loader = _make_loader(y_true)
    device = torch.device("cpu")

    result = evaluate(model, loader, device)

    labels = [0, 1, 2]
    expected_accuracy = accuracy_score(y_true, y_pred)
    expected_macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    expected_weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )

    tol = 0.001
    assert (
        abs(result["accuracy"] - expected_accuracy) < tol
    ), f"accuracy mismatch: got {result['accuracy']:.4f}, expected {expected_accuracy:.4f}"
    assert (
        abs(result["macro_f1"] - expected_macro_f1) < tol
    ), f"macro_f1 mismatch: got {result['macro_f1']:.4f}, expected {expected_macro_f1:.4f}"
    assert (
        abs(result["weighted_f1"] - expected_weighted_f1) < tol
    ), f"weighted_f1 mismatch: got {result['weighted_f1']:.4f}, expected {expected_weighted_f1:.4f}"


# ---------------------------------------------------------------------------
# 3. benchmark_table produces valid Markdown with required content
# ---------------------------------------------------------------------------


def test_benchmark_table_is_valid_markdown():
    """benchmark_table must contain pipe characters, all horizon keys, and 'Paper F1'."""
    results = {k: 0.70 for k in PAPER_BENCHMARKS}
    table = benchmark_table(results)

    assert "|" in table, "Table output contains no pipe characters"
    assert "Paper F1" in table, "'Paper F1' header not found in table"

    for k in PAPER_BENCHMARKS:
        assert str(k) in table, f"Horizon k={k} not found in table"


# ---------------------------------------------------------------------------
# 4. benchmark_table Δ sign is correct
# ---------------------------------------------------------------------------


def test_benchmark_table_delta_sign():
    """Δ column must be '+...' when ours > paper and '-...' when ours < paper."""
    # k=10: paper=0.83; 0.85 > 0.83 → positive delta
    above = benchmark_table({10: 0.85})
    assert "+" in above, f"Expected '+' in delta for result above paper benchmark; got:\n{above}"

    # k=10: paper=0.83; 0.80 < 0.83 → negative delta
    below = benchmark_table({10: 0.80})
    assert "-" in below, f"Expected '-' in delta for result below paper benchmark; got:\n{below}"


# ---------------------------------------------------------------------------
# 5. per_class_f1 has exactly 3 elements
# ---------------------------------------------------------------------------


def test_per_class_f1_length():
    """per_class_f1 must always contain exactly 3 elements (one per class)."""
    # Use predictions that only cover two classes to test the labels=[0,1,2] fix
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    logits = _make_logits(y_pred)
    model = StaticPredictor(logits)
    loader = _make_loader(y_true)
    device = torch.device("cpu")

    result = evaluate(model, loader, device)

    assert len(result["per_class_f1"]) == 3, (
        f"Expected per_class_f1 to have length 3, got {len(result['per_class_f1'])}: "
        f"{result['per_class_f1']}"
    )
