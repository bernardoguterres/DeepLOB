"""Tests for deeplob.model.

All tests use random tensors — the FI-2010 dataset is not required.

Tests are flat functions (no classes) following the
``test_<what>_<condition>`` naming convention.
"""

import torch

from deeplob.model import CNNBlock, DeepLOB, InceptionModule, count_parameters

# ---------------------------------------------------------------------------
# 1. Full model — output shape
# ---------------------------------------------------------------------------


def test_output_shape(model, batch):
    """Full forward pass must produce (batch_size, 3) logits."""
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (8, 3), f"Expected (8, 3), got {output.shape}"


# ---------------------------------------------------------------------------
# 2. CNNBlock — spatial feature extraction shape
# ---------------------------------------------------------------------------


def test_cnn_block_output_shape():
    """CNNBlock must map (B, 1, 100, 40) → (B, 32, 94, 20)."""
    torch.manual_seed(0)
    x = torch.randn(8, 1, 100, 40)
    cnn = CNNBlock()
    cnn.eval()
    with torch.no_grad():
        out = cnn(x)
    print(f"CNNBlock actual output shape: {out.shape}")
    assert out.shape == (8, 32, 94, 20), f"Expected (8, 32, 94, 20), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. InceptionModule — channel concatenation shape
# ---------------------------------------------------------------------------


def test_inception_output_channels():
    """InceptionModule must map (B, 32, T, 20) → (B, 192, T, 20)."""
    torch.manual_seed(0)
    x = torch.randn(8, 32, 94, 20)
    inception = InceptionModule()
    inception.eval()
    with torch.no_grad():
        out = inception(x)
    assert out.shape == (8, 192, 94, 20), f"Expected (8, 192, 94, 20), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. Parameter count — capacity sanity check
# ---------------------------------------------------------------------------


def test_parameter_count_in_range():
    """Trainable parameter count must be between 200k and 600k."""
    n = count_parameters(DeepLOB())
    print(f"DeepLOB trainable parameters: {n:,}")  # always visible with pytest -s
    assert (
        200_000 <= n <= 600_000
    ), f"Parameter count {n:,} is outside the expected range [200k, 600k]"


# ---------------------------------------------------------------------------
# 5. Numerical stability — no NaN or Inf in output
# ---------------------------------------------------------------------------


def test_no_nan_no_inf_in_output(model, batch):
    """Forward pass must not produce NaN or Inf values."""
    with torch.no_grad():
        output = model(batch)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


# ---------------------------------------------------------------------------
# 6. Gradient flow — every parameter receives a gradient
# ---------------------------------------------------------------------------


def test_gradient_flow_all_params():
    """All trainable parameters must receive a non-zero gradient after backward."""
    torch.manual_seed(0)
    m = DeepLOB()  # train mode by default
    x = torch.randn(4, 1, 100, 40)
    target = torch.randint(0, 3, (4,))

    output = m(x)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()

    for name, param in m.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert not (param.grad == 0).all(), f"All-zero gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# 7. Single-sample batch — batch-size-1 edge case
# ---------------------------------------------------------------------------


def test_single_sample_batch(model):
    """Model must handle batch_size=1 without error."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 100, 40)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 3), f"Expected (1, 3), got {output.shape}"


# ---------------------------------------------------------------------------
# 8. Raw logits — no softmax applied inside the model
# ---------------------------------------------------------------------------


def test_output_is_raw_logits(model, batch):
    """Row sums of the output must NOT be close to 1.0 (no internal softmax)."""
    with torch.no_grad():
        output = model(batch)
    row_sums = output.sum(dim=1)
    # If softmax were applied, every row sum would be exactly 1.0.
    # For raw logits the sum is arbitrary and extremely unlikely to be ~1.0.
    assert not torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-3
    ), f"Row sums {row_sums.tolist()} are all ~1.0 — model may be applying softmax"


# ---------------------------------------------------------------------------
# 9. Device agnostic — explicit CPU placement
# ---------------------------------------------------------------------------


def test_model_is_device_agnostic():
    """Model and input explicitly placed on CPU must complete without error."""
    torch.manual_seed(0)
    m = DeepLOB().cpu()
    m.eval()
    x = torch.randn(2, 1, 100, 40).cpu()
    with torch.no_grad():
        output = m(x)
    assert output.device.type == "cpu", f"Expected cpu, got {output.device.type}"
    assert output.shape == (2, 3)
