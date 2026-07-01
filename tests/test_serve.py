"""Tests for deeplob.serve (FastAPI inference server).

All tests use mocks — no checkpoint files or GPU required.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

import deeplob.serve as serve_module
from deeplob.serve import PredictRequest, _load_model_from_dir, app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with lifespan mocked out (no real checkpoint loading)."""
    with patch("deeplob.serve._load_model_from_dir"):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def restore_state():
    """Snapshot serve_module.state before a test and restore it afterward."""
    original = copy.copy(serve_module.state)
    yield serve_module.state
    serve_module.state.__dict__.update(original.__dict__)


@pytest.fixture
def mock_model():
    """Mock DeepLOB that returns a fixed (1, 3) logit tensor."""
    m = MagicMock()
    m.return_value = torch.tensor([[0.1, 0.7, 0.2]])
    return m


# ---------------------------------------------------------------------------
# 1. PredictRequest validation
# ---------------------------------------------------------------------------


def test_predict_request_accepts_40_features():
    req = PredictRequest(lob_snapshot=[0.5] * 40)
    assert len(req.lob_snapshot) == 40


def test_predict_request_rejects_wrong_length():
    with pytest.raises(Exception):
        PredictRequest(lob_snapshot=[0.5] * 10)


def test_predict_request_rejects_empty():
    with pytest.raises(Exception):
        PredictRequest(lob_snapshot=[])


# ---------------------------------------------------------------------------
# 2. GET /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data


def test_health_device_unknown_when_not_loaded(client, restore_state):
    restore_state.device = None
    resp = client.get("/health")
    assert resp.json()["device"] == "unknown"


# ---------------------------------------------------------------------------
# 3. POST /predict — model not loaded
# ---------------------------------------------------------------------------


def test_predict_503_when_model_not_loaded(client, restore_state):
    restore_state.model = None
    resp = client.post("/predict", json={"lob_snapshot": [0.1] * 40})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 4. POST /predict — happy path (no scaler)
# ---------------------------------------------------------------------------


def test_predict_returns_correct_structure(client, mock_model, restore_state):
    restore_state.model = mock_model
    restore_state.device = torch.device("cpu")
    restore_state.scaler = None
    resp = client.post("/predict", json={"lob_snapshot": [0.1] * 40})
    assert resp.status_code == 200
    data = resp.json()
    assert "direction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3
    assert 0 <= data["direction"] <= 2
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_direction_matches_argmax(client, mock_model, restore_state):
    # logits [0.1, 0.7, 0.2] → argmax index 1
    restore_state.model = mock_model
    restore_state.device = torch.device("cpu")
    restore_state.scaler = None
    resp = client.post("/predict", json={"lob_snapshot": [0.0] * 40})
    assert resp.json()["direction"] == 1


# ---------------------------------------------------------------------------
# 5. POST /predict — with scaler
# ---------------------------------------------------------------------------


def test_predict_applies_scaler_when_present(client, mock_model, restore_state):
    import numpy as np

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((1, 40), dtype=np.float32)

    restore_state.model = mock_model
    restore_state.device = torch.device("cpu")
    restore_state.scaler = mock_scaler
    resp = client.post("/predict", json={"lob_snapshot": [0.1] * 40})
    assert resp.status_code == 200
    mock_scaler.transform.assert_called_once()


# ---------------------------------------------------------------------------
# 6. POST /predict — input validation via HTTP
# ---------------------------------------------------------------------------


def test_predict_422_on_wrong_feature_count(client):
    resp = client.post("/predict", json={"lob_snapshot": [0.1] * 5})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 7. _load_model_from_dir — error paths
# ---------------------------------------------------------------------------


def test_load_model_raises_when_checkpoint_missing(tmp_path):
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        _load_model_from_dir(10, str(tmp_path))


def test_load_model_infers_hidden_size(tmp_path):
    ckpt_path = tmp_path / "best_model_k10.pt"
    ckpt_path.touch()

    mock_state = {"fc.weight": torch.zeros(3, 64)}
    mock_ckpt = {"model_state": mock_state, "epoch": 5, "val_f1": 0.75}

    with (
        patch("deeplob.serve.torch.load", return_value=mock_ckpt),
        patch("deeplob.serve.get_device", return_value=torch.device("cpu")),
        patch("deeplob.serve.DeepLOB") as MockDeepLOB,
    ):
        MockDeepLOB.return_value = MagicMock()
        _load_model_from_dir(10, str(tmp_path))
        MockDeepLOB.assert_called_once_with(hidden_size=64)


def test_load_model_loads_scaler_when_present(tmp_path):
    ckpt_path = tmp_path / "best_model_k10.pt"
    ckpt_path.touch()
    scaler_path = tmp_path / "scaler_k10.pkl"
    scaler_path.touch()

    mock_state = {"fc.weight": torch.zeros(3, 64)}
    mock_ckpt = {"model_state": mock_state, "epoch": 1, "val_f1": 0.5}
    mock_scaler = MagicMock()

    with (
        patch("deeplob.serve.torch.load", return_value=mock_ckpt),
        patch("deeplob.serve.get_device", return_value=torch.device("cpu")),
        patch("deeplob.serve.DeepLOB", return_value=MagicMock()),
        patch("deeplob.serve.pickle.load", return_value=mock_scaler),
        patch("builtins.open", MagicMock()),
    ):
        _load_model_from_dir(10, str(tmp_path))
        assert serve_module.state.scaler is mock_scaler


def test_load_model_warns_when_scaler_missing(tmp_path):
    ckpt_path = tmp_path / "best_model_k10.pt"
    ckpt_path.touch()
    # No scaler file — should log a warning but not raise.

    mock_state = {"fc.weight": torch.zeros(3, 64)}
    mock_ckpt = {"model_state": mock_state}

    with (
        patch("deeplob.serve.torch.load", return_value=mock_ckpt),
        patch("deeplob.serve.get_device", return_value=torch.device("cpu")),
        patch("deeplob.serve.DeepLOB", return_value=MagicMock()),
    ):
        _load_model_from_dir(10, str(tmp_path))  # must not raise
