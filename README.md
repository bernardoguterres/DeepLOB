# DeepLOB

PyTorch reimplementation of [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1902.09450) (Zhang, Zohren & Roberts, 2019) — mid-price movement classification from limit order book snapshots using a CNN + Inception + LSTM architecture on the FI-2010 benchmark.

![CI](https://github.com/bernardoguterres/DeepLOB/actions/workflows/ci.yml/badge.svg)

---

## Results

| k  | Paper (Zhang et al. 2019) | Ours   | Δ       |
|----|--------------------------|--------|---------|
| 1  | 0.67                     | 0.7401 | +10.5%  |
| 2  | 0.71                     | 0.6495 | −8.5%   |
| 3  | 0.75                     | 0.7298 | −2.7%   |
| 5  | 0.78                     | 0.7811 | +0.1%   |
| 10 | 0.83                     | 0.7565 | −8.9%   |

Metric: macro F1. Dataset: FI-2010, day-based 7/3 split.

k=2 and k=10 fall ~8–9% short of the paper. Both horizons show strong stationary-class F1 (≥0.83) but weaker minority-class F1 — weighted F1 for k=2 is 0.73 and for k=10 is 0.76, indicating the gap is driven by class imbalance at mid-range horizons rather than overall model failure.

---

## Architecture

```
Input (B, 1, 100, 40)
        │
        ▼
┌────────────────────────────────────┐
│            CNN Block               │
│  Conv2d(1→32, 1×2, stride 1×2)    │  pairs each price with its volume
│  Conv2d(32→32, 4×1) × 2           │  short-range temporal patterns
│  LeakyReLU(0.01) + BatchNorm2d    │
└────────────────────────────────────┘
        │ (B, 32, 94, 20)
        ▼
┌────────────────────────────────────┐
│         Inception Module           │
│  Branch A: 1×1 → 3×1 (pad 1)      │
│  Branch B: 1×1 → 5×1 (pad 2)      │  multi-scale temporal capture
│  Branch C: MaxPool(3×1) → 1×1     │
│  Concat → 192 channels            │
└────────────────────────────────────┘
        │ (B, 192, 94, 20)
        ▼
   permute + reshape
        │ (B, 1880, 192)
        ▼
┌────────────────────────────────────┐
│              LSTM                  │
│  hidden_size=64, layers=1          │  global sequence memory
│  take final timestep [:, -1, :]   │
└────────────────────────────────────┘
        │ (B, 64)
        ▼
   Linear(64 → 3)
        │
        ▼
Output (B, 3) — raw logits
```

---

## Quickstart

### 1. Get the data

```bash
python data/download.py
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Train (single horizon)

```bash
python -m deeplob.train --config configs/default.yaml --k 10
```

### 4. Train all horizons

```bash
for k in 1 2 3 5 10; do
    python -m deeplob.train --config configs/default.yaml --k $k
done
```

### 5. Evaluate

```bash
python -m deeplob.evaluate --config configs/default.yaml
```

### 6. Ablation study

```bash
python -m deeplob.ablation --config configs/default.yaml --k 10
```

### 7. Explainability

```bash
python -m deeplob.explain --config configs/default.yaml --k 10 --method ig
```

### 8. Serve predictions (AlphaLive integration)

```bash
python deeplob/serve.py --k 10 --checkpoint_dir outputs/ --port 8001
```

Check it's alive:

```bash
curl http://localhost:8001/health
# {"status":"ok","model":"deeplob_k10","device":"mps"}

curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"lob_snapshot": [0.1, 0.2, ..., 0.4]}'  # 40 floats
# {"direction":2,"confidence":0.81,"probabilities":[0.07,0.12,0.81]}
```

### 9. Tests

100 tests, 97% coverage.

```bash
pytest tests/ -v --cov=deeplob
```

---

## Inference Server

`deeplob/serve.py` is a single-file [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org) server designed for integration with [AlphaLive](https://github.com/bernardoguterres/AlphaLive).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Predict mid-price direction from a 40-feature LOB snapshot |
| `GET` | `/health` | Health check — model name and device |

**Request** (`POST /predict`):
```json
{"lob_snapshot": [<40 floats in FI-2010 order>]}
```
Features must be in FI-2010 column order: `ask_price_L1, ask_vol_L1, ask_price_L2, ask_vol_L2, … bid_vol_L10`. If your data is already z-score normalised (standard FI-2010 format), pass it directly. If a `scaler_k{k}.pkl` exists in the checkpoint directory, it will be applied automatically.

**Response**:
```json
{
  "direction": 2,
  "confidence": 0.81,
  "probabilities": [0.07, 0.12, 0.81]
}
```
`direction`: 0 = down, 1 = stationary, 2 = up.

**Startup flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--k` | `10` | Prediction horizon — loads `best_model_k{k}.pt` |
| `--checkpoint_dir` | `outputs/` | Directory containing the `.pt` checkpoint |
| `--port` | `8001` | HTTP port to listen on |

**Design notes:**
- Hidden size is inferred from the checkpoint weight shapes — no flag required and works for any `k`.
- A single 40-feature snapshot is tiled 100× to fill the temporal window the model expects. This is appropriate for AlphaLive's use case (no rolling LOB buffer available on Alpaca free tier; the filter fails open when no L2 data is present).
- Latency logged on every request. Typical: < 5 ms on CPU, < 2 ms on MPS/CUDA.

### AlphaLive integration

AlphaLive queries this server before placing any order. The `DeepLOBClient` (in `alphalive/services/deeplob_client.py`) runs concurrently with the AlphaSignal sentiment filter via `asyncio.gather`. Execution is allowed only if the predicted direction matches the strategy's intended direction **and** confidence ≥ threshold (default 0.6). Either filter fails open on timeout or connection error — the server being down never blocks a trade.

Configure in AlphaLive via env vars: `DEEPLOB_URL`, `DEEPLOB_CONFIDENCE_THRESHOLD`, `DEEPLOB_TIMEOUT_SECONDS`, `DEEPLOB_ENABLED`.

---

## Data Pipeline

**FI-2010** is a public limit order book dataset covering 10 trading days across 5 Finnish stocks (Nokia, WRT1V, KESBV, OUT1V, WRTBV) sampled from Nasdaq Nordic between June and October 2010. Each row is a LOB event: 40 features encoding 10 price-volume pairs across 5 bid and 5 ask levels. Five prediction horizons (k = 1, 2, 3, 5, 10 events ahead) are provided as integer labels derived from the direction of mid-price movement — 0 = down, 1 = stationary, 2 = up.

The train/test split follows a strict day boundary: the first 7 days form the training set, the final 3 days the test set. The split index is the exact cumulative event count at the end of day 7 — never a random index, never shuffled. `StandardScaler` is fit on the training events only; the test set is transformed using those parameters but contributes nothing to scale computation. This mirrors the walk-forward discipline in [AlphaLab](https://github.com/bernardo-guterres/AlphaLab): future data never touches the training distribution in any form.

Windowing applies a 100-event sliding window to the normalised sequences. Each sample is `X[i : i+100]` with label `y[i+99]` — the mid-price movement classification at the final event in the window. No label ever references an index beyond the window end. Temporal order is preserved throughout; `DataLoader` shuffles the training windows across batches, not across time.

---

## Architecture Rationale

The first convolutional kernel has shape (1×2) with stride (1×2). Structurally, this pairs each price column with its adjacent volume column at the same LOB level — the input layout is `ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, …`, and the kernel enforces the price-volume relationship rather than hoping the network discovers it. The same intuition underpins hand-crafted features in [AlphaLab](https://github.com/bernardo-guterres/AlphaLab) — price and volume are paired signals — encoded here directly into the architecture rather than as a preprocessing step.

A single temporal kernel commits to one timescale of LOB dynamics. The Inception module runs three parallel branches with kernel sizes 3×1, 5×1, and max-pooling, then concatenates their 64-channel outputs into 192 channels. The model learns which timescales are informative per horizon without the researcher making that choice. At longer horizons (k=10), slow-moving order flow patterns tend to dominate over tick-level noise; at k=1, the reverse holds. Fixing a single kernel size would force a tradeoff that varies across horizons.

The CNN and Inception blocks extract local spatial and temporal features but have no memory spanning the full 100-event sequence. The LSTM operates over 1880 flattened spatial-temporal tokens — `(94 time steps × 20 spatial positions) × 192 channels` — and maintains a hidden state across the entire sequence. Only the final timestep is passed to the classifier, forcing the model to compress all relevant order flow signal into a single 256-dimensional vector and prioritise patterns that persist to the moment of prediction.

---

## Explainability

Integrated Gradients (Sundararajan et al. 2017) attributes each prediction to individual LOB features by integrating gradients along a straight path from a zero baseline to the input. Unlike SHAP TreeSHAP — which applies only to tree ensembles — IG works directly on the DeepLOB forward pass via PyTorch autograd and satisfies the completeness axiom: attributions sum exactly to `F(x) − F(baseline)` for the target class, giving an honest per-feature accounting with no approximation error in the attribution sum. The same method was applied to neural dimensionality reduction architectures in the XAI-DR thesis; DeepLOB extends it to CNN-LSTM sequence models over limit order book data.

Ask volume at level 1 (`ask_vol_L1`) accounts for 8.9% of mean absolute IG attribution at k=10 — the single most influential feature — with volume features collectively accounting for 83.9% of total attribution versus only 16.1% for price features. The heatmap reveals asymmetric attribution across prediction classes: `ask_vol_L1` carries nearly twice the attribution for downward predictions (0.0050) as for upward ones (0.0025), consistent with the intuition that thinning ask-side liquidity at the best level signals impending sell pressure at longer horizons.

SHAP GradientExplainer broadly agrees — `ask_vol_L1` ranks first (11.5%) and the top-5 features overlap substantially — but pushes the volume/price split further to 99.5%/0.5%, suggesting SHAP is insensitive to price features that IG still captures at ~16%. The disagreement in ask/bid balance (IG: 54.5%/45.5%; SHAP: 68.1%/31.9%) reflects SHAP's approximation error under the gradient-baseline assumption rather than a genuine architectural difference.

---

## Ablation Study

| Model           | Macro F1 | Δ vs Full |
|-----------------|----------|-----------|
| CNN only        | 0.6624   | −12.4%    |
| CNN + Inception | 0.5397   | −28.7%    |
| Full DeepLOB    | 0.7565   | —         |

CNN-only achieves 0.6624 macro F1 — 12.4% below the full model — confirming that spatial price-volume feature extraction alone is insufficient, but provides a reasonable baseline by preserving all 60,160 flattened spatial features. Counterintuitively, CNN + Inception scores lower (0.5397, −28.7%) than CNN-only: the Inception module adds multi-scale temporal features but the Global Average Pooling used in its absence of an LSTM collapses the entire temporal dimension into 192 values, discarding the sequential structure that GAP cannot recover. The result isolates the LSTM's contribution: it is not just an add-on but the load-bearing component — without it, adding Inception's 192-channel output through GAP actively hurts performance relative to keeping the full flattened CNN representation.

---

## Project Structure

```
DeepLOB/
├── deeplob/
│   ├── __init__.py
│   ├── dataset.py       # loading, splitting, normalising, windowing
│   ├── model.py         # CNNBlock, InceptionModule, DeepLOB
│   ├── train.py         # training loop, early stopping, checkpointing
│   ├── evaluate.py      # metrics, benchmark comparison table
│   ├── ablation.py      # CNN-only, CNN+Inception, full model variants
│   ├── explain.py       # Integrated Gradients and SHAP attributions
│   ├── utils.py         # config, seed, device, checkpoint I/O
│   └── serve.py         # FastAPI inference server (POST /predict, GET /health)
├── tests/
│   ├── conftest.py      # shared fixtures
│   ├── test_dataset.py  # 22 tests
│   ├── test_model.py    # 9 tests
│   ├── test_train.py    # 21 tests
│   ├── test_evaluate.py # 8 tests
│   ├── test_ablation.py # 7 tests
│   ├── test_serve.py    # 17 tests
│   └── test_explain.py  # 16 tests
├── data/
│   ├── download.py      # instructions, extraction, verification
│   └── raw/             # FI-2010 .npy files (not tracked)
├── configs/
│   └── default.yaml     # all hyperparameters
├── docs/
│   ├── ARCHITECTURE.md
│   └── DATA_PIPELINE.md
├── outputs/             # checkpoints, logs, results (not tracked)
│   └── plots/           # IG and SHAP attribution figures
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@article{zhang2019deeplob,
  title={DeepLOB: Deep convolutional neural networks for limit order books},
  author={Zhang, Zihao and Zohren, Stefan and Roberts, Stephen},
  journal={IEEE Transactions on Signal Processing},
  volume={67},
  number={11},
  pages={3001--3012},
  year={2019},
  publisher={IEEE}
}
```
