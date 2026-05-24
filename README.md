# DeepLOB

PyTorch reimplementation of [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1902.09450) (Zhang, Zohren & Roberts, 2019) — mid-price movement classification from limit order book snapshots using a CNN + Inception + LSTM architecture on the FI-2010 benchmark.

---

## Results

| k  | Paper (Zhang et al. 2019) | Ours | Δ   |
|----|--------------------------|------|-----|
| 1  | 0.67                     | TBD  | TBD |
| 2  | 0.71                     | TBD  | TBD |
| 3  | 0.75                     | TBD  | TBD |
| 5  | 0.78                     | TBD  | TBD |
| 10 | 0.83                     | TBD  | TBD |

Metric: macro F1. Dataset: FI-2010, day-based 7/3 split.

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
│  hidden_size=256, layers=1         │  global sequence memory
│  take final timestep [:, -1, :]   │
└────────────────────────────────────┘
        │ (B, 256)
        ▼
   Linear(256 → 3)
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

### 8. Tests

```bash
pytest tests/ -v --cov=deeplob
```

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

*Findings to be updated after running `python -m deeplob.explain --k 10 --method ig`.* Placeholder: ask price at level 1 accounts for X% of mean absolute attribution at k=10. Attribution shifts toward deeper LOB levels at shorter horizons (k=1), consistent with the intuition that immediate microstructure dominates short-horizon prediction.

---

## Ablation Study

| Model           | Macro F1 | Δ vs Full |
|-----------------|----------|-----------|
| CNN only        | TBD      | TBD       |
| CNN + Inception | TBD      | TBD       |
| Full DeepLOB    | TBD      | —         |

Each row isolates one architectural component. CNN-only measures whether spatial price-volume pairing alone — without any temporal modelling — is sufficient to classify mid-price movement. CNN + Inception without LSTM tests whether local multi-scale features, captured across three parallel temporal kernel sizes, suffice without global sequence memory. The gap between each successive row is the empirical contribution of that component: Inception's multi-scale capture over the CNN baseline, and the LSTM's sequence memory over Inception alone.

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
│   └── utils.py         # config, seed, device, checkpoint I/O
├── tests/
│   ├── conftest.py      # shared fixtures
│   ├── test_dataset.py  # 16 tests
│   ├── test_model.py    # 9 tests
│   ├── test_train.py    # 10 tests
│   ├── test_evaluate.py # 8 tests
│   ├── test_ablation.py # 1 test
│   └── test_explain.py  # 6 tests
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
