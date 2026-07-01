"""
DeepLOB — Kaggle Ablation Script
==================================
Trains CNN-only and CNN+Inception variants for k=10.
Full DeepLOB result is loaded from outputs/results.json (k=10 macro_f1).

SETUP:
1. Same dataset as kaggle_train.py (bernardoguterresDeepLOb)
2. Save Version → Save & Run All (Commit)

Outputs saved to /kaggle/working/outputs/ablation/
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


def _probe_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1).cuda()
        _ = t + 1
        del t
        return True
    except Exception as e:
        print(f"WARNING: CUDA not usable — {e}")
        return False


device = torch.device("cuda" if _probe_cuda() else "cpu")
print(f"Using device: {device}")


# ── Data path discovery ───────────────────────────────────────────────────────
def find_data_dir(base: str = "/kaggle/input") -> str:
    base_path = Path(base)
    print(f"Kaggle input contents: {sorted(p.name for p in base_path.iterdir())}")
    npy_files = sorted(base_path.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {base}")
    found = str(npy_files[0].parent) + "/"
    print(f"Found {len(npy_files)} .npy files in: {found}")
    return found


DATA_DIR = find_data_dir()
OUTPUT_DIR = "/kaggle/working/outputs/ablation/"

# ── Config ────────────────────────────────────────────────────────────────────
K = 10


def _load_full_deeplob_f1(results_json: str = "/kaggle/working/outputs/results.json") -> float:
    """Read the k=10 macro-F1 from the completed training results file."""
    try:
        with open(results_json) as fh:
            data = json.load(fh)
        return float(data["10"]["macro_f1"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not read Full DeepLOB k=10 macro_f1 from {results_json}: {exc}. "
            "Run kaggle_train.py first so the ablation deltas are computed against a "
            "real baseline, not a stale hardcoded number."
        ) from exc


FULL_DEEPLOB_F1 = _load_full_deeplob_f1()

CFG = {
    "seed": 42,
    "lr": 0.01,
    "adam_eps": 1.0,
    "batch_size": 32,
    "window": 100,
    "train_days": 7,
    "epochs": 50,
    "patience": 20,
    "hidden_size": 64,
}


# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(CFG["seed"])

# ── Dataset ───────────────────────────────────────────────────────────────────
_K_MAP = {1: 40, 2: 41, 3: 42, 5: 43, 10: 44}


def get_dataloaders(data_dir, k, batch_size, window=100, train_days=7):
    path = Path(data_dir)
    npy_files = sorted(path.glob("*.npy"))
    assert npy_files, f"No .npy files in {data_dir}"
    label_col = _K_MAP[k]
    x_parts, y_parts, boundaries, cumulative = [], [], [], 0
    for fp in npy_files:
        arr = np.load(fp)
        x_parts.append(arr[:, :40].astype(np.float64))
        y_parts.append(arr[:, label_col].astype(np.int64) - 1)
        cumulative += arr.shape[0]
        boundaries.append(cumulative)
    X = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    split = boundaries[train_days - 1]
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def make_windows(X, y):
        Xw = sliding_window_view(X, (window, X.shape[1])).squeeze(1)
        return Xw, y[window - 1 :]

    X_tr_w, y_tr_w = make_windows(X_train, y_train)
    X_te_w, y_te_w = make_windows(X_test, y_test)

    class LOBDataset(Dataset):
        def __init__(self, X, y):
            self._X = X
            self._y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

        def __len__(self):
            return len(self._y)

        def __getitem__(self, i):
            x = torch.from_numpy(np.array(self._X[i], dtype=np.float32)).unsqueeze(0)
            return x, self._y[i]

    train_loader = DataLoader(
        LOBDataset(X_tr_w, y_tr_w),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        LOBDataset(X_te_w, y_te_w),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    n = len(y_tr_w)
    counts = np.bincount(y_tr_w, minlength=3)
    class_weights = torch.tensor(n / (3.0 * counts), dtype=torch.float32)
    return train_loader, test_loader, class_weights


# ── Model variants ────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

    def forward(self, x):
        return self.layers(x)


class InceptionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.branch_c = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)


class CNNOnlyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = CNNBlock()
        self.fc = nn.Linear(32 * 94 * 20, num_classes)

    def forward(self, x):
        return self.fc(self.cnn(x).flatten(start_dim=1))


class CNNInceptionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = CNNBlock()
        self.inception = InceptionModule()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.inception(self.cnn(x))
        return self.fc(self.gap(x).flatten(start_dim=1))


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


def validate(model, loader, criterion):
    model.eval()
    total, n, preds, labels = 0.0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += criterion(logits, y).item()
            n += 1
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    return total / n, f1_score(labels, preds, average="macro", zero_division=0)


# ── Main ablation loop ────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_loader, test_loader, class_weights = get_dataloaders(
    DATA_DIR, K, CFG["batch_size"], CFG["window"], CFG["train_days"]
)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

variants = [
    ("CNN only", CNNOnlyModel()),
    ("CNN + Inception", CNNInceptionModel()),
]

results = {"Full DeepLOB": FULL_DEEPLOB_F1}  # already trained to convergence

for name, model in variants:
    print(f"\n{'='*50}")
    print(f"Variant: {name}")
    print(f"{'='*50}")
    set_seed(CFG["seed"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], eps=CFG["adam_eps"])
    ckpt_path = (
        Path(OUTPUT_DIR) / f"ablation_{name.lower().replace(' ', '_').replace('+', 'plus')}_k{K}.pt"
    )

    best_f1, no_improve = -1.0, 0

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1 = validate(model, test_loader, criterion)

        print(
            f"  Epoch {epoch:3d}/{CFG['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_f1": val_f1}, ckpt_path
            )
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

    results[name] = round(best_f1, 4)
    print(f"{name} done. Best val F1: {best_f1:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
full_f1 = results["Full DeepLOB"]
print("\n" + "=" * 50)
print("ABLATION RESULTS")
print("=" * 50)
print(f"{'Model':<18} {'Macro F1':>9} {'Δ vs Full':>10}")
print("-" * 40)
for name in ["CNN only", "CNN + Inception", "Full DeepLOB"]:
    f1 = results[name]
    if name == "Full DeepLOB":
        delta = "—"
    else:
        delta = f"{(f1 - full_f1) / full_f1 * 100:+.1f}%"
    print(f"{name:<18} {f1:>9.4f} {delta:>10}")

out_path = Path(OUTPUT_DIR) / "ablation_results.json"
with out_path.open("w") as fh:
    json.dump({"k": K, "macro_f1": results}, fh, indent=2)
print(f"\nSaved → {out_path}")
