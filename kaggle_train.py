"""
DeepLOB — Kaggle Training Script
=================================
Self-contained: no local package needed.

SETUP (do this before running):
1. Create a Kaggle dataset from your local data/raw/ folder:
   - Go to kaggle.com → Datasets → New Dataset
   - Upload all 10 .npy files from data/raw/
   - Name it "deeplob-fi2010"
2. In your Kaggle notebook: Add Data → Your Datasets → deeplob-fi2010
3. Run all cells.

Outputs saved to /kaggle/working/outputs/
"""

# ── Cell 1: Dependencies ─────────────────────────────────────────────────────
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


# Sanity-check that the GPU is actually usable (P100 + PyTorch>=2.5 is not)
def _probe_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1).cuda()
        _ = t + 1  # triggers a real kernel call
        del t
        return True
    except Exception as e:
        print(f"WARNING: CUDA detected but not usable — {e}")
        print("         Falling back to CPU. Switch Kaggle accelerator to T4 x2 for GPU speed.")
        return False


_cuda_ok = _probe_cuda()
device = torch.device("cuda" if _cuda_ok else "cpu")
print(f"Using device: {device}")


# ── Cell 1b: Discover data path ───────────────────────────────────────────────
def find_data_dir(base: str = "/kaggle/input") -> str:
    """Walk /kaggle/input to find the directory that contains .npy files."""
    base_path = Path(base)
    print(f"\nKaggle input contents: {sorted(p.name for p in base_path.iterdir())}")
    # Search recursively for any .npy file
    npy_files = sorted(base_path.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found anywhere under {base}")
    # Return the directory containing the first .npy file
    found = str(npy_files[0].parent) + "/"
    print(f"Found {len(npy_files)} .npy files in: {found}")
    return found


# ── Cell 2: Config ───────────────────────────────────────────────────────────
DATA_DIR = find_data_dir()  # auto-discovers correct path under /kaggle/input
OUTPUT_DIR = "/kaggle/working/outputs/"
HORIZONS = [1, 2, 3, 5, 10]

CFG = {
    "seed": 42,
    "lr": 0.01,  # paper: 0.01
    "adam_eps": 1.0,  # paper: epsilon=1
    "batch_size": 32,  # paper: 32
    "window": 100,
    "train_days": 7,
    "epochs": 50,
    "patience": 20,  # paper: 20
    "hidden_size": 64,  # paper: 64
    "lstm_layers": 1,
}


# ── Cell 3: Seed ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(CFG["seed"])

# ── Cell 4: Dataset ──────────────────────────────────────────────────────────
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


# ── Cell 5: Model ────────────────────────────────────────────────────────────
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


class DeepLOB(nn.Module):
    def __init__(self, hidden_size=64, num_lstm_layers=1, num_classes=3):
        super().__init__()
        self.cnn = CNNBlock()
        self.inception = InceptionModule()
        self.lstm = nn.LSTM(192, hidden_size, num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.cnn(x)  # (B, 32, 94, 20)
        x = self.inception(x)  # (B, 192, 94, 20)
        x = x.permute(0, 2, 3, 1)  # (B, 94, 20, 192)
        x = x.reshape(B, 94 * 20, 192)  # (B, 1880, 192)
        x, _ = self.lstm(x)  # (B, 1880, hidden)
        x = x[:, -1, :]  # (B, hidden)
        return self.fc(x)  # (B, 3)


# ── Cell 6: Training functions ───────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
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


def validate(model, loader, criterion, device):
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
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return total / n, f1


# ── Cell 7: Main training loop ───────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
results = {}

for k in HORIZONS:
    print(f"\n{'='*50}")
    print(f"Training k={k}  (paper k={k*10})")
    print(f"{'='*50}")

    set_seed(CFG["seed"])
    train_loader, test_loader, class_weights = get_dataloaders(
        DATA_DIR, k, CFG["batch_size"], CFG["window"], CFG["train_days"]
    )

    model = DeepLOB(CFG["hidden_size"], CFG["lstm_layers"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], eps=CFG["adam_eps"])

    log_path = Path(OUTPUT_DIR) / f"training_log_k{k}.jsonl"
    ckpt_path = Path(OUTPUT_DIR) / f"best_model_k{k}.pt"

    best_f1, best_epoch, no_improve = -1.0, 0, 0

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = validate(model, test_loader, criterion, device)

        print(
            f"  Epoch {epoch:3d}/{CFG['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
        )

        with log_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "val_f1": round(val_f1, 6),
                    }
                )
                + "\n"
            )

        if val_f1 > best_f1:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

    results[k] = {"best_val_f1": round(best_f1, 4), "best_epoch": best_epoch}
    print(f"k={k} done. Best val F1: {best_f1:.4f} at epoch {best_epoch}")

# ── Cell 8: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
paper = {1: 77.66, 5: 74.96, 10: 76.58}  # Setup 1 F1%
for k, r in results.items():
    ours = r["best_val_f1"] * 100
    ref = paper.get(k, "—")
    gap = f"{ours - ref:+.1f}%" if isinstance(ref, float) else ""
    print(f"  k={k:2d} (paper k={k*10:3d}):  ours={ours:.2f}%  paper={ref}%  {gap}")
