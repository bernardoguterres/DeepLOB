"""Evaluation utilities for DeepLOB — per-class metrics, benchmark comparison, and reporting.

Entry point::

    python -m deeplob.evaluate --config configs/default.yaml --data_dir data/raw/

Computes accuracy, macro/weighted F1, per-class F1, and confusion matrix for each horizon *k*,
compares to published DeepLOB benchmarks, and saves ``results.json``.
"""

import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from deeplob.dataset import get_dataloaders
from deeplob.model import DeepLOB
from deeplob.utils import get_device, load_checkpoint, load_config

__all__ = ["PAPER_BENCHMARKS", "evaluate", "benchmark_table", "run_evaluation"]

# ---------------------------------------------------------------------------
# Published macro-F1 values from Zhang et al. (2019), Table IV
# ---------------------------------------------------------------------------

PAPER_BENCHMARKS: dict[int, float] = {
    1: 0.67,
    2: 0.71,
    3: 0.75,
    5: 0.78,
    10: 0.83,
}


def evaluate(
    model: nn.Module,
    loader: Union[DataLoader, list],
    device: torch.device,
) -> dict:
    """Evaluate a trained DeepLOB model and return a metrics dictionary.

    Args:
        model: Trained model (set to eval mode internally).
        loader: DataLoader (or any iterable of ``(x, y)`` batches).
        device: Compute device to move tensors to.

    Returns:
        Dictionary with keys:

        * ``accuracy`` (float) — overall accuracy.
        * ``macro_f1`` (float) — macro-averaged F1 across all three classes.
        * ``weighted_f1`` (float) — weighted-averaged F1.
        * ``per_class_f1`` (list[float], length 3) — F1 for classes 0, 1, 2.
        * ``confusion_matrix`` (list[list[int]], shape 3×3) — row = true, col = pred.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            if isinstance(y, torch.Tensor):
                y_list = y.cpu().tolist()
            else:
                y_list = list(y)

            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_list)

    labels = [0, 1, 2]
    accuracy: float = accuracy_score(all_labels, all_preds)
    macro_f1: float = f1_score(
        all_labels, all_preds, average="macro", labels=labels, zero_division=0
    )
    weighted_f1: float = f1_score(
        all_labels, all_preds, average="weighted", labels=labels, zero_division=0
    )
    per_class_f1: list[float] = f1_score(
        all_labels, all_preds, average=None, labels=labels, zero_division=0
    ).tolist()
    cm: list[list[int]] = confusion_matrix(all_labels, all_preds, labels=labels).tolist()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def benchmark_table(results_by_k: dict[int, float]) -> str:
    """Render a Markdown comparison table against the paper's published F1 scores.

    Args:
        results_by_k: Mapping from horizon *k* to the achieved macro-F1 score.

    Returns:
        Multi-line string — a valid GitHub-Flavored Markdown table with columns
        ``k | Paper F1 | Ours | Δ``.  The Δ column shows relative improvement
        over the paper as ``"+X.X%"`` (positive) or ``"-X.X%"`` (negative).

    Example::

        | k  | Paper F1 | Ours   | Δ       |
        |----|----------|--------|---------|
        | 1  | 0.670    | 0.682  | +1.8%   |
    """
    header = "| k  | Paper F1 | Ours   | Δ       |"
    sep = "|----|----------|--------|---------|"
    rows = [header, sep]

    for k in sorted(results_by_k):
        ours = results_by_k[k]
        paper = PAPER_BENCHMARKS.get(k, 0.0)
        if paper > 0:
            delta_pct = (ours - paper) / paper * 100.0
        else:
            delta_pct = 0.0
        delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
        rows.append(f"| {k:<2} | {paper:.3f}    | {ours:.4f} | {delta_str:<7} |")

    return "\n".join(rows)


def run_evaluation(
    config_path: str,
    data_dir: str,
    checkpoint_dir: str = "outputs/",
) -> None:
    """Evaluate all prediction horizons and write ``results.json``.

    Loads each ``best_model_k{k}.pt`` checkpoint from *checkpoint_dir*,
    evaluates it on the test split, prints a Markdown comparison table, and
    saves all metrics to ``{checkpoint_dir}/results.json``.

    Args:
        config_path: Path to YAML config file (must contain ``data.horizons``).
        data_dir: Path to FI-2010 ``.npy`` files.
        checkpoint_dir: Directory containing checkpoint files and where
            ``results.json`` will be written (default: ``"outputs/"``).
    """
    config = load_config(config_path)
    device = get_device()
    training_cfg = config["training"]
    model_cfg = config.get("model", {})
    horizons: list[int] = config.get("data", {}).get("horizons", [1, 2, 3, 5, 10])

    all_results: dict[int, dict] = {}
    macro_f1_by_k: dict[int, float] = {}

    for k in horizons:
        ckpt_path = str(Path(checkpoint_dir) / f"best_model_k{k}.pt")
        if not Path(ckpt_path).exists():
            print(f"[k={k}] Checkpoint not found at {ckpt_path} — skipping.")
            continue

        print(f"\n[k={k}] Loading checkpoint from {ckpt_path}")
        model = DeepLOB(
            hidden_size=model_cfg.get("hidden_size", 256),
            num_lstm_layers=model_cfg.get("lstm_layers", 1),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        epoch, val_f1 = load_checkpoint(ckpt_path, model, optimizer)
        print(f"[k={k}] Checkpoint: epoch={epoch}, val_f1={val_f1:.4f}")

        _, test_loader, _ = get_dataloaders(
            data_dir=data_dir,
            k=k,
            batch_size=training_cfg["batch_size"],
            window=training_cfg.get("window", 100),
            train_days=training_cfg.get("train_days", 7),
        )

        metrics = evaluate(model, test_loader, device)
        all_results[k] = metrics
        macro_f1_by_k[k] = metrics["macro_f1"]

        print(
            f"[k={k}] accuracy={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )
        print(f"[k={k}] per_class_f1={[round(v, 4) for v in metrics['per_class_f1']]}")

    if macro_f1_by_k:
        print("\n" + benchmark_table(macro_f1_by_k))

        out_path = Path(checkpoint_dir) / "results.json"
        # JSON keys must be strings; convert int keys
        serialisable = {str(k): v for k, v in all_results.items()}
        try:
            with out_path.open("w") as fh:
                json.dump(serialisable, fh, indent=2)
            print(f"\nResults saved to {out_path}")
        except OSError as exc:
            print(f"Warning: failed to save results to {out_path}: {exc}")
    else:
        print("No checkpoints found — nothing to evaluate.")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate DeepLOB checkpoints against FI-2010 test set."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data_dir", default="data/raw/", help="Path to FI-2010 .npy files")
    parser.add_argument(
        "--checkpoint_dir", default="outputs/", help="Directory with .pt checkpoints"
    )
    args = parser.parse_args()
    run_evaluation(args.config, args.data_dir, args.checkpoint_dir)
