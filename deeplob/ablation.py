"""Ablation study: CNN-only vs CNN+Inception vs Full DeepLOB.

Trains three model variants on a single prediction horizon and compares their
macro F1 scores to isolate the contribution of each architectural component:

* **CNN only** — baseline spatial feature extractor, no temporal modelling.
* **CNN + Inception** — adds multi-scale temporal patterns, still no LSTM.
* **Full DeepLOB** — complete architecture (CNN + Inception + LSTM).

Entry point::

    python -m deeplob.ablation --config configs/default.yaml --data_dir data/raw/ --k 10

Results are saved to ``outputs/ablation/ablation_results.json`` and a Markdown
comparison table is printed to stdout.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from deeplob.dataset import get_dataloaders
from deeplob.model import (
    CNN_OUT_CHANNELS,
    CNN_OUT_HEIGHT,
    CNN_OUT_WIDTH,
    CNNBlock,
    DeepLOB,
    InceptionModule,
)
from deeplob.train import run_epoch
from deeplob.utils import get_device, load_checkpoint, load_config, set_seed

__all__ = ["CNNOnlyModel", "CNNInceptionModel", "run_ablation"]


class CNNOnlyModel(nn.Module):
    """Ablation: CNN block only — no Inception, no LSTM.

    Applies :class:`~deeplob.model.CNNBlock` then flattens the spatial
    dimensions and projects directly to class logits with a single linear layer.

    Used to isolate the contribution of the Inception multi-scale block and the
    LSTM relative to a pure CNN baseline.

    Input shape:  ``(batch, 1, 100, 40)``
    Output shape: ``(batch, num_classes)`` — raw logits
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.cnn = CNNBlock()
        # After CNNBlock: (B, 32, 94, 20) → flatten → 32 × 94 × 20 = 60 160
        self._flat_size = CNN_OUT_CHANNELS * CNN_OUT_HEIGHT * CNN_OUT_WIDTH
        self.fc = nn.Linear(self._flat_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CNN, flatten, and classify.

        Args:
            x: Input tensor of shape ``(batch, 1, 100, 40)``.

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        x = self.cnn(x)  # (B, 32, 94, 20)
        x = x.flatten(start_dim=1)  # (B, 60160)
        return self.fc(x)  # (B, num_classes)


class CNNInceptionModel(nn.Module):
    """Ablation: CNN + Inception, no LSTM.

    Applies :class:`~deeplob.model.CNNBlock` followed by
    :class:`~deeplob.model.InceptionModule`, then uses **global average pooling**
    over the temporal dimension before a linear classifier.

    Used to isolate the contribution of the LSTM relative to a model that already
    has multi-scale Inception features.

    Input shape:  ``(batch, 1, 100, 40)``
    Output shape: ``(batch, num_classes)`` — raw logits
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.cnn = CNNBlock()
        self.inception = InceptionModule()
        # After Inception: (B, 192, 94, 20)
        # Global average pool over (H=94, W=20) → (B, 192)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CNN + Inception, global-average-pool, and classify.

        Args:
            x: Input tensor of shape ``(batch, 1, 100, 40)``.

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        x = self.cnn(x)  # (B, 32, 94, 20)
        x = self.inception(x)  # (B, 192, 94, 20)
        x = self.gap(x)  # (B, 192, 1, 1)
        x = x.flatten(start_dim=1)  # (B, 192)
        return self.fc(x)  # (B, num_classes)


def _ablation_table(results: dict[str, float]) -> str:
    """Render a Markdown table comparing ablation variants against the full model.

    Args:
        results: Mapping ``{model_name: macro_f1}``. Must contain the key
            ``"Full DeepLOB"`` for the baseline delta computation.

    Returns:
        Multi-line Markdown table string with columns
        ``Model | Macro F1 | Δ vs Full``.
    """
    full_f1 = results.get("Full DeepLOB", 0.0)
    header = "| Model           | Macro F1 | Δ vs Full |"
    sep = "|-----------------|----------|-----------|"
    rows = [header, sep]

    order = ["CNN only", "CNN + Inception", "Full DeepLOB"]
    for name in order:
        if name not in results:
            continue
        f1 = results[name]
        if name == "Full DeepLOB":
            delta_str = "—"
        else:
            delta_pct = (f1 - full_f1) / full_f1 * 100.0 if full_f1 > 0 else 0.0
            delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
        rows.append(f"| {name:<15} | {f1:.3f}    | {delta_str:<9} |")

    return "\n".join(rows)


def run_ablation(
    config_path: str,
    data_dir: str,
    k: int = 10,
    output_dir: str = "outputs/ablation/",
    pretrained_dir: str = "outputs/",
) -> None:
    """Train all three variants on horizon *k* and compare macro F1.

    CNN-only and CNN+Inception are always trained from scratch.  For Full
    DeepLOB, if ``{pretrained_dir}/best_model_k{k}.pt`` already exists the
    checkpoint's saved ``val_f1`` is reused and retraining is skipped — this
    avoids redundant 20-hour runs when the full model has already been trained
    to convergence.

    Results are written to ``{output_dir}/ablation_results.json`` and a
    Markdown comparison table is printed to stdout.

    Args:
        config_path: Path to YAML config (same schema as ``configs/default.yaml``).
        data_dir: Path to FI-2010 ``.npy`` files.
        k: Prediction horizon to run the ablation on (default 10).
        output_dir: Directory for ablation checkpoints and the JSON results file.
        pretrained_dir: Directory to look for an existing
            ``best_model_k{k}.pt`` for the Full DeepLOB variant.
            Set to ``""`` to always retrain from scratch.
    """
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device()

    training_cfg = config["training"]
    model_cfg = config.get("model", {})

    train_loader, test_loader, class_weights = get_dataloaders(
        data_dir=data_dir,
        k=k,
        batch_size=training_cfg["batch_size"],
        window=training_cfg.get("window", 100),
        train_days=training_cfg.get("train_days", 7),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    n_epochs = training_cfg.get("epochs", 50)
    patience = training_cfg.get("patience", 10)

    variants: list[tuple[str, nn.Module]] = [
        ("CNN only", CNNOnlyModel()),
        ("CNN + Inception", CNNInceptionModel()),
        (
            "Full DeepLOB",
            DeepLOB(
                hidden_size=model_cfg.get("hidden_size", 256),
                num_lstm_layers=model_cfg.get("lstm_layers", 1),
            ),
        ),
    ]

    results: dict[str, float] = {}

    for name, model in variants:
        print(f"\n{'─' * 60}")
        print(f"  Variant: {name}")
        print(f"{'─' * 60}")

        model = model.to(device)
        adam_eps = training_cfg.get("adam_eps", 1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"], eps=adam_eps)
        ckpt_path = str(
            out / f"ablation_{name.lower().replace(' ', '_').replace('+', 'plus')}_k{k}.pt"
        )

        # --- Full DeepLOB: reuse existing checkpoint if available ----------
        if name == "Full DeepLOB" and pretrained_dir:
            pretrained_ckpt = Path(pretrained_dir) / f"best_model_k{k}.pt"
            if pretrained_ckpt.exists():
                _, loaded_f1 = load_checkpoint(str(pretrained_ckpt), model, optimizer)
                results[name] = loaded_f1
                print(
                    f"  [{name}] Loaded pretrained checkpoint from {pretrained_ckpt} "
                    f"(val_f1={loaded_f1:.4f}) — skipping retraining."
                )
                continue
        # --- Train from scratch -------------------------------------------
        best_val_f1 = -1.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(1, n_epochs + 1):
            result = run_epoch(
                model,
                train_loader,
                test_loader,
                optimizer,
                criterion,
                device,
                epoch,
                best_val_f1,
                best_epoch,
                no_improve,
                patience,
                ckpt_path,
            )
            best_val_f1, best_epoch, no_improve = (
                result.best_val_f1,
                result.best_epoch,
                result.no_improve,
            )

            print(
                f"  [{name}] Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={result.train_loss:.4f}  val_loss={result.val_loss:.4f}  "
                f"val_f1={result.val_f1:.4f}"
            )

            if result.should_stop:
                print(f"  [{name}] Early stopping at epoch {epoch}.")
                break

        results[name] = best_val_f1
        print(f"  [{name}] Best macro F1: {best_val_f1:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Ablation results")
    print(f"{'─' * 60}")
    print(_ablation_table(results))

    results_path = out / "ablation_results.json"
    try:
        with results_path.open("w") as fh:
            json.dump({"k": k, "macro_f1": results}, fh, indent=2)
        print(f"\nResults saved to {results_path}")
    except OSError as exc:
        print(f"Warning: failed to save ablation results to {results_path}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Ablation study: CNN-only vs CNN+Inception vs Full DeepLOB."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data_dir", default="data/raw/", help="Path to FI-2010 .npy files")
    parser.add_argument("--k", type=int, default=10, help="Prediction horizon (default 10)")
    parser.add_argument("--output_dir", default="outputs/ablation/", help="Output directory")
    parser.add_argument(
        "--pretrained_dir",
        default="outputs/",
        help="Dir containing best_model_k{k}.pt — reused for Full DeepLOB to skip retraining",
    )
    args = parser.parse_args()
    run_ablation(args.config, args.data_dir, args.k, args.output_dir, args.pretrained_dir)
