"""Explainability for DeepLOB — Integrated Gradients and SHAP attributions.

Answers the question "which LOB levels drive mid-price predictions?" by
attributing the model's output to each of the 40 input features.

Two methods are provided:

* **Integrated Gradients** (Sundararajan et al., 2017) — exact completeness
  axiom, works directly through PyTorch autograd, no model modification needed.
* **SHAP GradientExplainer** — faster but approximate; useful for full
  test-set summaries.

LOB feature naming convention (40 features, 0-indexed):
  * ask_price_L1 … ask_price_L10 (even indices 0, 2, … 18)
  * ask_vol_L1   … ask_vol_L10   (odd  indices 1, 3, … 19)
  * bid_price_L1 … bid_price_L10 (even indices 20, 22, … 38)
  * bid_vol_L1   … bid_vol_L10   (odd  indices 21, 23, … 39)

Entry point::

    python -m deeplob.explain --config configs/default.yaml \\
        --data_dir data/raw/ --k 10 --method ig
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeplob.dataset import get_dataloaders
from deeplob.model import DeepLOB
from deeplob.utils import get_device, load_checkpoint, load_config

__all__ = [
    "FEATURE_NAMES",
    "integrated_gradients",
    "batch_integrated_gradients",
    "shap_summary",
    "plot_feature_importance",
    "plot_class_attribution_heatmap",
    "run_explanation",
]

# ---------------------------------------------------------------------------
# Feature names — 40 LOB features following FI-2010 column layout
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    f"{side}_{'price' if i % 2 == 0 else 'vol'}_L{(i // 2) + 1}"
    for side in ["ask", "bid"]
    for i in range(20)
]
# Produces: ask_price_L1, ask_vol_L1, ..., ask_vol_L10,
#           bid_price_L1, bid_vol_L1, ..., bid_vol_L10  (40 total)


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for a single sample.

    Attributes the prediction to each input feature by integrating
    gradients along a straight-line path from *baseline* to *x*.
    Satisfies the completeness axiom: the sum of all raw attributions
    (before window averaging) equals ``F(x)[target] - F(baseline)[target]``.

    Args:
        model: Trained DeepLOB model in eval mode.
        x: Input tensor, shape ``(1, 1, 100, 40)``. Single sample only.
        target_class: Class index to explain (0=down, 1=stationary, 2=up).
        baseline: Reference input, shape ``(1, 1, 100, 40)``.
            Defaults to all-zeros (zero LOB state).
        n_steps: Number of interpolation steps (default 50).
            Higher gives a more accurate integral. 50 is sufficient
            for most cases; use 200+ for precise completeness checks.
        device: Compute device. Defaults to the model's parameter device.

    Returns:
        Attribution tensor of shape ``(40,)`` — one value per LOB feature,
        averaged over the 100-event window.  Positive values push the
        prediction toward *target_class*; negative values push away.

    Notes:
        Completeness check: ``attrs.sum() * 100`` should equal
        ``F(x)[target_class] - F(baseline)[target_class]`` to within
        the integration error (decreases as ``O(1/n_steps)``).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = x.to(device)

    if baseline is None:
        baseline = torch.zeros_like(x)
    baseline = baseline.to(device)

    # delta: path direction; detach so it's not part of the computation graph
    delta = (x - baseline).detach()  # (1, 1, 100, 40)

    # Accumulate gradients at n_steps points along baseline → x
    alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
    accumulated_grads = torch.zeros_like(delta)

    for alpha in alphas:
        # Construct interpolated input; requires_grad for autograd
        interp = (baseline + alpha * delta).detach().requires_grad_(True)
        with torch.enable_grad():
            output = model(interp)  # (1, num_classes)
            score = output[0, target_class]
            (grad,) = torch.autograd.grad(score, interp)
        accumulated_grads = accumulated_grads + grad.detach()

    # Average gradients (Riemann sum approximation of the integral)
    avg_grads = accumulated_grads / n_steps  # (1, 1, 100, 40)

    # IG = avg_grads × (x − baseline)  [completeness: sum = F(x)−F(baseline)]
    ig = avg_grads * delta  # (1, 1, 100, 40)

    # Squeeze batch+channel dims, then average over the window (temporal) dim
    ig = ig.squeeze(0).squeeze(0)  # (100, 40)
    ig = ig.mean(dim=0)  # (40,) — one value per LOB feature

    return ig.cpu().float()


def batch_integrated_gradients(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 200,
    n_steps: int = 50,
) -> dict:
    """Compute Integrated Gradients attributions over a sample of the test set.

    Iterates the loader and computes IG for each sample w.r.t. the model's
    predicted class (argmax). Stops after *n_samples* samples.

    Args:
        model: Trained DeepLOB model.
        loader: Test DataLoader.
        device: Compute device.
        n_samples: Number of samples to explain (default 200).
            The full test set can be slow — 200 gives a representative summary.
        n_steps: IG interpolation steps per sample (default 50).

    Returns:
        Dictionary with keys:

        * ``attributions``: :class:`numpy.ndarray`, shape ``(n_samples, 40)``.
          Mean IG attribution per LOB feature per sample.
        * ``labels``: :class:`numpy.ndarray`, shape ``(n_samples,)``.
          True class labels.
        * ``predictions``: :class:`numpy.ndarray`, shape ``(n_samples,)``.
          Model-predicted class labels.
        * ``feature_names``: ``list[str]`` of length 40 (``FEATURE_NAMES``).
    """
    model.eval()
    all_attrs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_preds: list[int] = []
    count = 0

    for x_batch, y_batch in loader:
        if count >= n_samples:
            break
        for i in range(x_batch.shape[0]):
            if count >= n_samples:
                break
            xi = x_batch[i : i + 1].to(device)  # (1, 1, 100, 40)
            yi = int(y_batch[i].item())

            with torch.no_grad():
                pred = int(model(xi).argmax(dim=1).item())

            attrs = integrated_gradients(model, xi, pred, n_steps=n_steps, device=device)
            all_attrs.append(attrs.numpy())
            all_labels.append(yi)
            all_preds.append(pred)
            count += 1

    return {
        "attributions": np.array(all_attrs),  # (n_samples, 40)
        "labels": np.array(all_labels),  # (n_samples,)
        "predictions": np.array(all_preds),  # (n_samples,)
        "feature_names": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# SHAP GradientExplainer
# ---------------------------------------------------------------------------


def shap_summary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_background: int = 100,
    n_explain: int = 200,
) -> dict:
    """Compute SHAP GradientExplainer attributions over test samples.

    Faster than Integrated Gradients but approximate. Recommended for
    full test-set summary plots where per-sample precision is less critical.

    Args:
        model: Trained DeepLOB model.
        loader: Test DataLoader (used for both background and explain batches).
        device: Compute device.
        n_background: Number of background samples used as SHAP baseline
            (default 100). Larger → more stable but slower.
        n_explain: Number of test samples to explain (default 200).

    Returns:
        Dictionary with the same keys as :func:`batch_integrated_gradients`:
        ``attributions``, ``labels``, ``predictions``, ``feature_names``.

    Raises:
        ImportError: If ``shap`` is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for shap_summary: pip install shap") from exc

    model.eval()
    needed = n_background + n_explain

    # Collect enough samples from the loader
    all_x: list[torch.Tensor] = []
    all_y: list[int] = []
    for x_batch, y_batch in loader:
        all_x.append(x_batch)
        all_y.extend(y_batch.tolist())
        if sum(t.shape[0] for t in all_x) >= needed:
            break

    all_x_tensor = torch.cat(all_x, dim=0)[:needed].to(device)
    all_y_arr = np.array(all_y[:needed])

    bg = all_x_tensor[:n_background]
    explain_x = all_x_tensor[n_background : n_background + n_explain]
    explain_y = all_y_arr[n_background : n_background + n_explain]

    # GradientExplainer: background establishes E[F(x)]
    explainer = shap.GradientExplainer(model, bg)
    # shap_values: list of 3 arrays, each shape (n_explain, 1, 100, 40)
    shap_vals = explainer.shap_values(explain_x)

    with torch.no_grad():
        preds = model(explain_x).argmax(dim=1).cpu().numpy()

    actual_n = len(preds)
    attrs = np.zeros((actual_n, 40))

    if isinstance(shap_vals, list):
        # Old shap API (<0.46): list of n_classes arrays, each (n_explain, 1, 100, 40)
        for i, pred in enumerate(preds):
            sv = np.asarray(shap_vals[int(pred)])[i]  # (1, 100, 40) or (100, 40)
            if sv.ndim == 3:
                sv = sv.squeeze(0)  # (100, 40)
            attrs[i] = sv.mean(axis=0)  # (40,)
    else:
        # New shap API (>=0.46): shape (n_explain, 1, 100, 40, n_classes)
        sv_arr = np.asarray(shap_vals)
        for i, pred in enumerate(preds):
            sv = sv_arr[i, 0, :, :, int(pred)]  # (100, 40)
            attrs[i] = sv.mean(axis=0)

    return {
        "attributions": attrs,
        "labels": explain_y,
        "predictions": preds,
        "feature_names": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _use_agg_backend() -> None:
    """Switch matplotlib to the non-interactive Agg backend for file saving."""
    import matplotlib

    try:
        matplotlib.use("Agg")
    except (AttributeError, ImportError, RuntimeError):
        pass  # Backend already locked — the figure will still save correctly


def plot_feature_importance(
    attributions: np.ndarray,
    feature_names: list[str],
    title: str = "LOB Feature Importance (Integrated Gradients)",
    top_k: int = 15,
    save_path: str | None = None,
) -> None:
    """Plot mean absolute IG attribution per LOB feature (horizontal bar chart).

    Features are coloured by type:
    * ``ask_price`` → steelblue
    * ``ask_vol``   → lightblue
    * ``bid_price`` → coral
    * ``bid_vol``   → lightsalmon

    Args:
        attributions: IG or SHAP values, shape ``(n_samples, 40)``.
        feature_names: List of 40 feature name strings (use ``FEATURE_NAMES``).
        title: Plot title.
        top_k: Number of most-important features to display (default 15).
        save_path: Destination path for the PNG.  If ``None``, saves to
            ``outputs/plots/feature_importance.png``.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    _use_agg_backend()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from exc

    if save_path is None:
        save_path = "outputs/plots/feature_importance.png"

    mean_abs = np.abs(attributions).mean(axis=0)  # (40,)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_idx = sorted_idx[:top_k]

    top_names = [feature_names[i] for i in top_idx]
    top_values = mean_abs[top_idx]

    _color_map = {
        "ask_price": "steelblue",
        "ask_vol": "lightblue",
        "bid_price": "coral",
        "bid_vol": "lightsalmon",
    }

    def _feature_color(name: str) -> str:
        for prefix, color in _color_map.items():
            if name.startswith(prefix):
                return color
        return "grey"

    colors = [_feature_color(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_k), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1])
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean |Attribution|")
    ax.set_title(title)
    plt.tight_layout()

    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        print(f"Warning: failed to save figure to {save_path}: {exc}")
    finally:
        plt.close(fig)


def plot_class_attribution_heatmap(
    attributions: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    save_path: str | None = None,
) -> None:
    """Heatmap of mean attribution per class × feature.

    Rows = predicted classes (Down, Stationary, Up).
    Columns = top 20 features by mean absolute attribution.

    Visualises whether the model uses different LOB levels when predicting
    different price directions — the central XAI research question.

    Args:
        attributions: Shape ``(n_samples, 40)``.
        labels: True class labels, shape ``(n_samples,)``.
        feature_names: List of 40 feature name strings.
        save_path: Optional PNG destination.  Defaults to
            ``outputs/plots/class_attribution_heatmap.png``.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    _use_agg_backend()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from exc

    if save_path is None:
        save_path = "outputs/plots/class_attribution_heatmap.png"

    # Select top 20 features by mean absolute attribution
    mean_abs = np.abs(attributions).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[::-1][:20]
    top20_names = [feature_names[i] for i in top20_idx]

    # Mean attribution per class
    class_names = ["Down", "Stationary", "Up"]
    heatmap_data = np.zeros((3, 20))
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            heatmap_data[c] = attributions[mask][:, top20_idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(20))
    ax.set_xticklabels(top20_names, rotation=45, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax, label="Mean Attribution")
    ax.set_title("Mean Attribution per Class × Feature")
    plt.tight_layout()

    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        print(f"Warning: failed to save figure to {save_path}: {exc}")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Full explanation pipeline
# ---------------------------------------------------------------------------


def _load_model_and_test_loader(
    config: dict,
    data_dir: str,
    checkpoint_dir: str,
    k: int,
    device: torch.device,
) -> tuple[nn.Module, DataLoader]:
    """Build the test DataLoader and load ``best_model_k{k}.pt`` for explanation.

    Args:
        config: Parsed YAML config (``load_config`` output).
        data_dir: Path to FI-2010 ``.npy`` files.
        checkpoint_dir: Directory containing ``best_model_k{k}.pt``.
        k: Prediction horizon.
        device: Compute device the model is moved to.

    Returns:
        Tuple of ``(model, test_loader)``. The model is loaded from checkpoint
        and left in whatever mode ``load_checkpoint`` sets (not forced to eval).

    Raises:
        FileNotFoundError: If the checkpoint for horizon *k* is not found.
    """
    training_cfg = config["training"]
    model_cfg = config.get("model", {})

    _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        k=k,
        batch_size=training_cfg["batch_size"],
        window=training_cfg.get("window", 100),
        train_days=training_cfg.get("train_days", 7),
    )

    ckpt_path = str(Path(checkpoint_dir) / f"best_model_k{k}.pt")
    model = DeepLOB(
        hidden_size=model_cfg.get("hidden_size", 256),
        num_lstm_layers=model_cfg.get("lstm_layers", 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    epoch, val_f1 = load_checkpoint(ckpt_path, model, optimizer)
    print(f"Loaded checkpoint: epoch={epoch}, val_f1={val_f1:.4f}")

    return model, test_loader


def _save_attributions_npz(
    npz_path: Path,
    attributions: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save raw attributions to a ``.npz`` file, warning (not raising) on I/O failure."""
    try:
        np.savez(
            npz_path,
            attributions=attributions,
            labels=labels,
            predictions=predictions,
            feature_names=np.array(feature_names),
        )
        print(f"Saved raw attributions        → {npz_path}")
    except OSError as exc:
        print(f"Warning: failed to save attributions to {npz_path}: {exc}")


def _print_top_features(attributions: np.ndarray, feature_names: list[str], top_n: int = 5) -> None:
    """Print the top-N features by mean absolute attribution."""
    mean_abs = np.abs(attributions).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    print(f"\nTop {top_n} features by mean |attribution|:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {feature_names[idx]}: {mean_abs[idx]:.4f}")


def run_explanation(
    config_path: str,
    data_dir: str,
    checkpoint_dir: str = "outputs/",
    output_dir: str = "outputs/plots/",
    k: int = 10,
    method: str = "ig",
) -> None:
    """Load checkpoint and run the full explanation pipeline for horizon *k*.

    Steps:

    1. Load config and select compute device.
    2. Build the test DataLoader (via :func:`~deeplob.dataset.get_dataloaders`).
    3. Load ``best_model_k{k}.pt`` from *checkpoint_dir*.
    4. Compute attributions with *method* (``"ig"`` or ``"shap"``).
    5. Save ``feature_importance_k{k}.png`` and ``heatmap_k{k}.png``.
    6. Save raw attributions to ``attributions_k{k}.npz``.
    7. Print top-5 features by mean absolute attribution.

    Args:
        config_path: Path to ``default.yaml``.
        data_dir: Path to FI-2010 ``.npy`` files.
        checkpoint_dir: Directory containing ``best_model_k{k}.pt``
            (default ``"outputs/"``).
        output_dir: Directory for saved plots and ``.npz`` file
            (default ``"outputs/plots/"``).
        k: Prediction horizon to explain (default 10).
        method: Attribution method — ``"ig"`` (Integrated Gradients) or
            ``"shap"`` (SHAP GradientExplainer).

    Raises:
        FileNotFoundError: If the checkpoint for horizon *k* is not found.
        ValueError: If *method* is not ``"ig"`` or ``"shap"``.
    """
    if method not in {"ig", "shap"}:
        raise ValueError(f"method must be 'ig' or 'shap', got '{method}'")

    config = load_config(config_path)
    device = get_device()

    model, test_loader = _load_model_and_test_loader(config, data_dir, checkpoint_dir, k, device)

    print(f"Computing attributions via {method.upper()} (k={k}) …")
    if method == "ig":
        result = batch_integrated_gradients(model, test_loader, device)
    else:
        result = shap_summary(model, test_loader, device)

    attributions: np.ndarray = result["attributions"]
    labels: np.ndarray = result["labels"]
    feature_names: list[str] = result["feature_names"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Plots ────────────────────────────────────────────────────────────────
    fi_path = str(out / f"feature_importance_k{k}.png")
    plot_feature_importance(
        attributions,
        feature_names,
        title=f"LOB Feature Importance — {method.upper()}, k={k}",
        save_path=fi_path,
    )
    print(f"Saved feature importance plot → {fi_path}")

    hm_path = str(out / f"heatmap_k{k}.png")
    plot_class_attribution_heatmap(attributions, labels, feature_names, save_path=hm_path)
    print(f"Saved attribution heatmap     → {hm_path}")

    # ── Raw attributions ─────────────────────────────────────────────────────
    npz_path = out / f"attributions_k{k}.npz"
    _save_attributions_npz(npz_path, attributions, labels, result["predictions"], feature_names)

    # ── Top-5 summary ─────────────────────────────────────────────────────────
    _print_top_features(attributions, feature_names, top_n=5)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Explain DeepLOB predictions.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default="data/raw/")
    parser.add_argument("--checkpoint_dir", default="outputs/")
    parser.add_argument("--output_dir", default="outputs/plots/")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", choices=["ig", "shap"], default="ig")
    args = parser.parse_args()
    run_explanation(
        args.config,
        args.data_dir,
        args.checkpoint_dir,
        args.output_dir,
        args.k,
        args.method,
    )
