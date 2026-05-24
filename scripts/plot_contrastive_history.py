"""Plot Stage 1/Stage 2 Contrastive Astromer training histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_stage1_history(history: dict, output_dir: Path, output_name: str = "training_curves.png") -> Path | None:
    """Plot Stage 1 training & validation loss/accuracy curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not installed -- skipping plot.")
        return None

    epochs = history.get("epoch") or list(range(1, len(history["train_loss"]) + 1))
    has_val = len(history["val_loss"]) > 0
    title_suffix = f"Epoch 1-{epochs[-1]}" if epochs else "Training"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    if has_val:
        ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#d62728", linewidth=2)
        best_val_idx = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
        ax1.scatter([epochs[best_val_idx]], [history["val_loss"][best_val_idx]], color="#d62728", s=80, zorder=5)
        ax1.annotate(
            f"best val={history['val_loss'][best_val_idx]:.4f}\nepoch {epochs[best_val_idx]}",
            xy=(epochs[best_val_idx], history["val_loss"][best_val_idx]),
            xytext=(-82, 18),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8},
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("InfoNCE Loss")
    ax1.set_title(f"Stage 1 Contrastive Loss ({title_suffix})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#1f77b4", linewidth=2)
    if has_val:
        ax2.plot(epochs, history["val_acc"], "s-", label="Val Acc", color="#2ca02c", linewidth=2)
        best_acc_idx = max(range(len(history["val_acc"])), key=lambda i: history["val_acc"][i])
        ax2.scatter([epochs[best_acc_idx]], [history["val_acc"][best_acc_idx]], color="#2ca02c", s=80, zorder=5)
        ax2.annotate(
            f"best val acc={history['val_acc'][best_acc_idx]:.4f}\nepoch {epochs[best_acc_idx]}",
            xy=(epochs[best_acc_idx], history["val_acc"][best_acc_idx]),
            xytext=(-100, -35),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8},
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Batch-wise Positive Pair Matching Accuracy")
    ax2.set_title(f"Stage 1 Contrastive Accuracy ({title_suffix})")
    ax2.set_ylim(0.965, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / output_name
    fig.savefig(str(plot_path), dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved to {plot_path}")
    return plot_path


def _load_stage2_dependencies():
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def _as_array(history: dict, key: str):
    _, np = _load_stage2_dependencies()
    value = history.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return arr


def _epoch_axis(history: dict):
    _, np = _load_stage2_dependencies()
    candidate_lengths = []
    for key in [
        "val_loss",
        "val_supcon",
        "val_ce",
        "train_retr",
        "val_retr",
    ]:
        arr = _as_array(history, key)
        if arr is not None:
            candidate_lengths.append(arr.size)

    if not candidate_lengths:
        raise ValueError("No plottable series found in history JSON.")

    n_epochs = max(candidate_lengths)
    return np.arange(1, n_epochs + 1)


def _add_ce_warmup_annotation(
    ax,
    x: float = 0.03,
    y: float = 0.95,
    va: str = "top",
) -> None:
    """Mark epoch 1 as CE warm-up on the current axis."""
    ax.axvspan(0.5, 1.5, color="#e9ecef", alpha=0.35, zorder=0)
    ax.axvline(1.0, color="#6c757d", linestyle=":", linewidth=1.2, alpha=0.95)
    ax.text(
        x,
        y,
        "Epoch 1 = CE Warm-Up",
        transform=ax.transAxes,
        fontsize=9,
        va=va,
        ha="left",
        color="#343a40",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8},
    )


def _mask_epoch1_supcon_if_warmup(history: dict, supcon_arr):
    """Hide only epoch-1 SupCon when epoch-1 SupCon weight is zero."""
    _, np = _load_stage2_dependencies()
    weights = _as_array(history, "epoch_supcon_weight")
    masked = supcon_arr.copy()
    if weights is not None and weights.size > 0 and np.isclose(weights[0], 0.0):
        masked[0] = np.nan
    return masked


def _plot_val_loss_axis(ax, history: dict) -> None:
    epochs = _epoch_axis(history)
    val_loss = _as_array(history, "val_loss")
    val_supcon = _as_array(history, "val_supcon")
    val_ce = _as_array(history, "val_ce")

    if val_loss is not None:
        ax.plot(
            epochs[: val_loss.size],
            val_loss,
            marker="o",
            linewidth=2.4,
            label="Val Total Loss",
            color="#1f77b4",
        )

    if val_supcon is not None:
        val_supcon_plot = _mask_epoch1_supcon_if_warmup(history, val_supcon)
        ax.plot(
            epochs[: val_supcon_plot.size],
            val_supcon_plot,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label="Val SupCon Loss",
            color="#ff7f0e",
        )

    if val_ce is not None:
        ax.plot(
            epochs[: val_ce.size],
            val_ce,
            marker="^",
            linewidth=2.0,
            linestyle="-.",
            label="Val CE Loss",
            color="#2ca02c",
        )

    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    _add_ce_warmup_annotation(ax, x=0.03, y=0.50, va="center")
    ax.legend(loc="best", frameon=True)


def _plot_retrieval_axis(ax, history: dict) -> None:
    _, np = _load_stage2_dependencies()
    epochs = _epoch_axis(history)
    train_retr = _as_array(history, "train_retr")
    val_retr = _as_array(history, "val_retr")

    if train_retr is None and val_retr is None:
        raise ValueError("Neither train_retr nor val_retr found in history JSON.")

    if train_retr is not None:
        ax.plot(
            epochs[: train_retr.size],
            train_retr,
            marker="o",
            linewidth=2.2,
            label="Train Retrieval@1",
            color="#17becf",
        )

    if val_retr is not None:
        ax.plot(
            epochs[: val_retr.size],
            val_retr,
            marker="s",
            linewidth=2.2,
            label="Val Retrieval@1",
            color="#d62728",
        )

        best_idx = int(np.argmax(val_retr))
        best_epoch = best_idx + 1
        best_val = float(val_retr[best_idx])
        ax.scatter([best_epoch], [best_val], color="#d62728", s=60, zorder=3)
        ax.annotate(
            f"best val={best_val:.3f} @ epoch {best_epoch}",
            xy=(best_epoch, best_val),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75},
        )

    ax.set_title("Retrieval@1 (Top-1 Same-Class Match Rate)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Same-Class Match Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    _add_ce_warmup_annotation(ax, x=0.03, y=0.50, va="center")
    ax.legend(loc="best", frameon=True)


def plot_stage2_side_by_side(history: dict, output_dir: Path, output_name: str) -> Path:
    plt, _ = _load_stage2_dependencies()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    _plot_val_loss_axis(axes[0], history)
    _plot_retrieval_axis(axes[1], history)

    fig.suptitle("Stage 2 Supervised Contrastive Learning with CE", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_stage2_history(
    history: dict,
    output_dir: Path,
    output_name: str = "stage2_val_loss_retrieval.png",
) -> Path:
    combined_path = plot_stage2_side_by_side(
        history=history,
        output_dir=output_dir,
        output_name=output_name,
    )
    print(f"[OK] Saved combined figure: {combined_path}")
    return combined_path


def _load_history(history_json: Path) -> dict:
    if not history_json.exists():
        raise FileNotFoundError(f"History JSON not found: {history_json}")
    with open(history_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot contrastive training curves from history JSON.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--history_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    args = parser.parse_args()

    history_path = Path(args.history_json)
    output_dir = Path(args.output_dir) if args.output_dir else history_path.parent
    history = _load_history(history_path)

    if args.stage == "stage1":
        plot_stage1_history(
            history=history,
            output_dir=output_dir,
            output_name=args.output_name or "training_curves.png",
        )
    else:
        plot_stage2_history(
            history=history,
            output_dir=output_dir,
            output_name=args.output_name or "stage2_val_loss_retrieval.png",
        )


if __name__ == "__main__":
    main()
