"""All plotting for the project. Everything written here lands in visualize/.

Two things get drawn:

  plot_curves()      training vs validation loss, plus PSNR and SSIM
                     -> visualize/curves/

  save_comparison()  one figure per map: clean | noisy | denoised
                     -> visualize/comparisons/

Can also be run directly to redraw the curves from a finished (or in-progress)
training run without retraining anything:

    python visualize.py
"""

import json
import os

import matplotlib

# Agg: no display on the compute nodes, so never try to open a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from paths import CURVES_DIR, COMPARISONS_DIR, HISTORY_JSON, ensure_dirs

DPI = 150


def _integer_epochs(ax):
    """Epochs are whole numbers - stop matplotlib labelling them 1.25, 1.50..."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_curves(history, out_dir=CURVES_DIR):
    """Draw loss / PSNR / SSIM curves from a list of per-epoch dicts.

    `history` is what train.py writes to checkpoints/history.json - one entry
    per epoch, each with train_loss, val_loss, val_psnr, val_ssim and the
    matching baselines.

    Returns the list of files written.
    """
    if not history:
        print("No history to plot yet.")
        return []

    ensure_dirs(out_dir)
    epochs = [h["epoch"] for h in history]
    written = []

    # --- 1. the loss curve (the one that shows over/under-fitting) ---------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history],
            label="Training loss", color="#1f77b4", linewidth=2)
    ax.plot(epochs, [h["val_loss"] for h in history],
            label="Validation loss", color="#d62728", linewidth=2)

    # Mark the best epoch - if validation loss starts climbing away from
    # training loss after this point, that is overfitting.
    best_i = int(np.argmin([h["val_loss"] for h in history]))
    ax.axvline(epochs[best_i], color="grey", linestyle=":", linewidth=1.2)
    # Flip the label to the left of the line once the best epoch is past the
    # midpoint, otherwise it runs off the right edge of the axes.
    past_middle = best_i > len(epochs) / 2
    ax.annotate(
        f"best val loss\nepoch {epochs[best_i]}: {history[best_i]['val_loss']:.4f}",
        xy=(epochs[best_i], history[best_i]["val_loss"]),
        xytext=(-8 if past_middle else 8, 22), textcoords="offset points",
        ha="right" if past_middle else "left",
        fontsize=8, color="grey",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss  (lower is better)")
    ax.set_title("Training vs validation loss")
    _integer_epochs(ax)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    written.append(path)

    # --- 2. PSNR and SSIM, each against the do-nothing baseline -----------
    for key, label, unit in [("psnr", "PSNR", " (dB)"), ("ssim", "SSIM", "")]:
        if f"val_{key}" not in history[0]:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [h[f"val_{key}"] for h in history],
                label=f"Denoised {label}", color="#2ca02c", linewidth=2)

        # A flat line for the noisy input's own score: anything above this
        # line is genuine improvement, anything below is making it worse.
        baseline = history[-1].get(f"baseline_{key}")
        if baseline is not None:
            ax.axhline(baseline, color="#ff7f0e", linestyle="--", linewidth=1.6,
                       label=f"Noisy input baseline ({baseline:.2f})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{label}{unit}  (higher is better)")
        ax.set_title(f"Validation {label}")
        _integer_epochs(ax)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = os.path.join(out_dir, f"{key}_curve.png")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        written.append(path)

    return written


def plot_curves_from_file(history_path=HISTORY_JSON, out_dir=CURVES_DIR):
    if not os.path.isfile(history_path):
        raise FileNotFoundError(
            f"no training history at {history_path} - run train.py first"
        )
    with open(history_path) as f:
        return plot_curves(json.load(f), out_dir)


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def _to_display(tensor):
    """CHW float tensor in [0,1] -> HWC uint8 array for imshow."""
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    return (np.transpose(array, (1, 2, 0)) * 255).round().astype(np.uint8)


def save_comparison(clean, noisy, denoised, out_path, title=None, scores=None):
    """Write one figure showing clean | noisy | denoised, side by side.

    clean / noisy / denoised are CHW float tensors in [0, 1].
    `scores` is an optional dict like
        {"noisy": (psnr, ssim), "denoised": (psnr, ssim)}
    which is printed under the relevant panels.
    """
    ensure_dirs(os.path.dirname(os.path.abspath(out_path)))

    panels = [
        ("Clean (ground truth)", clean, None),
        ("Noisy (input)", noisy, (scores or {}).get("noisy")),
        ("Denoised (model output)", denoised, (scores or {}).get("denoised")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (label, tensor, score) in zip(axes, panels):
        ax.imshow(_to_display(tensor), interpolation="nearest")
        if score is not None:
            label = f"{label}\nPSNR {score[0]:.2f} dB   SSIM {score[1]:.4f}"
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12)
    # Leave headroom for suptitle so it never collides with the panel titles.
    fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_grid(rows, out_path, title=None):
    """Stack several comparisons into one figure: one map per row.

    `rows` is a list of (name, clean, noisy, denoised, scores) tuples. Handy
    for eyeballing a handful of maps at once instead of opening many files.
    """
    ensure_dirs(os.path.dirname(os.path.abspath(out_path)))
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4.3 * n), squeeze=False)

    for r, (name, clean, noisy, denoised, scores) in enumerate(rows):
        panels = [
            ("Clean (ground truth)", clean, None),
            ("Noisy (input)", noisy, (scores or {}).get("noisy")),
            ("Denoised (model output)", denoised, (scores or {}).get("denoised")),
        ]
        for c, (label, tensor, score) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(_to_display(tensor), interpolation="nearest")
            if score is not None:
                label = f"{label}\nPSNR {score[0]:.2f} dB   SSIM {score[1]:.4f}"
            # Only label the top row's columns; every row gets its map name.
            ax.set_title(label if r == 0 else
                         (f"PSNR {score[0]:.2f} dB   SSIM {score[1]:.4f}"
                          if score is not None else ""), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for p in plot_curves_from_file():
        print(f"Wrote {p}")
