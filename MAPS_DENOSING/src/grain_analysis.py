"""Grain-level evaluation: did the denoised map reproduce the microstructure?

PSNR and SSIM are generic image metrics - they do not know what a grain is.
This module asks the question a materials scientist would ask instead:

  * how many grains are recovered, against how many are really there?
  * how far (in pixels) are the recovered grain boundaries from the true ones?
  * do the grain interiors come out flat, the way real grains are?
  * does the grain size distribution survive?

Method
------
Grains are found the way the noise-processing literature does it: a *local
standard deviation* map. Inside a grain the colour is constant, so the local
std is ~0; across a boundary it jumps. Threshold that map, label the connected
interiors, and you have the grains - no ground truth needed, so the exact same
procedure runs on the clean, noisy and denoised maps and the comparison is fair.

Measured on this dataset, the separation is wide:

    clean     median local std  0.00   (interiors perfectly flat)
    denoised  median local std  0.51   (essentially as flat)
    noisy     median local std 23.61   (no flat interiors at all)

Run it
------
    python grain_analysis.py                    # 6 per-map figures + 200-map summary
    python grain_analysis.py --num 10 --summary-num 500
    python grain_analysis.py --maps Map_00006.png

Everything lands in visualize/grain_analysis/.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi

from architecture import UNetDenoiser
from dataset import EBSD_Dataset
from paths import (CLEAN_DIR, NOISY_DIR, BEST_CHECKPOINT, GRAIN_ANALYSIS_DIR,
                   ensure_dirs)

DPI = 150

# Local std below this counts as "grain interior". Calibrated on the clean maps,
# where the true grain count is known exactly: at 2.0 the segmentation recovers
# 49.2 grains per map against 49.2 true, i.e. the method itself contributes no
# error, so any discrepancy on the denoised maps is the model's. Raising it
# starts merging neighbouring grains (-0.6 grains/map by 4.0, -11 by 15.0).
STD_THRESHOLD = 2.0

# Connected interiors smaller than this are noise speckle, not grains.
MIN_GRAIN_SIZE = 10

# A recovered boundary within this many pixels of a true one counts as matched.
MATCH_TOLERANCE = 1.0


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def local_std(img, window=3):
    """Per-pixel local standard deviation, averaged over the RGB channels.

    img is HxWx3 in 0-255. Uses the E[x^2] - E[x]^2 identity so the whole map
    comes from two box filters.
    """
    x = img.astype(np.float64)
    mean = ndi.uniform_filter(x, size=(window, window, 1))
    mean_sq = ndi.uniform_filter(x * x, size=(window, window, 1))
    return np.sqrt(np.clip(mean_sq - mean * mean, 0, None)).mean(axis=2)


def segment_grains(img, threshold=STD_THRESHOLD, min_size=MIN_GRAIN_SIZE,
                   window=3):
    """Label grains via the local-std map. Returns (labels, n_grains).

    Boundary pixels are handed to their nearest grain at the end so the labels
    tile the whole image - otherwise grain areas would be systematically
    undercounted by the width of the boundary band.
    """
    interior = local_std(img, window) <= threshold
    labels, n = ndi.label(interior)

    if n:
        # Drop speckle, then renumber so labels stay 1..n with no gaps.
        sizes = np.bincount(labels.ravel())
        too_small = np.flatnonzero(sizes < min_size)
        if too_small.size:
            labels[np.isin(labels, too_small)] = 0
        remaining = np.unique(labels)
        remaining = remaining[remaining > 0]
        remap = np.zeros(labels.max() + 1, dtype=np.int32)
        remap[remaining] = np.arange(1, remaining.size + 1)
        labels = remap[labels]
        n = remaining.size

    if n and (labels == 0).any():
        _, (ri, rj) = ndi.distance_transform_edt(labels == 0, return_indices=True)
        labels = labels[ri, rj]

    return labels, n


def exact_grain_labels(clean_img):
    """Ground-truth grains: connected components of exactly-equal colour.

    Only valid on the clean maps, where every grain really is one flat colour.
    This is the reference the segmentation is judged against.
    """
    h, w, _ = clean_img.shape
    _, color_id = np.unique(clean_img.reshape(-1, 3), axis=0, return_inverse=True)
    color_id = color_id.reshape(h, w)

    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    for cid in np.unique(color_id):
        mask = color_id == cid
        comp, n = ndi.label(mask)
        labels[mask] = comp[mask] + next_label
        next_label += n
    return labels, next_label


def boundaries_from_labels(labels):
    """1-pixel boundary mask: True where a 4-neighbour has a different label."""
    b = np.zeros(labels.shape, dtype=bool)
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:, :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return b


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def boundary_metrics(true_bnd, pred_bnd, tolerance=MATCH_TOLERANCE):
    """Compare two boundary masks.

    Precision  - of the boundaries we found, how many are real?
    Recall     - of the real boundaries, how many did we find?
    Displacement - for each predicted boundary pixel, distance to the nearest
                   true one (in pixels).

    A plain pixel-for-pixel IoU is useless for thin structures (a one-pixel
    shift scores zero), so matching is done within `tolerance` pixels, which is
    the standard boundary-F1 construction.
    """
    out = {
        "n_true": int(true_bnd.sum()),
        "n_pred": int(pred_bnd.sum()),
    }
    if not true_bnd.any() or not pred_bnd.any():
        return {**out, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "iou_tol": 0.0, "disp_mean": float("nan"),
                "disp_median": float("nan"), "disp_p95": float("nan"),
                "within_1px": 0.0, "within_2px": 0.0, "distances": np.array([])}

    # Distance from every pixel to the nearest true / predicted boundary.
    dist_to_true = ndi.distance_transform_edt(~true_bnd)
    dist_to_pred = ndi.distance_transform_edt(~pred_bnd)

    d_pred = dist_to_true[pred_bnd]   # how far each found boundary is from truth
    d_true = dist_to_pred[true_bnd]   # how far each true boundary is from a find

    precision = float((d_pred <= tolerance).mean())
    recall = float((d_true <= tolerance).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    # IoU after dilating both masks by the tolerance, for a set-overlap number.
    grow = ndi.binary_dilation(true_bnd, iterations=int(round(tolerance)))
    grow_p = ndi.binary_dilation(pred_bnd, iterations=int(round(tolerance)))
    union = (grow | grow_p).sum()
    iou = float((grow & grow_p).sum() / union) if union else 0.0

    return {
        **out,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_tol": iou,
        "disp_mean": float(d_pred.mean()),
        "disp_median": float(np.median(d_pred)),
        "disp_p95": float(np.percentile(d_pred, 95)),
        "within_1px": float((d_pred <= 1.0).mean()),
        "within_2px": float((d_pred <= 2.0).mean()),
        "distances": d_pred,
    }


def intra_grain_std(img, labels):
    """Mean colour spread inside each true grain - 0 means perfectly flat.

    This is the cleanest single number for "are the grain interiors right":
    the ground truth scores exactly 0 by construction.
    """
    x = img.astype(np.float64)
    idx = labels.ravel()
    n = idx.max() + 1
    spreads = []
    for ch in range(3):
        v = x[:, :, ch].ravel()
        count = np.bincount(idx, minlength=n).astype(np.float64)
        s = np.bincount(idx, weights=v, minlength=n)
        s2 = np.bincount(idx, weights=v * v, minlength=n)
        keep = count > 0
        var = s2[keep] / count[keep] - (s[keep] / count[keep]) ** 2
        spreads.append(np.sqrt(np.clip(var, 0, None)))
    return float(np.mean(spreads))


def grain_sizes(labels):
    counts = np.bincount(labels.ravel())
    return counts[counts > 0]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_std_panels(name, clean, noisy, denoised, out_path,
                      threshold=STD_THRESHOLD):
    """3x3 grid: for clean / noisy / denoised, show image, local-std map, histogram.

    The middle column is the point of the figure - the denoised std map should
    look like the clean one (thin bright boundaries, black interiors) and
    nothing like the noisy one (bright everywhere).
    """
    rows = [("Clean (ground truth)", clean), ("Noisy (input)", noisy),
            ("Denoised (model output)", denoised)]

    # One shared colour scale, or the three std maps cannot be compared by eye.
    vmax = max(local_std(img).max() for _, img in rows)

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
    tags = iter("abcdefghi")

    for r, (label, img) in enumerate(rows):
        s = local_std(img)

        axes[r][0].imshow(img, interpolation="nearest")
        axes[r][0].set_title(f"({next(tags)}) {label}", fontsize=10)

        im = axes[r][1].imshow(s, cmap="viridis", vmin=0, vmax=vmax,
                               interpolation="nearest")
        axes[r][1].set_title(f"({next(tags)}) local standard deviation", fontsize=10)
        fig.colorbar(im, ax=axes[r][1], fraction=0.046, pad=0.04)

        ax = axes[r][2]
        ax.hist(s.ravel(), bins=60, color="#1f77b4")
        ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"interior/boundary\nthreshold = {threshold:g}")
        ax.set_yscale("log")
        ax.set_xlim(0, vmax)
        ax.set_xlabel("local std (grey levels)")
        ax.set_ylabel("pixel count (log)")
        ax.set_title(f"({next(tags)}) distribution", fontsize=10)
        ax.legend(fontsize=7)
        # Median says it all: 0 for clean, ~0.5 denoised, ~24 noisy.
        ax.text(0.97, 0.80, f"median {np.median(s):.2f}", ha="right",
                transform=ax.transAxes, fontsize=8, color="#444")

        for c in (0, 1):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

    fig.suptitle(f"Grain interiors and boundaries via local standard deviation - {name}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_boundary_overlay(name, clean, denoised, true_bnd, pred_bnd, stats,
                            out_path, tolerance=MATCH_TOLERANCE):
    """Where the recovered boundaries agree with the true ones, and where they don't."""
    dist_to_true = ndi.distance_transform_edt(~true_bnd)
    dist_to_pred = ndi.distance_transform_edt(~pred_bnd)

    matched = pred_bnd & (dist_to_true <= tolerance)   # found, and real
    spurious = pred_bnd & (dist_to_true > tolerance)   # found, but not real
    missed = true_bnd & (dist_to_pred > tolerance)     # real, but not found

    agreement = np.ones((*true_bnd.shape, 3))
    agreement[matched] = (0.13, 0.65, 0.27)   # green
    agreement[spurious] = (0.84, 0.15, 0.16)  # red
    agreement[missed] = (0.12, 0.35, 0.85)    # blue

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))

    axes[0].imshow(clean, interpolation="nearest")
    axes[0].contour(true_bnd, levels=[0.5], colors="black", linewidths=0.5)
    axes[0].set_title("(a) Clean + true boundaries", fontsize=10)

    axes[1].imshow(denoised, interpolation="nearest")
    axes[1].contour(pred_bnd, levels=[0.5], colors="black", linewidths=0.5)
    axes[1].set_title("(b) Denoised + recovered boundaries", fontsize=10)

    axes[2].imshow(agreement, interpolation="nearest")
    axes[2].set_title(
        f"(c) green matched ({stats['precision']*100:.1f}%)\n"
        f"red spurious   blue missed", fontsize=10)

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[3]
    d = stats["distances"]
    if d.size:
        ax.hist(d, bins=np.arange(0, max(6.0, d.max() + 1), 0.5),
                color="#2ca02c", edgecolor="white")
    ax.axvline(tolerance, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"{tolerance:g} px tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("distance to nearest true boundary (px)")
    ax.set_ylabel("boundary pixels (log)")
    ax.set_title(f"(d) displacement\nmedian {stats['disp_median']:.2f} px, "
                 f"{stats['within_1px']*100:.1f}% within 1 px", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Boundary localisation - {name}   |   "
        f"grains {stats['n_grains_pred']} vs {stats['n_grains_true']} true   |   "
        f"F1 {stats['f1']:.3f}   IoU {stats['iou_tol']:.3f}",
        fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_summary(records, out_path, tolerance=MATCH_TOLERANCE):
    """Aggregate over every analysed map: counts, displacement, sizes, flatness."""
    n_true = np.array([r["n_grains_true"] for r in records])
    n_pred = np.array([r["n_grains_pred"] for r in records])
    all_d = np.concatenate([r["distances"] for r in records if len(r["distances"])])
    sizes_true = np.concatenate([r["sizes_true"] for r in records])
    sizes_pred = np.concatenate([r["sizes_pred"] for r in records])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    # (a) grain count agreement
    ax = axes[0][0]
    lim = [min(n_true.min(), n_pred.min()) - 2, max(n_true.max(), n_pred.max()) + 2]
    ax.plot(lim, lim, color="#d62728", linestyle="--", linewidth=1.2,
            label="perfect agreement")
    ax.scatter(n_true, n_pred, s=18, alpha=0.55, color="#1f77b4",
               edgecolor="none")
    exact = float((n_true == n_pred).mean())
    within1 = float((np.abs(n_true - n_pred) <= 1).mean())
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true grain count")
    ax.set_ylabel("recovered grain count")
    ax.set_title(f"(a) Grain count\nexact {exact*100:.1f}%, within 1: {within1*100:.1f}%",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) pooled boundary displacement
    ax = axes[0][1]
    ax.hist(all_d, bins=np.arange(0, max(6.0, np.percentile(all_d, 99.9) + 1), 0.5),
            color="#2ca02c", edgecolor="white")
    ax.axvline(tolerance, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"{tolerance:g} px tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("distance to nearest true boundary (px)")
    ax.set_ylabel("boundary pixels (log)")
    ax.set_title(f"(b) Boundary displacement\nmedian {np.median(all_d):.2f} px, "
                 f"{(all_d <= 1).mean()*100:.1f}% within 1 px", fontsize=10)
    ax.legend(fontsize=8)

    # (c) grain size distribution
    ax = axes[1][0]
    bins = np.logspace(np.log10(max(1, min(sizes_true.min(), sizes_pred.min()))),
                       np.log10(max(sizes_true.max(), sizes_pred.max())), 40)
    ax.hist(sizes_true, bins=bins, alpha=0.55, label="true", color="#1f77b4")
    ax.hist(sizes_pred, bins=bins, alpha=0.55, label="recovered", color="#ff7f0e")
    ax.set_xscale("log")
    ax.set_xlabel("grain area (pixels)")
    ax.set_ylabel("grain count")
    ax.set_title("(c) Grain size distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) interior flatness - the ground truth is exactly 0
    ax = axes[1][1]
    flat_den = np.array([r["intra_std_denoised"] for r in records])
    flat_noisy = np.array([r["intra_std_noisy"] for r in records])
    ax.hist(flat_noisy, bins=30, alpha=0.6, label=f"noisy (mean {flat_noisy.mean():.1f})",
            color="#d62728")
    ax.hist(flat_den, bins=30, alpha=0.75, label=f"denoised (mean {flat_den.mean():.2f})",
            color="#2ca02c")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="clean = 0")
    ax.set_xlabel("mean colour spread inside true grains (grey levels)")
    ax.set_ylabel("maps")
    ax.set_title("(d) Grain interior flatness", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Grain-level accuracy over {len(records)} validation maps", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Grain-level evaluation of the denoiser")
    p.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    p.add_argument("--clean-dir", default=CLEAN_DIR)
    p.add_argument("--noisy-dir", default=NOISY_DIR)
    p.add_argument("--out-dir", default=GRAIN_ANALYSIS_DIR)

    p.add_argument("--maps", nargs="+", default=None,
                   help="specific filenames instead of the first --num validation maps")
    p.add_argument("--num", type=int, default=6,
                   help="maps to draw per-map figures for")
    p.add_argument("--summary-num", type=int, default=200,
                   help="maps to pool for the summary figure and the stats table")

    p.add_argument("--threshold", type=float, default=STD_THRESHOLD)
    p.add_argument("--min-grain-size", type=int, default=MIN_GRAIN_SIZE)
    p.add_argument("--tolerance", type=float, default=MATCH_TOLERANCE)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    return p.parse_args()


def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = UNetDenoiser(base=ckpt.get("args", {}).get("base_channels", 64))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"Loaded {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def analyse_one(clean, noisy, denoised, args):
    """All the grain metrics for one map. Arrays are HxWx3 uint8-valued floats."""
    true_labels, n_true = exact_grain_labels(clean.astype(np.uint8))
    true_bnd = boundaries_from_labels(true_labels)

    pred_labels, n_pred = segment_grains(denoised, args.threshold, args.min_grain_size)
    pred_bnd = boundaries_from_labels(pred_labels)

    # Control: the same segmentation run on the clean map. Any gap between this
    # and n_true is the method's own error, not the model's - it lets you say
    # how much of the discrepancy below actually belongs to the denoiser.
    _, n_clean_seg = segment_grains(clean, args.threshold, args.min_grain_size)

    stats = boundary_metrics(true_bnd, pred_bnd, args.tolerance)
    stats.update({
        "n_grains_true": int(n_true),
        "n_grains_pred": int(n_pred),
        "n_grains_clean_seg": int(n_clean_seg),
        "sizes_true": grain_sizes(true_labels),
        "sizes_pred": grain_sizes(pred_labels),
        "intra_std_denoised": intra_grain_std(denoised, true_labels),
        "intra_std_noisy": intra_grain_std(noisy, true_labels),
        "intra_std_clean": intra_grain_std(clean, true_labels),
    })
    return stats, true_bnd, pred_bnd


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    ensure_dirs(args.out_dir)

    val_set = EBSD_Dataset(args.clean_dir, args.noisy_dir, split="val",
                           val_fraction=args.val_fraction, seed=args.seed)
    if args.maps:
        names = args.maps
        index = {n: i for i, n in enumerate(val_set.image_filenames)}
        missing = [n for n in names if n not in index]
        if missing:
            raise SystemExit(f"not in the validation split: {missing}")
        indices = [index[n] for n in names]
    else:
        indices = list(range(min(args.summary_num, len(val_set))))
        names = [val_set.image_filenames[i] for i in indices]

    n_figures = len(names) if args.maps else min(args.num, len(indices))
    print(f"Analysing {len(indices)} map(s); per-map figures for the first {n_figures}")

    records, written = [], []

    with torch.no_grad():
        for k, idx in enumerate(indices):
            noisy_t, clean_t = val_set[idx]
            den_t = model(noisy_t.unsqueeze(0).to(device)).float().clamp(0, 1)[0].cpu()

            clean = clean_t.permute(1, 2, 0).numpy() * 255.0
            noisy = noisy_t.permute(1, 2, 0).numpy() * 255.0
            denoised = den_t.permute(1, 2, 0).numpy() * 255.0

            stats, true_bnd, pred_bnd = analyse_one(clean, noisy, denoised, args)
            records.append(stats)

            if k < n_figures:
                stem = os.path.splitext(names[k])[0]
                written.append(figure_std_panels(
                    names[k], clean.astype(np.uint8), noisy.astype(np.uint8),
                    denoised.astype(np.uint8),
                    os.path.join(args.out_dir, f"{stem}_local_std.png"),
                    args.threshold))
                written.append(figure_boundary_overlay(
                    names[k], clean.astype(np.uint8), denoised.astype(np.uint8),
                    true_bnd, pred_bnd, stats,
                    os.path.join(args.out_dir, f"{stem}_boundaries.png"),
                    args.tolerance))

            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(indices)}", flush=True)

    summary_path = os.path.join(args.out_dir, "grain_summary.png")
    figure_summary(records, summary_path, args.tolerance)
    written.append(summary_path)

    # --- numbers ----------------------------------------------------------
    n_true = np.array([r["n_grains_true"] for r in records])
    n_pred = np.array([r["n_grains_pred"] for r in records])
    all_d = np.concatenate([r["distances"] for r in records if len(r["distances"])])

    summary = {
        "maps": len(records),
        "grain_count_true_mean": float(n_true.mean()),
        "grain_count_pred_mean": float(n_pred.mean()),
        "grain_count_clean_seg_mean": float(
            np.mean([r["n_grains_clean_seg"] for r in records])),
        "grain_count_exact_match": float((n_true == n_pred).mean()),
        "grain_count_within_1": float((np.abs(n_true - n_pred) <= 1).mean()),
        "grain_count_mae": float(np.abs(n_true - n_pred).mean()),
        "boundary_precision": float(np.mean([r["precision"] for r in records])),
        "boundary_recall": float(np.mean([r["recall"] for r in records])),
        "boundary_f1": float(np.mean([r["f1"] for r in records])),
        "boundary_iou": float(np.mean([r["iou_tol"] for r in records])),
        "disp_median_px": float(np.median(all_d)),
        "disp_mean_px": float(all_d.mean()),
        "disp_p95_px": float(np.percentile(all_d, 95)),
        "within_1px": float((all_d <= 1.0).mean()),
        "within_2px": float((all_d <= 2.0).mean()),
        "intra_grain_std_clean": float(np.mean([r["intra_std_clean"] for r in records])),
        "intra_grain_std_denoised": float(np.mean([r["intra_std_denoised"] for r in records])),
        "intra_grain_std_noisy": float(np.mean([r["intra_std_noisy"] for r in records])),
        "tolerance_px": args.tolerance,
        "std_threshold": args.threshold,
    }

    json_path = os.path.join(args.out_dir, "grain_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    written.append(json_path)

    print(f"\nGrain-level results over {summary['maps']} validation maps")
    print(f"  grains          {summary['grain_count_pred_mean']:.2f} recovered vs "
          f"{summary['grain_count_true_mean']:.2f} true "
          f"(exact on {summary['grain_count_exact_match']*100:.1f}% of maps, "
          f"within 1 on {summary['grain_count_within_1']*100:.1f}%)")
    print(f"                  control: same segmentation on the CLEAN map gives "
          f"{summary['grain_count_clean_seg_mean']:.2f}, so the method itself "
          f"costs {summary['grain_count_true_mean']-summary['grain_count_clean_seg_mean']:+.2f}")
    print(f"  boundary F1     {summary['boundary_f1']:.4f}  "
          f"(precision {summary['boundary_precision']:.4f}, "
          f"recall {summary['boundary_recall']:.4f})")
    print(f"  boundary IoU    {summary['boundary_iou']:.4f}  "
          f"at {args.tolerance:g} px tolerance")
    print(f"  displacement    median {summary['disp_median_px']:.2f} px, "
          f"{summary['within_1px']*100:.1f}% within 1 px, "
          f"{summary['within_2px']*100:.1f}% within 2 px")
    print(f"  interior std    clean {summary['intra_grain_std_clean']:.2f} | "
          f"denoised {summary['intra_grain_std_denoised']:.2f} | "
          f"noisy {summary['intra_grain_std_noisy']:.2f}  (grey levels)")
    print(f"\nWrote {len(written)} file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
