#!/usr/bin/env python3
"""
Stage 1 -- pattern-quality charts.

01_pattern_quality_holdout.py only ever saves the *mean* PSNR/NCC over the
whole holdout set (4 numbers total, in the JSON). This script reads its
saved pattern arrays (stage1_denoised_patterns.h5 / _harsh.h5) and recomputes
PSNR/NCC per pattern -- no TensorFlow needed, since the denoised patterns are
already on disk -- to show two things the JSON can't: the full distribution
across all 4,125 patterns, and calibrated vs. harsh side by side.

Run (any environment with h5py/numpy/matplotlib -- the main venv is fine,
no TensorFlow/pyebsdindex required):
    python3 03_pattern_quality_charts.py
"""

import os

import h5py
import matplotlib
matplotlib.use("Agg")          # no display under WSL; save figures instead
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito colorblind-safe pair, same semantic roles as Stage 2's CDF plot:
# noisy/baseline = orange, denoised/result = blue.
COLOR_NOISY    = "#E69F00"
COLOR_DENOISED = "#0072B2"

RUNS = [
    ("Calibrated (SNR 1.27)", f"{OUT_DIR}/stage1_denoised_patterns.h5"),
    ("Harsh (SNR ~0.50)",     f"{OUT_DIR}/stage1_denoised_patterns_harsh.h5"),
]


def psnr_per_pattern(a, b):
    mse = np.mean((a - b) ** 2, axis=(1, 2))
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))


def ncc_per_pattern(a, b):
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).mean(axis=1) / np.maximum(a.std(axis=1) * b.std(axis=1), 1e-12)


# ----------------------------------------------------------------------
# Load both runs and compute per-pattern metrics
# ----------------------------------------------------------------------
data = {}
for label, path in RUNS:
    with h5py.File(path, "r") as f:
        noisy = f["noisy"][:]
        denoised = f["denoised"][:]
        clean = f["clean"][:]
    data[label] = {
        "psnr_noisy":    psnr_per_pattern(noisy, clean),
        "psnr_denoised": psnr_per_pattern(denoised, clean),
        "ncc_noisy":     ncc_per_pattern(noisy, clean),
        "ncc_denoised":  ncc_per_pattern(denoised, clean),
    }
    print(f"{label}: {len(noisy)} patterns loaded from {os.path.basename(path)}")


# ----------------------------------------------------------------------
# Chart 1 -- summary bars: mean PSNR / NCC, noisy vs. denoised, calibrated
# vs. harsh side by side. Two separate axes rather than a dual-axis chart,
# since PSNR (dB) and NCC (unitless, 0-1) aren't the same scale.
# ----------------------------------------------------------------------
fig, (ax_psnr, ax_ncc) = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)

labels = [label for label, _ in RUNS]
x = np.arange(len(labels))
width = 0.32

for ax, key, ylabel, title, fmt in [
    (ax_psnr, "psnr", "PSNR (dB)", "Pattern fidelity -- PSNR", "%.2f"),
    (ax_ncc,  "ncc",  "NCC",       "Pattern fidelity -- NCC",  "%.3f"),
]:
    noisy_means    = [data[l][f"{key}_noisy"].mean() for l in labels]
    denoised_means = [data[l][f"{key}_denoised"].mean() for l in labels]
    b1 = ax.bar(x - width / 2, noisy_means, width, label="noisy input", color=COLOR_NOISY)
    b2 = ax.bar(x + width / 2, denoised_means, width, label="denoised", color=COLOR_DENOISED)
    ax.bar_label(b1, fmt=fmt, fontsize=9, padding=2)
    ax.bar_label(b2, fmt=fmt, fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Stage 1 -- pattern quality, noisy vs. denoised\n(mean over all 4,125 holdout patterns)",
             fontsize=12)
bar_path = f"{OUT_DIR}/stage1_quality_bars.png"
plt.savefig(bar_path, dpi=150)
plt.close(fig)
print(f"saved {os.path.basename(bar_path)}")


# ----------------------------------------------------------------------
# Chart 2 -- per-pattern distributions: does denoising shift every pattern
# by roughly the same amount, or help some far more than others? The mean
# alone (Chart 1) can't distinguish "uniform small gain" from "big gain on
# a few bad patterns, no change on the rest."
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)

for row, (label, _) in enumerate(RUNS):
    d = data[label]
    ax_p, ax_n = axes[row]

    for ax, nkey, dkey, xlabel in [
        (ax_p, "psnr_noisy", "psnr_denoised", "PSNR (dB)"),
        (ax_n, "ncc_noisy",  "ncc_denoised",  "NCC"),
    ]:
        lo = min(d[nkey].min(), d[dkey].min())
        hi = max(d[nkey].max(), d[dkey].max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(d[nkey], bins=bins, color=COLOR_NOISY, alpha=0.6, label="noisy input")
        ax.hist(d[dkey], bins=bins, color=COLOR_DENOISED, alpha=0.6, label="denoised")
        ax.axvline(d[nkey].mean(), color=COLOR_NOISY, linestyle="--", linewidth=1.5)
        ax.axvline(d[dkey].mean(), color=COLOR_DENOISED, linestyle="--", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_p.set_title(f"{label} -- PSNR per pattern", fontsize=10.5)
    ax_n.set_title(f"{label} -- NCC per pattern", fontsize=10.5)

fig.suptitle("Stage 1 -- per-pattern distribution, all 4,125 holdout patterns\n"
             "(dashed lines = mean; how cleanly orange and blue separate shows how "
             "consistently denoising helps, not just the average)",
             fontsize=11.5)
dist_path = f"{OUT_DIR}/stage1_quality_distributions.png"
plt.savefig(dist_path, dpi=150)
plt.close(fig)
print(f"saved {os.path.basename(dist_path)}")
