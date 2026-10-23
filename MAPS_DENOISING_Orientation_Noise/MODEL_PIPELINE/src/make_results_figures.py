#!/usr/bin/env python3
"""
Paper figures for the Results section, drawn from saved outputs only (no model
is run, so this works on a login node).

  SHARED/figures/paper/training_loss.pdf   training and validation loss of the runs
  SHARED/figures/paper/denoised_examples.pdf
                                    one test map at three noise conditions:
                                    noisy input, denoised map, and the error
                                    of each against the truth

The denoised maps are the predictions eval_model.sh saved in
datasets/results_test_set_model/<set>/<stem>__<run>.txt.

Usage
-----
    python make_results_figures.py --runs blind_learned_seed1 blind_learned_seed2 blind_learned_seed3
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Shared code (orientation maths, paths, scoring) lives in SHARED/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SHARED" / "src"))
from orientation import disorientation_deg, euler_to_quat, load_euler
from plot_maps import ipf_z_color
from paths import MODEL_RESULTS, clean_map, noise_dir

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
ERR_RAMP = LinearSegmentedColormap.from_list(
    "err", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
CONDITIONS = [("scatter5_mis6", r"$\theta_{\max}$ = 5°, $p$ = 6%"),
              ("scatter15", r"$\theta_{\max}$ = 15°, $p$ = 0%"),
              ("scatter15_mis9", r"$\theta_{\max}$ = 15°, $p$ = 9%")]


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": MUTED})


def training_loss(runs, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for k, run in enumerate(runs):
        h = json.loads((ROOT / f"MODEL_PIPELINE/checkpoints/runs/{run}/history.json").read_text())
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["train_loss"] for e in h], color=SERIES[k], lw=1.2, ls="--")
        ax.plot(ep, [e["val"]["loss"] for e in h], color=SERIES[k], lw=1.6,
                label=f"run {k + 1}")
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label="training")
    ax.plot([], [], color=MUTED, lw=1.6, label="validation")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def examples(run, map_id, out):
    stem = f"map_{map_id:05d}_clean_euler_noisy"
    clean = euler_to_quat(load_euler(clean_map(map_id)))
    norm = LogNorm(vmin=0.1, vmax=60)
    fig = plt.figure(figsize=(7.0, 5.9))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.06, hspace=0.3,
                          left=0.05, right=0.88, top=0.94, bottom=0.01)
    letters = iter("abcdefghijkl")
    for r, (name, label) in enumerate(CONDITIONS):
        noisy = euler_to_quat(load_euler(noise_dir(name) / f"{stem}.txt"))
        den = euler_to_quat(load_euler(MODEL_RESULTS / name / f"{stem}_noisy__{run}.txt"))
        d_in, d_out = disorientation_deg(noisy, clean), disorientation_deg(den, clean)
        panels = [(ipf_z_color(noisy).reshape(*SHAPE, 3), "noisy input"),
                  (ipf_z_color(den).reshape(*SHAPE, 3), "denoised"),
                  (d_in.reshape(SHAPE), f"error of input\nmean {d_in.mean():.2f}°"),
                  (d_out.reshape(SHAPE), f"error of denoised\nmean {d_out.mean():.2f}°")]
        for c, (img, title) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if img.ndim == 2:
                im = ax.imshow(np.clip(img, 0.1, None), cmap=ERR_RAMP, norm=norm,
                               interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_title(f"({next(letters)}) {title}")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, color=INK)
    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.1, 0.3, 1, 3, 10, 30, 60])
    cb.set_ticklabels(["≤0.1", "0.3", "1", "3", "10", "30", "60"])
    cb.set_label("disorientation to the true map (deg)", color=INK2)
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--map", type=int, default=1)
    args = p.parse_args()
    out = ROOT / "SHARED/figures/paper"
    out.mkdir(parents=True, exist_ok=True)
    style()
    training_loss(args.runs, out / "training_loss")
    examples(args.runs[0], args.map, out / "denoised_examples")
    print(f"wrote {out}/training_loss.pdf and {out}/denoised_examples.pdf (+ .png)")


if __name__ == "__main__":
    main()
