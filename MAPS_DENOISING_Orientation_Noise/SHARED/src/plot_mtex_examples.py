#!/usr/bin/env python3
"""
Example maps for the paper: the network and the best MTEX filter on the same
test map.

For each test set, two rows:

    orientation (IPF-Z)  truth | noisy input | best MTEX filter | network
    error vs the truth   true grain boundaries | input | MTEX | network

The best MTEX filter of a set is the one compare_with_mtex.py found to have the
lowest ALL error on that set (read from SHARED/figures/mtex_comparison/per_set_tests.csv,
so run that first). The network map is one training run (--run). Errors share
one log colour scale. Writes SHARED/figures/paper/mtex_examples.pdf and .png.

Usage
-----
    python plot_mtex_examples.py
    python plot_mtex_examples.py --sets scatter5_mis6 scatter15_mis9 --map 1
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from orientation import boundary_mask, disorientation_deg, euler_to_quat, load_euler
from plot_maps import ipf_z_color
from paths import MODEL_RESULTS, MTEX_RESULTS, clean_map, noise_dir

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ERR_RAMP = LinearSegmentedColormap.from_list(
    "err", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
NAMES = {"clean-mean": "mean", "clean-median": "median", "clean-kuwahara": "Kuwahara",
         "clean-spline": "spline", "clean-halfquad": "half-quadratic",
         "clean-infconv": "infimal conv."}


def label(s):
    m = re.match(r"scatter(\d+)(?:_mis(\d+))?$", s)
    return rf"$\theta_{{\max}}$ = {m.group(1)}°, $p$ = {m.group(2) or 0}%"


def best_filters():
    with open(ROOT / "SHARED/figures/mtex_comparison/per_set_tests.csv") as f:
        return {r["set"]: r["best_mtex"] for r in csv.DictReader(f) if r["measure"] == "all"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=["scatter5_mis6", "scatter15_mis9"])
    p.add_argument("--map", type=int, default=1)
    p.add_argument("--run", default="blind_learned_seed1")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 7.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": MUTED})
    best = best_filters()
    stem = f"map_{args.map:05d}_clean_euler_noisy"
    clean = euler_to_quat(load_euler(clean_map(args.map)))
    bnd = boundary_mask(clean, SHAPE).reshape(SHAPE)
    norm = LogNorm(vmin=0.1, vmax=60)

    n = len(args.sets)
    fig = plt.figure(figsize=(7.0, 3.75 * n))
    gs = fig.add_gridspec(2 * n, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.06, hspace=0.32,
                          left=0.05, right=0.88, top=0.97, bottom=0.01)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for k, s in enumerate(args.sets):
        f = best[s]
        noisy = euler_to_quat(load_euler(noise_dir(s) / f"{stem}.txt"))
        mtex = euler_to_quat(load_euler(MTEX_RESULTS / s / f"{stem}_noisy__{f}.txt"))
        ours = euler_to_quat(load_euler(MODEL_RESULTS / s / f"{stem}_noisy__{args.run}.txt"))
        errs = {name: disorientation_deg(q, clean) for name, q in
                (("input", noisy), ("mtex", mtex), ("ours", ours))}

        def err_title(name, what):
            d = errs[name]
            return f"error: {what}\nmean {d.mean():.2f}°, >10°: {100 * (d > 10).mean():.1f}%"

        rows = [
            [(ipf_z_color(clean).reshape(*SHAPE, 3), "truth"),
             (ipf_z_color(noisy).reshape(*SHAPE, 3), "noisy input"),
             (ipf_z_color(mtex).reshape(*SHAPE, 3), f"MTEX: {NAMES[f]} (best)"),
             (ipf_z_color(ours).reshape(*SHAPE, 3), "network")],
            [(np.where(bnd, 0.15, 1.0)[..., None].repeat(3, axis=2), "true grain boundaries"),
             (errs["input"].reshape(SHAPE), err_title("input", "noisy input")),
             (errs["mtex"].reshape(SHAPE), err_title("mtex", f"MTEX {NAMES[f]}")),
             (errs["ours"].reshape(SHAPE), err_title("ours", "network"))],
        ]
        for r, row in enumerate(rows):
            for c, (img, title) in enumerate(row):
                ax = fig.add_subplot(gs[2 * k + r, c])
                if img.ndim == 2:
                    im = ax.imshow(np.clip(img, 0.1, None), cmap=ERR_RAMP, norm=norm,
                                   interpolation="nearest")
                else:
                    ax.imshow(img, interpolation="nearest")
                ax.set_title(f"({next(letters)}) {title}")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c == 0 and r == 0:
                    ax.set_ylabel(label(s), fontsize=8, color=INK)
        print(f"{s}: best MTEX {f}; mean error input {errs['input'].mean():.3f}, "
              f"MTEX {errs['mtex'].mean():.3f}, network {errs['ours'].mean():.3f} deg")

    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.1, 0.3, 1, 3, 10, 30, 60])
    cb.set_ticklabels(["≤0.1", "0.3", "1", "3", "10", "30", "60"])
    cb.set_label("disorientation to the true map (deg)", color=INK2)
    out = ROOT / "SHARED/figures/paper/mtex_examples"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out}.pdf (+ .png)")


if __name__ == "__main__":
    main()
