#!/usr/bin/env python3
"""
Look at one test map: truth, noisy input, denoised map, and the error of each.

Reads the predictions eval_model.sh saved in datasets/results_test_set_model/<set>/, so no
model is run. One row per test set; columns: clean map, noisy input, denoised
map, error of the input, error of the denoised map. With --circle, every group
of pixels still more than --threshold degrees wrong is circled on the
clean map, the denoised map and the denoised error map.

Writes MODEL_PIPELINE/figures/error_maps/<run>_map<id>_<sets>.png and prints, per set, how
many pixels remain above the threshold and how many of those lie on a grain
boundary or were misindexed in the input.

Usage
-----
    python plot_error_maps.py --run blind_learned_seed1 --map 1 \
        --sets scatter15 scatter15_mis9 --circle
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle
from scipy import ndimage

# Shared code (orientation maths, paths, scoring) lives in SHARED/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SHARED" / "src"))
from orientation import boundary_mask, disorientation_deg, euler_to_quat, load_euler
from plot_maps import ipf_z_color
from paths import MODEL_RESULTS, clean_map, noise_dir

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128)
INK, INK2 = "#0b0b0b", "#52514e"
ERR_RAMP = LinearSegmentedColormap.from_list(
    "err", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]


def circles(err, max_r=6.0):
    """One circle per group of touching wrong pixels; long groups get several."""
    lab, n = ndimage.label(err, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        todo = list(zip(*np.nonzero(lab == k)))
        while todo:
            cy, cx = todo[0]
            near = [(y, x) for y, x in todo if np.hypot(y - cy, x - cx) <= max_r - 2.5]
            yy, xx = np.array(near).T
            my, mx = yy.mean(), xx.mean()
            r = min(max(np.hypot(yy - my, xx - mx).max() + 2.5, 3.5), max_r)
            out.append((mx, my, r))
            todo = [p for p in todo if p not in near]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="method name of the predictions, e.g. blind_learned_seed1")
    p.add_argument("--map", type=int, default=1)
    p.add_argument("--sets", nargs="+", default=["scatter15", "scatter15_mis9"])
    p.add_argument("--circle", action="store_true", help="circle the pixels still above --threshold")
    p.add_argument("--threshold", type=float, default=10.0, help="degrees")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    stem = f"map_{args.map:05d}_clean_euler_noisy"
    clean = euler_to_quat(load_euler(clean_map(args.map)))
    bnd = boundary_mask(clean, SHAPE)
    ipf = lambda q: ipf_z_color(q).reshape(*SHAPE, 3)
    norm = LogNorm(vmin=0.1, vmax=60)

    plt.rcParams.update({"font.size": 9, "text.color": INK})
    nrow = len(args.sets)
    fig = plt.figure(figsize=(14, 3.15 * nrow + 0.5))
    gs = fig.add_gridspec(nrow, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.05, hspace=0.32,
                          left=0.01, right=0.94, top=1 - 0.55 / (3.15 * nrow + 0.5), bottom=0.02)
    for r, name in enumerate(args.sets):
        f_noisy = noise_dir(name) / f"{stem}.txt"
        noisy = euler_to_quat(load_euler(f_noisy))
        den = euler_to_quat(load_euler(MODEL_RESULTS / name / f"{stem}_noisy__{args.run}.txt"))
        mask = Path(str(f_noisy) + ".badmask.npy")
        bad = np.load(mask).ravel() if mask.exists() else np.zeros(len(clean), bool)
        d_in, d_out = disorientation_deg(noisy, clean), disorientation_deg(den, clean)
        err = d_out > args.threshold
        cs = circles(err.reshape(SHAPE)) if args.circle else []
        print(f"{name:<16} input mean {d_in.mean():.3f} deg -> denoised {d_out.mean():.3f} deg | "
              f"{err.sum()} px > {args.threshold:g} deg, {100 * (err & bnd).sum() / max(err.sum(), 1):.0f}% on a "
              f"boundary, {100 * (err & bad).sum() / max(err.sum(), 1):.0f}% misindexed in the input")
        panels = [(ipf(clean), "clean (truth)", True),
                  (ipf(noisy), f"noisy input: {name}", False),
                  (ipf(den), "denoised", True),
                  (d_in.reshape(SHAPE), f"error of input\nmean {d_in.mean():.2f}°", False),
                  (d_out.reshape(SHAPE), f"error of denoised, mean {d_out.mean():.2f}°\n"
                   f"{err.sum()} px > {args.threshold:g}°" + (" (circled)" if args.circle else ""), True)]
        for c, (img, title, circled) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if img.ndim == 2:
                im = ax.imshow(np.clip(img, 0.1, None), cmap=ERR_RAMP, norm=norm, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            for cx, cy, rad in (cs if circled else []):
                ax.add_patch(Circle((cx, cy), rad, fill=False, edgecolor="black", lw=1.1, path_effects=HALO))
            ax.set_title(title, fontsize=9); ax.axis("off")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 5]))
    cb.set_ticks([0.1, 0.3, 1, 3, 10, 30, 60])
    cb.set_ticklabels(["≤0.1", "0.3", "1", "3", "10", "30", "60"])
    cb.set_label("disorientation to the truth (deg, log scale)", color=INK2)
    fig.suptitle(f"{args.run}, test map {args.map}", x=0.01, ha="left", fontsize=11)

    out = Path(args.out) if args.out else ROOT / "MODEL_PIPELINE/figures/error_maps" / f"{args.run}_map{args.map}_{'_'.join(args.sets)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
