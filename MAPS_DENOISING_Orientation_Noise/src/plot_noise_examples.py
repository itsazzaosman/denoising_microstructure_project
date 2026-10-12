#!/usr/bin/env python3
"""
Paper figure: one test map, clean and under the noise of several test sets.

    row 1   clean, then scatter only:  theta_max = 1, 5, 9, 15 deg
    row 2   misindexing only, p = 3, 6, 9 %, and where the misindexed pixels
            of the p = 9 % map fall relative to the grain boundaries
    row 3   both together, from the mildest to the most severe test set:
            (1 deg, 3 %), (5 deg, 6 %), (9 deg, 9 %), (15 deg, 9 %)

Every test set has some scatter (the grid starts at 1 deg), so the
misindexing-only panels take the replaced pixels of the 1 deg sets and put
them on the clean map. A replaced pixel is a fresh random orientation that does
not depend on the scatter, so this is exactly the misindexing in those sets.

Every panel is read from the test-set files themselves, so the figure shows
exactly the data the methods are scored on. Maps are coloured with the
standard IPF-Z key for cubic symmetry, drawn alongside.

Writes figures/paper/noise_examples.pdf and .png, and prints the numbers the
caption quotes.

Usage
-----
    python plot_noise_examples.py --map 1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from orientation import boundary_mask, euler_to_quat, load_euler
from plot_maps import ipf_rgb, ipf_z_color

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (128, 128)
INK, INK_2 = "#0b0b0b", "#52514e"        # primary and secondary text
GRID_GREY, ACCENT = "#c3c2b7", "#2a78d6"   # recessive boundaries, one accent

ROW1 = [(None, "(a) clean"),
        ("scatter1", r"(b) $\theta_{\mathrm{max}}$ = 1°"),
        ("scatter5", r"(c) $\theta_{\mathrm{max}}$ = 5°"),
        ("scatter9", r"(d) $\theta_{\mathrm{max}}$ = 9°"),
        ("scatter15", r"(e) $\theta_{\mathrm{max}}$ = 15°")]
ROW2 = [("scatter1_mis3", "(f) misindexing\n$p$ = 3%"),
        ("scatter1_mis6", "(g) misindexing\n$p$ = 6%"),
        ("scatter1_mis9", "(h) misindexing\n$p$ = 9%")]
ROW3 = [("scatter1_mis3", "(j) $\\theta_{\\mathrm{max}}$ = 1°\n$p$ = 3%"),
        ("scatter5_mis6", "(k) $\\theta_{\\mathrm{max}}$ = 5°\n$p$ = 6%"),
        ("scatter9_mis9", "(l) $\\theta_{\\mathrm{max}}$ = 9°\n$p$ = 9%"),
        ("scatter15_mis9", "(m) $\\theta_{\\mathrm{max}}$ = 15°\n$p$ = 9%")]


def ipf_key(ax, n=400):
    """IPF colour key: the standard [001]-[101]-[111] triangle, stereographic."""
    X, Y = np.meshgrid(np.linspace(0, 0.43, n), np.linspace(0, 0.39, n))
    r2 = X ** 2 + Y ** 2
    v = np.stack([2 * X, 2 * Y, 1 - r2], -1) / (1 + r2)[..., None]
    inside = (Y <= X) & (v[..., 0] <= v[..., 2])
    rgb = ipf_rgb(v.reshape(-1, 3)).reshape(n, n, 3)
    rgba = np.concatenate([rgb, inside[..., None].astype(float)], -1)
    ax.imshow(rgba, origin="lower", extent=(0, 0.43, 0, 0.39), interpolation="bilinear")
    t = np.tan(np.radians(54.74) / 2)      # [111] lies at 54.74 deg from [001]
    p111 = (t / np.sqrt(2), t / np.sqrt(2))
    p101 = (np.tan(np.radians(45) / 2), 0.0)
    for (x, y), lab, ha, va, dx, dy in [((0, 0), "001", "right", "top", -0.01, -0.01),
                                        (p101, "101", "left", "top", 0.01, -0.01),
                                        (p111, "111", "left", "bottom", 0.0, 0.01)]:
        ax.text(x + dx, y + dy, lab, color=INK, fontsize=7, ha=ha, va=va)
    ax.set_xlim(-0.1, 0.52); ax.set_ylim(-0.16, 0.42)    # room for the corner labels
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.215, -0.1, "IPF-Z key", fontsize=7, color=INK_2, ha="center", va="top")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=int, default=1)
    p.add_argument("--out", default=str(ROOT / "figures/paper/noise_examples"))
    args = p.parse_args()

    stem = f"map_{args.map:05d}"
    clean = euler_to_quat(load_euler(ROOT / f"datasets/clean_euler/{stem}_clean_euler.txt"))

    def noisy(name):
        f = ROOT / f"datasets/noise_{name}/{stem}_clean_euler_noisy.txt"
        return euler_to_quat(load_euler(f)), np.load(str(f) + ".badmask.npy").ravel()

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "text.color": INK})
    fig = plt.figure(figsize=(7.0, 4.95))
    # Column 5 is an empty spacer, so the IPF key never touches panel (e).
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 0.1, 0.8],
                          left=0.005, right=0.995, top=0.963, bottom=0.005,
                          wspace=0.06, hspace=0.24)

    for col, (name, title) in enumerate(ROW1):
        q = clean if name is None else noisy(name)[0]
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(ipf_z_color(q).reshape(*SHAPE, 3), interpolation="nearest")
        ax.set_title(title); ax.axis("off")
    for col, (name, title) in enumerate(ROW2):
        q, bad = noisy(name)
        q = np.where(bad[:, None], q, clean)        # replaced pixels only, no scatter
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(ipf_z_color(q).reshape(*SHAPE, 3), interpolation="nearest")
        ax.set_title(title); ax.axis("off")

    # (j)-(m) scatter and misindexing together, straight from the test sets
    for col, (name, title) in enumerate(ROW3):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(ipf_z_color(noisy(name)[0]).reshape(*SHAPE, 3), interpolation="nearest")
        ax.set_title(title); ax.axis("off")

    # (i) where misindexing lands: boundaries recessive grey, misindexed pixels one accent
    bnd = boundary_mask(clean, SHAPE)
    _, bad = noisy("scatter1_mis9")
    layer = np.zeros(bnd.shape, dtype=int)
    layer[bnd] = 1
    layer[bad] = 2
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(layer.reshape(SHAPE), cmap=ListedColormap(["#ffffff", GRID_GREY, ACCENT]),
              vmin=0, vmax=2, interpolation="nearest")
    ax.set_title("(i) misindexed pixels\nof map (h)"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID_GREY); s.set_linewidth(0.6)

    ipf_key(fig.add_subplot(gs[0, 6]))
    lg = fig.add_subplot(gs[1, 4]); lg.axis("off")      # legend next to panel (i)
    for y, colour, text in [(0.62, ACCENT, "misindexed\npixel in (h)"), (0.36, GRID_GREY, "grain\nboundary")]:
        lg.add_patch(plt.Rectangle((0.02, y - 0.05), 0.14, 0.1, color=colour, transform=lg.transAxes))
        lg.text(0.22, y, text, transform=lg.transAxes, va="center", fontsize=7, color=INK_2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)

    on_bnd = bad[bnd].sum() / bad.sum()
    print(f"wrote {out}.pdf and .png (map {args.map})")
    print(f"caption numbers: boundary pixels {100 * bnd.mean():.1f}% of the map; "
          f"misindexed pixels in (h) {bad.sum()} ({100 * bad.mean():.1f}%), "
          f"{100 * on_bnd:.0f}% of them on a boundary")


if __name__ == "__main__":
    main()
