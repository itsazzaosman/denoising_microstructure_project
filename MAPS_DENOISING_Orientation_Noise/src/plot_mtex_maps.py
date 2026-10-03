#!/usr/bin/env python3
"""
Show what MTEX was given and what every MTEX filter gave back, map by map.

One figure per map:

    row 1   truth | MTEX input | input error | IPF-Z colour key
    row 2   each filter run on the noisy map        IPF-Z colours
    row 3   the same filters                        error vs the truth

The clean-* outputs (pre-cleaned, then filtered) are left out.

The MTEX input is read from the _noisy_noisy.ang file itself, not from the
matching noisy .txt, so the figure shows exactly what MTEX loaded. Error is
the per-pixel disorientation from the clean truth under cubic symmetry, on one
log colour scale for every panel and every map, so figures can be compared
side by side.

Usage
-----
    # map 1 at 5 deg scatter -> figures/mtex_maps/scatter5/map_00001.png
    python src/plot_mtex_maps.py --scatter 5

    # several maps, or every map MTEX has finished
    python src/plot_mtex_maps.py --scatter 7 --maps 1 2 3
    python src/plot_mtex_maps.py --scatter 9 --all

    # zoom into rows Y..Y+H, columns X..X+W to see the grain boundaries
    python src/plot_mtex_maps.py --scatter 5 --crop 40 40 48 48

    # any other layout: name the folders yourself
    python src/plot_mtex_maps.py --ang-dir path/to/ang_files \
        --results path/to/mtex_out --out figures/mtex_maps/other
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from orientation import disorientation_deg, euler_to_quat, load_euler
from plot_maps import ipf_rgb, ipf_z_color

ROOT = Path(__file__).resolve().parent.parent

# Display order of the MTEX filters. Any other method found is appended.
FILTERS = ["mean", "median", "kuwahara", "spline", "halfquad", "infconv"]

# From well below the scatter up to the largest cubic disorientation, so
# scatter, smoothing error and misindexing all show on one scale.
ERR_NORM = LogNorm(vmin=0.01, vmax=63.0)
ERR_CMAP = "inferno"


def load_ang(path):
    """
    Euler angles from a TSL .ang file, in raster order (x fastest).

    Pixels are placed by their x/y columns rather than by row order, so this
    stays right for an .ang written in any order. Returns (euler, (ny, nx)).
    """
    d = np.loadtxt(path, comments="#")
    x, y = d[:, 3], d[:, 4]
    xs = np.unique(x)
    step = np.min(np.diff(xs)) if len(xs) > 1 else 1.0
    col = np.rint((x - x.min()) / step).astype(int)
    row = np.rint((y - y.min()) / step).astype(int)
    order = np.lexsort((col, row))          # by row, then by column
    return d[order, :3], (row.max() + 1, col.max() + 1)


def as_image(values, shape, crop):
    img = values.reshape(shape[0], shape[1], -1)
    if crop:
        y, x, h, w = crop
        img = img[y:y + h, x:x + w]
    return img[..., 0] if img.shape[-1] == 1 else img


def draw_ipf_key(ax, n=300):
    """The standard triangle [001]-[101]-[111], stereographic, coloured like the maps."""
    xmax = np.sqrt(2.0) - 1.0               # [101] projected
    ymax = (np.sqrt(3.0) - 1.0) / 2.0       # [111] projected, x = y
    X, Y = np.meshgrid(np.linspace(0, xmax, n), np.linspace(0, ymax, n))
    s = 1.0 + X**2 + Y**2
    v = np.stack([2 * X / s, 2 * Y / s, (1 - X**2 - Y**2) / s], axis=-1).reshape(-1, 3)
    inside = (v[:, 1] <= v[:, 0]) & (v[:, 0] <= v[:, 2])
    rgba = np.concatenate([ipf_rgb(v), inside[:, None].astype(float)], axis=1)
    ax.imshow(rgba.reshape(n, n, 4), origin="lower", extent=(0, xmax, 0, ymax))
    ax.text(0, 0, "[001]", fontsize=8, ha="right", va="top")
    ax.text(xmax, 0, "[101]", fontsize=8, ha="left", va="top")
    ax.text(ymax, ymax, "[111]", fontsize=8, ha="left", va="bottom")
    ax.set_xlim(-0.1, xmax + 0.1)
    ax.set_ylim(-0.08, ymax + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("IPF-Z colour key", fontsize=9)


def in_display_order(names):
    known = [m for m in FILTERS if m in names]
    return known + sorted(set(names) - set(known))


def stats(d):
    return f"mean {d.mean():.2f}$\\degree$  median {np.median(d):.2f}$\\degree$"


def plot_map(mid, args):
    stem = f"map_{mid:05d}_clean_euler"
    e_in, shape = load_ang(args.ang_dir / f"{stem}_noisy_noisy.ang")
    n = shape[0] * shape[1]

    e_clean = load_euler(args.clean_dir / f"{stem}.txt")
    if e_clean.shape[0] != n:
        print(f"  [skip] map {mid}: truth has {e_clean.shape[0]} rows, .ang has {n} pixels")
        return
    q_clean = euler_to_quat(e_clean)
    q_in = euler_to_quat(e_in)
    d_in = disorientation_deg(q_clean, q_in)

    outputs = {f.stem.split("__")[-1]: f
               for f in args.results.glob(f"map_{mid:05d}_*__*.txt")}
    outputs = {name: f for name, f in outputs.items() if not name.startswith("clean-")}
    if not outputs:
        print(f"  [skip] map {mid}: no MTEX outputs in {args.results}")
        return

    results = {}
    for name, f in outputs.items():
        e = load_euler(f)
        if e.shape[0] != n:
            print(f"  [skip] map {mid} {name}: {e.shape[0]} rows, expected {n}")
            continue
        q = euler_to_quat(e)
        results[name] = (q, disorientation_deg(q_clean, q))

    names = in_display_order(list(results))

    ncols = max(4, len(names))
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows),
                             constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    crop = tuple(args.crop) if args.crop else None

    def ipf_panel(ax, title, q):
        ax.imshow(as_image(ipf_z_color(q), shape, crop), interpolation="nearest")
        ax.set_title(title, fontsize=9)

    def err_panel(ax, title, d):
        ax.imshow(as_image(np.clip(d, ERR_NORM.vmin, None)[:, None], shape, crop),
                  cmap=ERR_CMAP, norm=ERR_NORM, interpolation="nearest")
        ax.set_title(f"{title}\n{stats(d)}", fontsize=8)

    # ---- row 1: what MTEX was given ----
    top = axes[0]
    ipf_panel(top[0], "truth (clean)", q_clean)
    ipf_panel(top[1], "MTEX input (.ang)", q_in)
    err_panel(top[2], "MTEX input error", d_in)
    draw_ipf_key(top[3])
    for ax in top[4:]:
        ax.axis("off")
    top[0].set_ylabel("input", fontsize=10)

    # ---- rows 2 and 3: each filter's output, and its error ----
    print(f"map {mid}")
    print(f"  {'method':<18}{'mean':>8}{'median':>8}{'>10deg %':>10}")
    print(f"  {'MTEX input':<18}{d_in.mean():8.3f}{np.median(d_in):8.3f}"
          f"{100 * (d_in > 10).mean():10.2f}")
    ipf_row, err_row = axes[1], axes[2]
    for j, name in enumerate(names):
        q, d = results[name]
        ipf_panel(ipf_row[j], name, q)
        err_panel(err_row[j], name, d)
        print(f"  {name:<18}{d.mean():8.3f}{np.median(d):8.3f}"
              f"{100 * (d > 10).mean():10.2f}")
    for ax in list(ipf_row[len(names):]) + list(err_row[len(names):]):
        ax.axis("off")
    ipf_row[0].set_ylabel("filter output\nIPF-Z", fontsize=10)
    err_row[0].set_ylabel("filter output\nerror vs truth", fontsize=10)

    fig.colorbar(ScalarMappable(norm=ERR_NORM, cmap=ERR_CMAP), ax=axes.ravel().tolist(),
                 shrink=0.5, label="disorientation from truth (degrees, log scale)")
    where = f", rows {crop[0]}-{crop[0] + crop[2]}, cols {crop[1]}-{crop[1] + crop[3]}" if crop else ""
    fig.suptitle(f"Map {mid}: MTEX input and outputs ({args.label}{where})", fontsize=12)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / (f"map_{mid:05d}" + ("_crop" if crop else "") + ".png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scatter", type=float, default=None,
                   help="noise level; fills in --ang-dir, --results and --out for "
                        "datasets/noise_scatterN and datasets/mtex_out_scatterN")
    p.add_argument("--ang-dir", type=Path, default=None,
                   help="folder of map_XXXXX_clean_euler_noisy_noisy.ang (the MTEX inputs)")
    p.add_argument("--results", type=Path, default=None,
                   help="folder of map_XXXXX_..._noisy_noisy__<method>.txt (the MTEX outputs)")
    p.add_argument("--clean-dir", type=Path, default=ROOT / "datasets/clean_euler")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--maps", type=int, nargs="+", default=[1], help="map ids (default: 1)")
    p.add_argument("--all", action="store_true",
                   help="every map that has MTEX outputs in --results")
    p.add_argument("--crop", type=int, nargs=4, default=None, metavar=("Y", "X", "H", "W"),
                   help="zoom into a sub-region")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if args.scatter is not None:
        tag = f"scatter{args.scatter:g}"
        args.ang_dir = args.ang_dir or ROOT / f"datasets/noise_{tag}/ang_files"
        args.results = args.results or ROOT / f"datasets/mtex_out_{tag}"
        args.out = args.out or ROOT / f"figures/mtex_maps/{tag}"
        args.label = f"{args.scatter:g}$\\degree$ scatter"
    else:
        if not (args.ang_dir and args.results):
            p.error("give --scatter, or both --ang-dir and --results")
        args.out = args.out or ROOT / "figures/mtex_maps"
        args.label = args.results.name

    if args.all:
        ids = {int(m.group(1)) for f in args.results.glob("map_*__*.txt")
               if (m := re.search(r"map_(\d+)", f.name))}
        maps = sorted(ids)
    else:
        maps = args.maps
    if not maps:
        sys.exit(f"no MTEX outputs found in {args.results}")

    for mid in maps:
        plot_map(mid, args)


if __name__ == "__main__":
    main()
