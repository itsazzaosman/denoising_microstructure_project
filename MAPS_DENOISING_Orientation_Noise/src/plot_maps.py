#!/usr/bin/env python3
"""
Visualise orientation maps: clean, noisy, and every denoising result.

Produces two figures:

  *_ipf.png     IPF-Z coloured orientation maps, the familiar view
  *_error.png   per-pixel disorientation from the truth, on a log colour
                scale so both the 0.5 deg scatter and the 40 deg misindexing
                are visible in one image

The error maps are usually the more useful of the two: misindexed pixels are
single specks that are easy to miss in an IPF map but glow brightly here.

Usage
-----
    python plot_maps.py --clean clean.txt --noisy noisy.txt \
        --results mtex_out/ --shape 128 128 --out figures/

    # only some methods, in a chosen order
    python plot_maps.py --clean c.txt --noisy n.txt --results mtex_out/ \
        --shape 128 128 --only spline clean-spline median clean-median

    # zoom into a region to see boundaries properly
    python plot_maps.py ... --crop 40 40 48 48
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --------------------------------------------------------------------------
# orientation maths
# --------------------------------------------------------------------------


def cubic_sym():
    h, r = 0.5, np.sqrt(2.0) / 2.0
    return np.array([
        (1, 0, 0, 0),
        (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
        (r, r, 0, 0), (r, -r, 0, 0), (r, 0, r, 0), (r, 0, -r, 0),
        (r, 0, 0, r), (r, 0, 0, -r),
        (0, r, r, 0), (0, r, -r, 0), (0, r, 0, r), (0, r, 0, -r),
        (0, 0, r, r), (0, 0, r, -r),
        (h, h, h, h), (h, -h, -h, -h), (h, h, -h, h), (h, -h, h, -h),
        (h, -h, h, h), (h, h, -h, -h), (h, -h, -h, h), (h, h, h, -h),
    ], dtype=np.float64)


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_to_quat(e):
    phi1, Phi, phi2 = e[:, 0], e[:, 1], e[:, 2]
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack([c * np.cos(sigma), -s * np.cos(delta),
                  -s * np.sin(delta), -c * np.sin(sigma)], axis=-1)
    q[q[:, 0] < 0] *= -1.0
    return q


def quat_to_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),
    ], axis=-2)


def disorientation_deg(q1, q2, sym):
    m = qmul(q2, qconj(q1))
    best = np.zeros(m.shape[0])
    for si in sym:
        left = qmul(si[None, :], m)
        for sj in sym:
            np.maximum(best, np.abs(qmul(left, sj[None, :])[:, 0]), out=best)
    np.clip(best, -1.0, 1.0, out=best)
    return np.degrees(2.0 * np.arccos(best))


def ipf_z_color(q):
    """
    Standard IPF-Z colouring for cubic m-3m.

    The sample Z direction is expressed in crystal coordinates, folded into
    the standard stereographic triangle using |components| sorted ascending,
    then mapped to RGB with [001] red, [101] green, [111] blue.
    """
    R = quat_to_matrix(q)
    v = R[:, :, 2]                      # sample z in crystal frame
    a = np.sort(np.abs(v), axis=1)      # a <= b <= c, folds into the triangle
    lo, mid, hi = a[:, 0], a[:, 1], a[:, 2]

    rgb = np.stack([hi - mid,
                    (mid - lo) * np.sqrt(2.0),
                    lo * np.sqrt(3.0)], axis=1)
    peak = rgb.max(axis=1, keepdims=True)
    peak[peak < 1e-12] = 1.0
    rgb = np.sqrt(rgb / peak)           # sqrt lifts the midtones, as TSL does
    return np.clip(rgb, 0, 1)


# --------------------------------------------------------------------------


def load_euler(path):
    with open(path) as f:
        first = f.readline()
    try:
        [float(t) for t in first.split()]
        skip = 0
    except ValueError:
        skip = 1
    d = np.loadtxt(path, skiprows=skip)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    return d[:, :3]


def grid(arr, shape, crop):
    g = arr.reshape(shape[0], shape[1], -1)
    if crop:
        y, x, h, w = crop
        g = g[y:y + h, x:x + w]
    return np.squeeze(g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--noisy", required=True)
    p.add_argument("--results", default=None, help="folder of <stem>__<method>.txt")
    p.add_argument("--shape", nargs=2, type=int, required=True, metavar=("NY", "NX"))
    p.add_argument("--out", default="figures")
    p.add_argument("--only", nargs="*", default=None,
                   help="method names to include, in order")
    p.add_argument("--crop", nargs=4, type=int, default=None,
                   metavar=("Y", "X", "H", "W"), help="zoom into a sub-region")
    p.add_argument("--ncols", type=int, default=4)
    args = p.parse_args()

    shape = tuple(args.shape)
    n = int(np.prod(shape))
    sym = cubic_sym()

    e_clean = load_euler(args.clean)
    if e_clean.shape[0] != n:
        sys.exit(f"clean has {e_clean.shape[0]} rows, --shape implies {n}")
    q_clean = euler_to_quat(e_clean)

    panels = [("clean (truth)", q_clean, None)]
    q_noisy = euler_to_quat(load_euler(args.noisy))
    panels.append(("noisy (input)", q_noisy, disorientation_deg(q_clean, q_noisy, sym)))

    if args.results:
        # Keep only results for THIS map before keying by method name. The glob
        # spans every map, so without the filter `found["unet"]` silently picks
        # up whichever map sorted last and plots it against map 1's truth.
        import re
        want = re.search(r"map_(\d+)", Path(args.clean).name)
        files = sorted(Path(args.results).glob("*__*.txt"))
        if want:
            wid = int(want.group(1))
            files = [f for f in files
                     if (m := re.search(r"map_(\d+)", f.name)) and int(m.group(1)) == wid]
        found = {f.stem.split("__")[-1]: f for f in files}
        names = args.only if args.only else sorted(found)
        for name in names:
            if name not in found:
                print(f"  [skip] no result named '{name}'")
                continue
            e = load_euler(found[name])
            if e.shape[0] != n:
                print(f"  [skip] {name}: {e.shape[0]} rows")
                continue
            q = euler_to_quat(e)
            panels.append((name, q, disorientation_deg(q_clean, q, sym)))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    crop = tuple(args.crop) if args.crop else None
    ncols = min(args.ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))

    # ---------------- IPF figure ----------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, q, d) in zip(axes, panels):
        ax.imshow(grid(ipf_z_color(q), shape, crop), interpolation="nearest")
        title = name if d is None else f"{name}\nmedian {np.median(d):.3f}$\\degree$"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("IPF-Z orientation maps", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "maps_ipf.png", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir / 'maps_ipf.png'}")

    # ---------------- error figure ----------------
    err = [(nm, d) for nm, q, d in panels if d is not None]
    ncols_e = min(args.ncols, len(err))
    nrows_e = int(np.ceil(len(err) / ncols_e))
    fig, axes = plt.subplots(nrows_e, ncols_e,
                             figsize=(3.2 * ncols_e, 3.4 * nrows_e))
    axes = np.atleast_1d(axes).ravel()
    norm = LogNorm(vmin=0.01, vmax=63.0)
    for ax, (name, d) in zip(axes, err):
        im = ax.imshow(grid(np.clip(d, 0.01, None)[:, None], shape, crop),
                       cmap="inferno", norm=norm, interpolation="nearest")
        ax.set_title(f"{name}\n>10$\\degree$: {100*(d>10).mean():.2f}%", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(err):]:
        ax.axis("off")
    fig.suptitle("Disorientation from truth (log scale, degrees)", fontsize=12)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="degrees")
    fig.savefig(outdir / "maps_error.png", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir / 'maps_error.png'}")


if __name__ == "__main__":
    main()