#!/usr/bin/env python3
"""
Preview one map at chosen noise levels, before deciding which levels become
test sets.

The noise is made on the fly by add_ebsd_noise.add_noise with the seed the test
sets use (seed + map index), so a level already in the grid looks exactly like
its test-set file, and a level that is not (say 15 % misindexed) looks like the
test set it would become. Only the figure is written; no data.

    first column    the clean map, and the IPF-Z colour key under it
    other columns   one per misindexing level
    rows            one per scatter level

Usage
-----
    # one level
    python preview_noise.py --map 1 --scatter 5 --misindex 3

    # several levels side by side, e.g. is 15 % misindexed worth adding?
    python preview_noise.py --map 1 --scatter 1 5 9 --misindex 0 9 15
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from add_ebsd_noise import add_noise, euler_to_quat, load_euler
from paths import clean_map
from orientation import ipf_z_color
from plot_noise_examples import ipf_key

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=int, required=True, help="map id, e.g. 1 for map_00001")
    p.add_argument("--scatter", type=float, nargs="+", required=True,
                   help="max scatter angle(s) in deg; the angle is uniform in [0, max]")
    p.add_argument("--misindex", type=float, nargs="+", required=True,
                   help="misindexed pixels in percent, e.g. 3 for 3%%")
    p.add_argument("--boundary-bias", type=float, default=4.0,
                   help="how much more likely a boundary pixel is to be misindexed (test sets: 4)")
    p.add_argument("--seed", type=int, default=0, help="base seed (test sets: 0)")
    p.add_argument("--shape", nargs=2, type=int, default=[128, 128], metavar=("NY", "NX"))
    p.add_argument("--out", help="output .png (default SHARED/figures/noise_preview/map_<id>_s<...>_m<...>.png)")
    args = p.parse_args()

    shape = tuple(args.shape)
    stem = f"map_{args.map:05d}"
    euler = load_euler(clean_map(args.map))
    image = lambda e: ipf_z_color(euler_to_quat(e)).reshape(*shape, 3)

    ns, nm = len(args.scatter), len(args.misindex)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(max(ns, 2), nm + 1, figsize=(2.0 * (nm + 1), 2.4 * max(ns, 2)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    axes[0, 0].imshow(image(euler), interpolation="nearest")
    axes[0, 0].set_title(f"{stem}\nclean")
    ipf_key(axes[1, 0])

    for r, s in enumerate(args.scatter):
        for c, m in enumerate(args.misindex):
            # The test sets give map i (counting from 0) the generator seed + i,
            # and a fresh one per set, so every panel starts from the same draw.
            rng = np.random.default_rng(args.seed + args.map - 1)
            noisy, bad = add_noise(euler, shape, s, m / 100, args.boundary_bias, rng)
            ax = axes[r, c + 1]
            ax.imshow(image(noisy), interpolation="nearest")
            ax.set_title(f"scatter {s:g}°, misindex {m:g}%\n({len(bad)} px replaced)")

    levels = lambda v: "-".join(f"{x:g}" for x in v)
    out = Path(args.out) if args.out else (
        ROOT / "SHARED/figures/noise_preview" / f"{stem}_s{levels(args.scatter)}_m{levels(args.misindex)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
