#!/usr/bin/env python3
"""
One test map cleaned by every method, as a 3 x 3 grid of IPF-Z maps in the style of
Atindama et al. (2023), Fig. 2:

    clean            | noisy input    | MTEX mean
    MTEX median      | MTEX Kuwahara  | MTEX spline
    MTEX half-quad.  | MTEX inf. conv.| network

The title of every map gives its mean per-pixel error against the clean map. The
MTEX maps are the pre-cleaned (clean-*) filters; the network map is one training
run (--run). Writes SHARED/figures/slides/maps_all_filters_<set>.png and .pdf.

Usage
-----
    python plot_all_filters.py
    python plot_all_filters.py --set scatter5_mis6 --map 3
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orientation import disorientation_deg, euler_to_quat, load_euler
from paths import MODEL_RESULTS, MTEX_RESULTS, clean_map, noise_dir
from plot_maps import ipf_z_color
from plot_mtex_examples import NAMES, label
from plot_noise_examples import ipf_key

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128)
INK, INK2 = "#0b0b0b", "#52514e"
FILTERS = ["clean-mean", "clean-median", "clean-kuwahara", "clean-spline",
           "clean-halfquad", "clean-infconv"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="scatter15_mis9")
    p.add_argument("--map", type=int, default=1)
    p.add_argument("--run", default="blind_learned_seed1")
    p.add_argument("--out", default=str(ROOT / "SHARED/figures/slides"))
    args = p.parse_args()

    stem = f"map_{args.map:05d}_clean_euler_noisy"
    clean = euler_to_quat(load_euler(clean_map(args.map)))
    maps = [("clean", clean),
            ("noisy input", euler_to_quat(load_euler(noise_dir(args.set) / f"{stem}.txt")))]
    maps += [(f"MTEX {NAMES[f]}", euler_to_quat(load_euler(MTEX_RESULTS / args.set / f"{stem}_noisy__{f}.txt")))
             for f in FILTERS]
    maps.append(("ours", euler_to_quat(load_euler(MODEL_RESULTS / args.set / f"{stem}_noisy__{args.run}.txt"))))

    plt.rcParams.update({"font.size": 8, "text.color": INK})
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.06, hspace=0.22,
                          left=0.01, right=0.99, top=0.9, bottom=0.01)
    for k, (name, q) in enumerate(maps):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.imshow(ipf_z_color(q).reshape(*SHAPE, 3), interpolation="nearest")
        err = "" if name == "clean" else f"\nmean error {disorientation_deg(q, clean).mean():.2f}°"
        ax.set_title(f"({'abcdefghi'[k]}) {name}{err}", fontsize=8.5,
                     weight="bold" if name == "ours" else "normal")
        ax.axis("off")
        print(f"{name:22s}{err.strip()}")
    ipf_key(fig.add_subplot(gs[1, 3]))
    fig.suptitle(f"Test map {args.map}, {label(args.set)}", fontsize=10, color=INK)

    out = Path(args.out) / f"maps_all_filters_{args.set}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out}.png (+ .pdf)")


if __name__ == "__main__":
    main()
