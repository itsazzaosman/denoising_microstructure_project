#!/usr/bin/env python3
"""
Slide figures: the curves of make_slide_figures.py, with every test set where
the network is NOT significantly better than its closest MTEX filter marked
on the curve.

The tests are those behind the slide tables (make_slide_tables.test: closest
MTEX filter of each set and measure, paired t-test over maps 1-10,
Holm-corrected over the 29 sets of that measure). A grey band marks no
significant difference, a red band marks the network significantly worse;
the two points compared are ringed. Every unmarked point is significantly
better for the network. The mixed set is not on the scatter grid, so it is
not shown.

Only measures with at least one such set get a figure (CORRUPTED has none).

Writes SHARED/figures/slides/curves_<measure>_tests.png/.pdf.

Usage
-----
    python plot_nonsignificant.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_with_mtex import INK, INK2, MUTED
from make_slide_figures import LEGEND, MEASURES, METHODS, colour, mean_sd, name, save, status, style_axes
from make_slide_tables import SCATTER, fmt_p, load, set_name, test

ROOT = Path(__file__).resolve().parents[2]
BAND = {"ns": "#e4e3dd", "worse": "#fbd9cd"}
LABEL = {"ns": "n.s.", "worse": "worse"}


def curves_marked(vals, key, tests, out):
    stem, _, _, _, levels, ylabel = MEASURES[key]
    gross, square = key == "gross", len(levels) == 4
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED})
    fig, axes = plt.subplots(2 if square else 1, 2 if square else 3, sharex=True, sharey=True,
                             figsize=(7.0, 5.4) if square else (9.6, 3.2), squeeze=False)
    x = np.array(SCATTER, dtype=float)
    for ax, mis, letter in zip(axes.flat, levels, "abcd"):
        sets = [set_name(t, mis) for t in SCATTER]
        ax.plot(x, [np.nanmean(vals[s][key]["input"]) for s in sets], color=MUTED, ls="--", lw=1.0,
                label="noisy input")
        for m in METHODS:
            mu, sd = np.array([mean_sd(vals[s][key][m]) for s in sets]).T
            low = np.minimum(sd, mu if gross else 0.9 * mu)   # bar stays above the axis floor
            ours = m == "ours"
            ax.errorbar(x, mu, yerr=[low, sd], color=colour(m), lw=2.0 if ours else 1.0,
                        marker="o", ms=4.0 if ours else 2.5, elinewidth=0.8, capsize=1.5,
                        zorder=3 if ours else 2, label="ours (3 runs)" if ours else name(m))
        style_axes(ax, gross)
        for theta, s in zip(SCATTER, sets):
            t = tests.get(s)
            if t is None or status(t) == "better":
                continue
            st = status(t)
            ax.axvspan(theta - 0.75, theta + 0.75, color=BAND[st], zorder=0, lw=0)
            for m in (t["best"], "ours"):          # ring the two points the test compares
                ax.plot(theta, np.nanmean(vals[s][key][m]), marker="o", ms=8.5, mfc="none",
                        mec=INK, mew=1.0, zorder=5)
            ax.text(theta, 0.985, f"{LABEL[st]}\np={fmt_p(t['p_holm'])}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.8, color=INK if st == "ns" else "#a8361a",
                    fontweight="bold" if st == "worse" else "normal", linespacing=1.15)
        ax.set_title(f"({letter}) misindexed pixels $p$ = {mis}%", loc="left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"maximum scatter angle $\theta_{\max}$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    # Where curves_<measure>.png puts it, except >10 deg: the bands fill the top of
    # panel (a) there, so the legend moves to the empty top left of panel (b).
    leg_ax, loc = (axes[0, 1], "upper left") if gross else (axes[0, 0], LEGEND[key])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg_ax.legend(handles, labels, frameon=False, fontsize=6.5, ncol=2, loc=loc)
    fig.tight_layout()
    save(fig, out, f"curves_{stem}_tests", 300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "SHARED/figures/slides"))
    args = p.parse_args()
    out = Path(args.out)

    vals = load()
    for key in MEASURES:
        tests = test(vals, key)
        if all(status(t) == "better" for t in tests.values()):
            print(f"{MEASURES[key][1]}: every set significantly better, no figure")
            continue
        curves_marked(vals, key, tests, out)
        print(f"wrote {out}/curves_{MEASURES[key][0]}_tests.png/.pdf")


if __name__ == "__main__":
    main()
