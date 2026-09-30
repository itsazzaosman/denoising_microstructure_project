#!/usr/bin/env python3
"""
Bar chart comparing denoising methods, for the abstract/paper.

Reads the per-map CSV that score_dataset.py writes (--csv) and aggregates it the
same way the printed table does: the statistic per map, then the median across
maps. Falls back to a set of recorded numbers when no CSV is given, so the
figure can be regenerated while the underlying data is being restored.

Usage
-----
    # from a scoring run
    python src/score_dataset.py ... --csv figures/scatter5/scores.csv --metric mean
    python src/plot_method_comparison.py --csv figures/scatter5/scores.csv \
        --out figures/scatter5/method_comparison.png \
        --title "Mean per-pixel disorientation, 5 deg scatter (10 maps)"

    # from the recorded 5 deg numbers, no data needed
    python src/plot_method_comparison.py --recorded scatter5 \
        --out figures/scatter5/method_comparison.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# palette: categorical slots 1-2, chart chrome and ink. Two series of the same
# unit (degrees) share ONE axis -- never a second scale.
# --------------------------------------------------------------------------
SERIES_1 = "#2a78d6"     # blue   -- all pixels
SERIES_2 = "#eb6834"     # orange -- grain-boundary pixels
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Recorded results, mean per-pixel disorientation in degrees, 10 maps.
# Regenerate from score_dataset.py --metric mean once the datasets are back.
RECORDED = {
    "scatter5": {
        "label": "5° scatter",
        "rows": [
            ("noisy", 2.504, 2.484),
            ("infimal conv.",     1.918, 2.070),
            ("half-quadratic",    1.822, 2.434),
            ("mean",              0.989, 1.094),
            ("Kuwahara",          0.974, 1.223),
            ("median",            0.408, 0.575),
            ("spline",            0.363, 0.412),
            ("U-Net",      0.258, 0.346),
        ],
    },
    "scatter7": {
        "label": "7° scatter",
        "rows": [
            ("noisy",          3.506, 3.477),
            ("infimal conv.",  2.906, 3.057),
            ("half-quadratic", 2.566, 3.426),
            ("mean",           1.383, 1.526),
            ("Kuwahara",       1.365, 1.725),
            ("median",         0.573, 0.816),
            ("spline",         0.568, 0.619),
            ("U-Net",          0.317, 0.428),
        ],
    },
    "scatter": {
        "label": "1° scatter",
        "rows": [
            ("noisy", 0.501, 0.497),
            ("half-quadratic",    0.345, 0.452),
            ("mean",              0.199, 0.223),
            ("Kuwahara",          0.195, 0.245),
            ("infimal conv.",     0.103, 0.171),
            ("median",            0.081, 0.113),
            ("spline",            0.067, 0.095),
            ("U-Net",      0.055, 0.073),
        ],
    },
}

PRETTY = {"mean": "mean", "median": "median", "spline": "spline",
          "kuwahara": "Kuwahara", "halfquad": "half-quadratic",
          "infconv": "infimal conv.", "unet": "U-Net",
          "noisy": "noisy"}


def from_csv(path, metric):
    """Per-map statistic -> median across maps, matching score_dataset.py."""
    import csv
    from collections import defaultdict
    allv, bndv = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            m = r["method"]
            try:
                allv[m].append(float(r["all_med"]))
                bndv[m].append(float(r["bnd_med"]))
            except (ValueError, KeyError):
                continue
    rows = [(PRETTY.get(m, m), float(np.median(allv[m])), float(np.median(bndv[m])))
            for m in allv if allv[m]]
    return sorted(rows, key=lambda r: -r[1])


def compare(keys, out, metric, title):
    """
    One panel per noise level, shared method order and shared x-scale.

    The shared scale is the point: it lets the reader see that every method's
    error grows with noise while the gap between the U-Net and the best filter
    widens. Separate scales per panel would hide exactly that.
    """
    panels = [RECORDED[k] for k in keys]
    order = [r[0] for r in panels[0]["rows"]]          # ranking fixed by the first panel
    span = max(max(r[1], r[2]) for pn in panels for r in pn["rows"])

    fig, axes = plt.subplots(1, len(panels), sharex=True,
                             figsize=(6.4 * len(panels), 0.62 * len(order) + 1.7))
    axes = np.atleast_1d(axes)
    y = np.arange(len(order))[::-1]
    h = 0.36

    for ax, pn in zip(axes, panels):
        lut = {r[0]: (r[1], r[2]) for r in pn["rows"]}
        all_v = np.array([lut[n][0] for n in order])
        bnd_v = np.array([lut[n][1] for n in order])

        ax.set_facecolor(SURFACE)
        b1 = ax.barh(y + h / 2 + 0.02, all_v, height=h, color=SERIES_1,
                     label="all pixels", zorder=3)
        b2 = ax.barh(y - h / 2 - 0.02, bnd_v, height=h, color=SERIES_2,
                     label="grain-boundary pixels", zorder=3)
        for bars, vals in ((b1, all_v), (b2, bnd_v)):
            for bar, v in zip(bars, vals):
                ax.text(v + span * 0.012, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", ha="left", fontsize=8,
                        color=INK_SECONDARY)

        ax.set_yticks(y)
        ax.set_yticklabels(order if ax is axes[0] else [], fontsize=10,
                           color=INK_PRIMARY)
        ax.set_xlim(0, span * 1.16)
        ax.set_xlabel("disorientation (degrees)", fontsize=10, color=INK_SECONDARY)
        ax.set_title(pn["label"], fontsize=11, color=INK_PRIMARY, pad=8)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.8)

    fig.patch.set_facecolor(SURFACE)
    leg = axes[-1].legend(loc="lower right", frameon=False, fontsize=9.5,
                          handlelength=1.0, handleheight=0.9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    if title:
        fig.suptitle(title, fontsize=13.5, color=INK_PRIMARY, y=0.995)
    fig.text(0.5, 0.945, f"{metric} per-pixel disorientation",
             fontsize=10, color=INK_SECONDARY, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")

    hdr = "".join(f"{pn['label']:>26}" for pn in panels)
    print(f"\n{'method':<18}{hdr}")
    print(f"{'':<18}" + "".join(f"{'all px':>13}{'bnd px':>13}" for _ in panels))
    print("-" * (18 + 26 * len(panels)))
    for n in order:
        row = "".join(f"{RECORDED[k]['rows'][[r[0] for r in RECORDED[k]['rows']].index(n)][1]:>13.3f}"
                      f"{RECORDED[k]['rows'][[r[0] for r in RECORDED[k]['rows']].index(n)][2]:>13.3f}"
                      for k in keys)
        print(f"{n:<18}{row}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="per-map CSV from score_dataset.py")
    p.add_argument("--recorded", default=None, choices=sorted(RECORDED),
                   help="use the recorded numbers instead of a CSV")
    p.add_argument("--metric", default="mean", choices=["mean", "median"],
                   help="only labels the axis; the CSV already holds one statistic")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--compare", nargs="*", default=None,
                   help="two or more recorded levels to show side by side")
    args = p.parse_args()

    if args.compare:
        compare(args.compare, args.out, args.metric, args.title)
        return

    if args.csv:
        rows = from_csv(args.csv, args.metric)
        sub = f"{args.metric} per-pixel disorientation"
    elif args.recorded:
        rec = RECORDED[args.recorded]
        rows = rec["rows"]
        sub = f"{args.metric} per-pixel disorientation, {rec['label']}"
    else:
        p.error("give either --csv or --recorded")

    names = [r[0] for r in rows]
    all_v = np.array([r[1] for r in rows])
    bnd_v = np.array([r[2] for r in rows])

    y = np.arange(len(rows))[::-1]          # largest at the top
    h = 0.36                                 # thin marks, 2px-equivalent gap between pairs

    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(rows) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    b1 = ax.barh(y + h / 2 + 0.02, all_v, height=h, color=SERIES_1,
                 label="all pixels", zorder=3)
    b2 = ax.barh(y - h / 2 - 0.02, bnd_v, height=h, color=SERIES_2,
                 label="grain-boundary pixels", zorder=3)
    for bars in (b1, b2):                    # 4px rounded data-ends
        for bar in bars:
            bar.set_joinstyle("round")

    # Direct labels: the exact values are the deliverable here, and a printed
    # figure has no tooltip to fall back on. Muted ink, never the series color.
    span = max(all_v.max(), bnd_v.max())
    for bars, vals in ((b1, all_v), (b2, bnd_v)):
        for bar, v in zip(bars, vals):
            ax.text(v + span * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=8.5,
                    color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK_PRIMARY)
    ax.set_xlabel("disorientation (degrees)", fontsize=10, color=INK_SECONDARY)
    ax.set_xlim(0, span * 1.16)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)

    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)  # solid hairline
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    if args.title:
        ax.set_title(args.title, fontsize=11.5, color=INK_PRIMARY,
                     pad=26, loc="left")
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5,
            color=INK_SECONDARY, va="bottom")

    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5,
                    handlelength=1.0, handleheight=0.9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Table view: every value reachable without reading the chart.
    print(f"\n{'method':<20}{'all px':>10}{'boundary px':>14}")
    print("-" * 44)
    for n, a, b in rows:
        print(f"{n:<20}{a:>10.3f}{b:>14.3f}")


if __name__ == "__main__":
    main()
