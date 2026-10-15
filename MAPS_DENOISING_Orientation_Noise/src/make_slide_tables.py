#!/usr/bin/env python3
"""
Slide tables of the network against the six MTEX filters, one per error measure,
in the style of Atindama et al. (2023), Tables 4-5: one row per noise level, the
best value in bold, the second best in italics, stars for the p-value.

Unlike the paper tables (make_mtex_tables.py), which test every measure against
the filter that is best on ALL, each measure here is tested against its own
closest competitor: the MTEX filter with the lowest mean on that measure. This is
the stricter comparison.

Each table shows two blocks of the test grid (scatter only, and scatter with the
highest misindexing; 3% and 9% for CORRUPTED, which needs misindexing) plus the
mixed set. The Holm correction is still over all sets of the measure.

Reads  figures/mtex_comparison/per_map.csv (written by compare_with_mtex.py).
Writes figures/slides/table_<measure>.png/.pdf and closest_competitor_tests.csv.

Usage
-----
    python make_slide_tables.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats

from compare_with_mtex import holm

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "figures/mtex_comparison/per_map.csv"
FILTERS = ["clean-mean", "clean-median", "clean-kuwahara", "clean-spline",
           "clean-halfquad", "clean-infconv"]
METHODS = ["input"] + FILTERS + ["ours"]
HEAD = ["Input", "Mean", "Median", "Kuwahara", "Spline", "Half-quad.", "Inf. conv.", "Ours"]
COLUMN = {"all": "all_mean", "bnd": "bnd_mean", "bad": "bad_mean", "gross": "gross_pct"}
MEASURES = {   # key: (file name, caption, decimals, misindexing levels of the two blocks)
    "all":   ("table_all", "ALL: mean error over all pixels (degrees)", 3, (0, 9)),
    "bnd":   ("table_boundary", "BOUNDARY: mean error on grain-boundary pixels (degrees)", 3, (0, 9)),
    "bad":   ("table_corrupted", "CORRUPTED: mean error on misindexed pixels (degrees)", 2, (3, 9)),
    "gross": ("table_gross", ">10°: pixels more than 10° from the truth (% of the map)", 3, (0, 9)),
}
SCATTER = [1, 3, 5, 7, 9, 12, 15]
INK, INK_2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
TINT = "#fbe9ec"      # light CMU red behind the "Ours" column
MINUS = "−"


def set_name(theta, p):
    return f"scatter{theta}" + (f"_mis{p}" if p else "")


def load():
    """vals[set][measure][method] = per-map values, maps in the same order for every method."""
    raw = defaultdict(lambda: defaultdict(dict))            # set, method, map -> row
    with open(IN) as f:
        for r in csv.DictReader(f):
            raw[r["set"]][r["method"]][int(r["map"])] = r
    vals = defaultdict(lambda: defaultdict(dict))
    for s, by_method in raw.items():
        maps = sorted(set.intersection(*(set(by_method[m]) for m in METHODS)))
        for key, col in COLUMN.items():
            for m in METHODS:
                vals[s][key][m] = np.array([float(by_method[m][i][col]) for i in maps])
    return vals


def test(vals, key):
    """Ours against the closest MTEX filter of each set: mean difference, 95% CI, p, Holm p."""
    out = {}
    for s, by_key in vals.items():
        v = by_key[key]
        if not np.isfinite(v["ours"]).any():
            continue                                         # CORRUPTED without misindexing
        best = min(FILTERS, key=lambda f: np.nanmean(v[f]))
        d = v["ours"] - v[best]
        n, mean = len(d), float(d.mean())
        half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        p = stats.ttest_rel(v["ours"], v[best]).pvalue if np.any(d != 0) else np.nan
        out[s] = {"best": best, "maps": n, "ours": float(v["ours"].mean()), "mtex": float(v[best].mean()),
                  "diff": mean, "lo": mean - half, "hi": mean + half, "p": p}
    for s, q in zip(out, holm([t["p"] for t in out.values()])):
        out[s]["p_holm"] = q
    return out


def fmt(x, dec):
    if round(x, dec) == 0:
        x = 0.0                                             # no "-0.000"
    return f"{x:.{dec}f}".replace("-", MINUS)


def fmt_p(p):
    if not np.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def stars(p):
    if not np.isfinite(p) or p >= 0.05:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*"


def draw(vals, tests, key, out_dir):
    name, caption, dec, blocks = MEASURES[key]
    rows = []                                               # ("block", title) or ("row", label, set)
    for p in blocks:
        rows.append(("block", "Scatter only, no misindexing" if p == 0
                     else f"Scatter + {p}% misindexed pixels"))
        rows += [("row", f"{t}°", set_name(t, p)) for t in SCATTER]
    rows += [("block", "Mixed noise levels (random level per map)"), ("row", "Mixed", "test_random")]

    # x positions (axes fraction): right edge of each numeric column
    x_label = 0.012
    x_num = [0.135, 0.215, 0.295, 0.385, 0.465, 0.555, 0.645, 0.735]   # Input .. Ours
    x_delta, x_p = 0.925, 0.995
    line = 1.0 / (len(rows) + 5.6)                          # height of one text line
    plt.rcParams["font.family"] = "Liberation Sans"
    fig, ax = plt.subplots(figsize=(12.3, 6.4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 12.5

    y = 1 - 0.55 * line
    ax.text(x_label, y, caption, fontsize=fs - 1, color=INK_2, va="center")
    y -= 1.25 * line
    ax.text(x_label, y, "θmax", fontsize=fs, color=INK, va="center", weight="bold")
    for x, h in zip(x_num, HEAD):
        ax.text(x, y, h, fontsize=fs, color=INK if h != "Input" else MUTED, va="center",
                ha="right", weight="bold")
    ax.text(x_delta, y, "Δ vs best MTEX [95% CI]", fontsize=fs, color=INK, va="center",
            ha="right", weight="bold")
    ax.text(x_p, y, "p", fontsize=fs, color=INK, va="center", ha="right", weight="bold",
            style="italic")
    top_rule = y - 0.6 * line
    ax.plot([0.005, 0.995], [top_rule] * 2, color=INK, lw=1.0)
    y = top_rule - 0.15 * line

    any_worse = False
    for r in rows:
        y -= line
        if r[0] == "block":
            ax.text(x_label, y, r[1], fontsize=fs - 1, color=INK_2, va="center", style="italic")
            continue
        _, label, s = r
        v = {m: float(np.nanmean(vals[s][key][m])) for m in METHODS}
        shown = {m: fmt(v[m], dec) for m in METHODS}
        ranked = sorted({float(shown[m].replace(MINUS, "-")) for m in METHODS if m != "input"})
        first, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        n_first = sum(float(shown[m].replace(MINUS, "-")) == first for m in METHODS if m != "input")
        t = tests[s]
        ax.text(x_label, y, label, fontsize=fs, color=INK, va="center")
        for x, m in zip(x_num, METHODS):
            val = float(shown[m].replace(MINUS, "-"))
            weight = "bold" if m != "input" and val == first else "normal"
            style = ("italic" if m != "input" and n_first == 1 and val == second else "normal")
            ax.text(x, y, shown[m], fontsize=fs, va="center", ha="right", weight=weight,
                    style=style, color=MUTED if m == "input" else INK)
        worse = t["diff"] > 0 and t["p_holm"] < 0.05
        any_worse = any_worse or worse
        st = "†" if worse else stars(t["p_holm"]) if t["diff"] < 0 else ""
        if st:
            ax.text(x_num[-1] + 0.002, y + 0.18 * line, st, fontsize=fs - 3, va="center",
                    ha="left", color=INK)
        delta = f"{fmt(t['diff'], dec)} [{fmt(t['lo'], dec)}, {fmt(t['hi'], dec)}]"
        ax.text(x_delta, y, delta, fontsize=fs, va="center", ha="right", color=INK)
        ax.text(x_p, y, fmt_p(t["p_holm"]), fontsize=fs, va="center", ha="right", color=INK)

    bottom_rule = y - 0.6 * line
    ax.plot([0.005, 0.995], [bottom_rule] * 2, color=INK, lw=1.0)
    ax.add_patch(Rectangle((x_num[-2] + 0.012, bottom_rule), x_num[-1] - x_num[-2] + 0.012,
                           top_rule + 1.6 * line - bottom_rule, color=TINT, zorder=-1, lw=0))
    sets = "22 sets" if key == "bad" else "29 sets"
    note = (f"Mean over test maps 1–10 (ours: mean of 3 training runs).  Bold = best, italic = second best.  "
            f"Δ = ours − best MTEX filter on this measure.\n"
            f"p: paired t-test over the 10 maps, Holm-corrected over all {sets}.  "
            f"* p < 0.05   ** p < 0.01   *** p < 0.001."
            + ("   † ours significantly worse." if any_worse else "")
            + "  All MTEX filters with pre-cleaning (clean-).")
    ax.text(x_label, bottom_rule - 0.45 * line, note, fontsize=fs - 3.5, color=MUTED, va="top",
            linespacing=1.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "figures/slides"))
    args = p.parse_args()
    out = Path(args.out)

    vals = load()
    rows = []
    for key in MEASURES:
        tests = test(vals, key)
        draw(vals, tests, key, out)
        sig = [s for s, t in tests.items() if t["p_holm"] < 0.05 and t["diff"] < 0]
        lower = [s for s, t in tests.items() if t["diff"] < 0]
        print(f"{key:6s} ours lower in {len(lower)}/{len(tests)} sets, significantly in {len(sig)}; "
              f"not significant: {sorted(set(tests) - set(sig))}")
        for s, t in tests.items():
            rows.append({"set": s, "measure": key, "closest_mtex": t["best"], "maps": t["maps"],
                         **{k: t[k] for k in ("ours", "mtex", "diff", "lo", "hi", "p", "p_holm")}})
    with open(out / "closest_competitor_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}/table_*.png/.pdf and closest_competitor_tests.csv")


if __name__ == "__main__":
    main()
