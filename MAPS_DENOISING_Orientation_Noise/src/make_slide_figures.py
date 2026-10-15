#!/usr/bin/env python3
"""
Slide figures of the network against the six MTEX filters (maps 1-10 of every test set).

  curves_<measure>.png  error against scatter, one panel per misindexing level, like
                        figures/paper/mtex_comparison.png (which shows ALL only)
  mixed_set.png         the four measures on the mixed set
  stats_<measure>.png   per test set: ours - closest MTEX filter, 95% CI and p
  stats_summary.png     one line per measure

In the curves and bars, points are the mean over the maps and the bars the standard
deviation across them: the spread between maps, not a confidence interval. The tests
are those of make_slide_tables.py: each measure against its own closest MTEX filter,
paired t-test over the maps, Holm-corrected over the sets of that measure.

Usage
-----
    python make_slide_figures.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

from compare_with_mtex import COLOURS, GRID, INK, INK2, MUTED, PRETTY
from make_slide_tables import FILTERS, SCATTER, fmt, fmt_p, load, set_name, stars, test

ROOT = Path(__file__).resolve().parent.parent
METHODS = FILTERS + ["ours"]
# key: (file name, name, unit, decimals, misindexing levels, y-axis label)
MEASURES = {
    "all":   ("all", "ALL", "deg", 3, (0, 3, 6, 9), "mean error, all pixels (deg)"),
    "bnd":   ("boundary", "BOUNDARY", "deg", 3, (0, 3, 6, 9), "mean error, boundary pixels (deg)"),
    "bad":   ("corrupted", "CORRUPTED", "deg", 2, (3, 6, 9), "mean error, misindexed pixels (deg)"),
    "gross": ("gross", ">10°", "% of pixels", 3, (0, 3, 6, 9), "pixels more than 10° off (% of map)"),
}
LEGEND = {"all": "lower right", "bnd": "lower right", "bad": "lower right", "gross": "upper left"}
GOOD, NS, BAD = "#e3eefb", "#f1f0ec", "#fde3d9"     # cell tints: better / not significant / worse


def name(m):
    return "ours" if m == "ours" else PRETTY[m.removeprefix("clean-")]


def colour(m):
    return COLOURS[m.removeprefix("clean-")]


def mean_sd(v):
    v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1))


def status(t):
    if t["p_holm"] < 0.05:
        return "better" if t["diff"] < 0 else "worse"
    return "ns"


def set_label(s):
    if s == "test_random":
        return "mixed"
    theta, _, p = s.removeprefix("scatter").partition("_mis")
    return f"{theta}°, {p or 0}%"


def save(fig, out, stem, dpi):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=dpi)
    plt.close(fig)


def style_axes(ax, gross):
    if gross:   # zeros are common here, so the axis is linear below 0.01 % and log above
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.4)
        ax.yaxis.set_major_locator(FixedLocator([0, 0.01, 0.1, 1, 10]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(-0.002, 70)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(SCATTER)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def curves(vals, key, out):
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
        ax.set_title(f"({letter}) misindexed pixels $p$ = {mis}%", loc="left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"maximum scatter angle $\theta_{\max}$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if key == "bad":
        axes[0, 0].set_ylim(bottom=0.15)                      # room for the legend under the curves
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2, loc=LEGEND[key])
    fig.tight_layout()
    save(fig, out, f"curves_{stem}", 300)


def mixed(vals, out):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED})
    order = ["ours"] + FILTERS + ["input"]                    # top to bottom
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.3), sharey=True)
    for ax, (key, (_, title, unit, _, _, _)) in zip(axes, MEASURES.items()):
        mu, sd = np.array([mean_sd(vals["test_random"][key][m]) for m in order]).T
        cols = [MUTED if m == "input" else colour(m) for m in order]
        ax.barh(y, mu, color=cols, height=0.7, xerr=sd,
                error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK2})
        for yi, m_, s_, m in zip(y, mu, sd, order):
            ax.text(m_ + s_ + 0.02 * (mu + sd).max(), yi, f"{m_:.2f}", va="center", fontsize=7,
                    color=INK, weight="bold" if m == "ours" else "normal")
        ax.set_xlim(0, 1.25 * (mu + sd).max())
        ax.set_title(f"{title} ({unit})", loc="left", fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["ours (3 runs)"] + [name(f) for f in FILTERS] + ["noisy input"])
    axes[0].get_yticklabels()[0].set_weight("bold")
    fig.text(0.01, 0.015, "Mixed test set: maps 1–10, each with its own random noise level. "
             "Bar: mean over the 10 maps; line: spread (standard deviation) across them.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "mixed_set", 300)


def signed(x, dec):
    return ("+" if round(x, dec) > 0 else "") + fmt(x, dec)


def stats_grid(tests, key, out):
    """Per set: difference, 95% CI and p, as a θmax x misindexing grid."""
    stem, title, unit, dec, levels, _ = MEASURES[key]
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["font.family"] = "Liberation Sans"
    cw, rh, x0, y0 = 2.45, 0.62, 1.0, 1.3                     # inches; y grows downward
    W, H = x0 + cw * len(levels) + 0.2, y0 + rh * (len(SCATTER) + 1) + 1.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    ax.text(0.15, 0.32, f"{title}: ours − closest MTEX filter ({unit}), maps 1–10 of each test set",
            fontsize=13, color=INK, va="center")
    ax.text(x0 + cw * len(levels) / 2, 0.75, "misindexed pixels p", fontsize=11, color=INK2,
            ha="center", va="center")
    ax.text(0.15, 1.05, "θmax", fontsize=12, color=INK, weight="bold", va="center")
    for j, mis in enumerate(levels):
        ax.text(x0 + (j + 0.5) * cw, 1.05, f"{mis}%", fontsize=12, color=INK, weight="bold",
                ha="center", va="center")

    def cell(x, y, w, t, one_line=False):
        st = status(t)
        ax.add_patch(Rectangle((x + 0.04, y + 0.04), w - 0.08, rh - 0.08, lw=0,
                               color={"better": GOOD, "ns": NS, "worse": BAD}[st]))
        mark = "†" if st == "worse" else stars(t["p_holm"]) if st == "better" else ""
        diff = f"{signed(t['diff'], dec)}{mark}"
        ci = f"[{fmt(t['lo'], dec)}, {fmt(t['hi'], dec)}]"
        p = f"p {fmt_p(t['p_holm']).replace('<', '< ') if fmt_p(t['p_holm']).startswith('<') else '= ' + fmt_p(t['p_holm'])}"
        cx = x + w / 2
        if one_line:
            ax.text(cx, y + rh / 2, f"{diff}   {ci}   {p}", fontsize=12, ha="center",
                    va="center", color=INK)
            return
        ax.text(cx, y + 0.2, diff, fontsize=12.5, ha="center", va="center", color=INK,
                weight="bold" if st != "ns" else "normal")
        ax.text(cx, y + 0.44, f"{ci}   {p}", fontsize=9.5, ha="center", va="center", color=INK2)

    for i, theta in enumerate(SCATTER):
        y = y0 + i * rh
        ax.text(0.15, y + rh / 2, f"{theta}°", fontsize=12, color=INK, va="center")
        for j, mis in enumerate(levels):
            cell(x0 + j * cw, y, cw, tests[set_name(theta, mis)])
    y = y0 + len(SCATTER) * rh + 0.12
    ax.text(0.15, y + rh / 2, "Mixed", fontsize=12, color=INK, va="center")
    cell(x0, y, cw * len(levels), tests["test_random"], one_line=True)

    y += rh + 0.3
    x = 0.15
    for tint, text in [(GOOD, "ours significantly better"), (NS, "no significant difference"),
                       (BAD, "ours significantly worse (†)")]:
        if tint == BAD and not any(status(t) == "worse" for t in tests.values()):
            continue
        ax.add_patch(Rectangle((x, y - 0.1), 0.3, 0.2, color=tint, lw=0))
        ax.text(x + 0.4, y, text, fontsize=10, color=INK2, va="center")
        x += 0.55 + 0.085 * len(text)
    ax.text(0.15, y + 0.5, "Top: mean difference (negative = ours better).  Bottom: 95% CI and "
            "p (paired t-test over the 10 maps, Holm-corrected).\n* p < 0.05   ** p < 0.01   "
            "*** p < 0.001", fontsize=9, color=MUTED, va="center", linespacing=1.5)
    save(fig, out, f"stats_{stem}", 200)


def compress(sets):
    """'1°, 3°, 5° (0%); mixed' from a list of set names."""
    groups = {}
    for s in sets:
        if s == "test_random":
            continue
        theta, _, p = s.removeprefix("scatter").partition("_mis")
        groups.setdefault(int(p or 0), []).append(int(theta))
    parts = [", ".join(f"{t}°" for t in sorted(th)) + f" ({p}%)" for p, th in sorted(groups.items())]
    return "; ".join(parts + (["mixed"] if "test_random" in sets else [])) or "–"


def stats_summary(all_tests, out):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["font.family"] = "Liberation Sans"
    head = ["Measure", "Ours lower", "Significant", "Ours − best MTEX\n(range over sets)",
            "Mixed set\nΔ [95% CI]", "p\n(mixed)", "Not significant / worse"]
    xs = [0.15, 3.35, 4.75, 7.15, 9.95, 10.95, 11.3]          # right edges, except text columns
    align = ["left", "right", "right", "right", "right", "right", "left"]
    W, rh = 15.4, 0.55
    H = 1.0 + rh * len(all_tests) + 0.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    for x, h, a in zip(xs, head, align):
        ax.text(x, 0.45, h, fontsize=12, weight="bold", color=INK, ha=a, va="center",
                linespacing=1.2)
    ax.plot([0.1, W - 0.1], [0.85, 0.85], color=INK, lw=1.0)
    y = 0.85
    for key, tests in all_tests.items():
        _, title, unit, dec, _, _ = MEASURES[key]
        grid = {s: t for s, t in tests.items() if s != "test_random"}
        n = len(tests)
        lower = sum(t["diff"] < 0 for t in tests.values())
        sig = sum(status(t) == "better" for t in tests.values())
        worse = [s for s, t in tests.items() if status(t) == "worse"]
        ns = [s for s, t in tests.items() if status(t) == "ns"]
        d = [t["diff"] for t in grid.values()]
        mix = tests["test_random"]
        mark = stars(mix["p_holm"]) if status(mix) == "better" else ""
        exceptions = compress(ns) + (f";  worse: {compress(worse)}" if worse else "")
        cells = [f"{title} ({unit})", f"{lower} / {n}", f"{sig} / {n}",
                 f"{signed(max(d), dec)} to {signed(min(d), dec)}",
                 f"{fmt(mix['diff'], dec)}{mark} [{fmt(mix['lo'], dec)}, {fmt(mix['hi'], dec)}]",
                 fmt_p(mix["p_holm"]), exceptions]
        y += rh
        for i, (x, c, a) in enumerate(zip(xs, cells, align)):
            ax.text(x, y - rh / 2, c, fontsize=12, color=INK, ha=a, va="center",
                    weight="bold" if i == 0 else "normal")
    ax.plot([0.1, W - 0.1], [y + 0.08, y + 0.08], color=INK, lw=1.0)
    ax.text(0.15, y + 0.4, "Maps 1–10 of each test set (29 sets; 22 with misindexing for CORRUPTED). "
            "Each measure against its own closest MTEX filter; negative = ours better.\n"
            "Significant: paired t-test over the 10 maps, Holm-corrected, p < 0.05.   "
            "* p < 0.05   ** p < 0.01   *** p < 0.001", fontsize=9.5, color=MUTED, va="center",
            linespacing=1.5)
    save(fig, out, "stats_summary", 200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "figures/slides"))
    args = p.parse_args()
    out = Path(args.out)

    vals = load()
    all_tests = {key: test(vals, key) for key in MEASURES}
    for key in MEASURES:
        curves(vals, key, out)
        stats_grid(all_tests[key], key, out)
    mixed(vals, out)
    stats_summary(all_tests, out)
    print(f"wrote curves_*, stats_*, mixed_set (.png/.pdf) to {out}")


if __name__ == "__main__":
    main()
