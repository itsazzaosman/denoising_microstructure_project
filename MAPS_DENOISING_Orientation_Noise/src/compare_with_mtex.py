#!/usr/bin/env python3
"""
Compare the network with the MTEX filters on the maps MTEX was run on.

For every test set, only the maps that have an output from every network run
and from every MTEX filter found for that set are used, so all methods are
scored on exactly the same maps. Per map: mean per-pixel disorientation (ALL),
the same over grain-boundary pixels (BOUNDARY) and misindexed pixels
(CORRUPTED), and the percentage of pixels more than 10 deg wrong. The network
value is the mean of its runs.

Each filter exists twice, as <filter> and clean-<filter> (pre-cleaned). The
table and figure give, per set, the better of the two: the comparison is the
best MTEX can do with that filter. summary.csv keeps both.

Across the test sets, a paired t-test compares the network with each filter
and with the best MTEX result of each set, on the per-set ALL errors (n =
number of sets), so even one map per set gives a test over the 28 conditions.

Writes <out>/per_map.csv, summary.csv and mtex_comparison.png/.pdf
(default <out> = figures/mtex_comparison).

Usage
-----
    python compare_with_mtex.py
    python compare_with_mtex.py --sets scatter5 scatter7     # a subset
"""

import argparse
import csv
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from orientation import boundary_mask, disorientation_deg, euler_to_quat, load_euler
from paths import MODEL_RESULTS, MTEX_RESULTS, clean_map, noise_dir

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (128, 128)
BASE = ["mean", "median", "kuwahara", "spline", "halfquad", "infconv"]
ALL_FILTERS = BASE + [f"clean-{f}" for f in BASE]
PRETTY = {"mean": "mean", "median": "median", "kuwahara": "Kuwahara", "spline": "spline",
          "halfquad": "half-quadratic", "infconv": "infimal conv."}
RUNS = ["blind_learned_seed1", "blind_learned_seed2", "blind_learned_seed3"]
KEYS = ("all", "bnd", "bad", "gross")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLOURS = {"ours": "#2a78d6", "mean": "#eb6834", "median": "#1baf7a", "kuwahara": "#eda100",
           "spline": "#e87ba4", "halfquad": "#008300", "infconv": "#6250d6"}


def measures(q, qc, bad, bnd):
    d = disorientation_deg(q, qc)
    ok = np.isfinite(d)                               # NaN = pixel MTEX left unfilled
    m = lambda sel: float(d[sel & ok].mean()) if (sel & ok).any() else np.nan
    return {"all": m(np.ones_like(ok)), "bnd": m(bnd), "bad": m(bad) if bad.any() else np.nan,
            "gross": float(100 * (d[ok] > 10).mean()), "unfilled": int((~ok).sum())}


def nanmean(values):
    """Mean ignoring NaN; NaN when nothing is left (no misindexed pixels in a set)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def score_map(args):
    s, i, filters = args
    stem = f"map_{i:05d}_clean_euler_noisy"
    qc = euler_to_quat(load_euler(clean_map(i)))
    f_noisy = noise_dir(s) / f"{stem}.txt"
    mask = Path(str(f_noisy) + ".badmask.npy")
    bad = np.load(mask).ravel() if mask.exists() else np.zeros(len(qc), bool)
    bnd = boundary_mask(qc, SHAPE)
    out = {"input": measures(euler_to_quat(load_euler(f_noisy)), qc, bad, bnd)}
    model, mtex = MODEL_RESULTS / s, MTEX_RESULTS / s
    runs = [measures(euler_to_quat(load_euler(model / f"{stem}_noisy__{r}.txt")), qc, bad, bnd) for r in RUNS]
    out["ours"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for f in filters:
        out[f] = measures(euler_to_quat(load_euler(mtex / f"{stem}_noisy__{f}.txt")), qc, bad, bnd)
    return s, i, out


def maps_and_filters(s):
    """Maps with every run and every MTEX filter present for this set."""
    model, mtex = MODEL_RESULTS / s, MTEX_RESULTS / s
    if not mtex.exists():
        return [], []
    have = lambda name: {int(m.group(1)) for d in (model, mtex) for p in d.glob(f"map_*__{name}.txt")
                         if (m := re.match(r"map_(\d+)", p.name))}
    filters = [f for f in ALL_FILTERS if have(f)]
    if not filters:
        return [], []
    return sorted(set.intersection(*[have(n) for n in filters + RUNS])), filters


def level(s):
    m = re.match(r"scatter(\d+)(?:_mis(\d+))?$", s)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--out", default=str(ROOT / "figures/mtex_comparison"))
    args = p.parse_args()
    sets = args.sets or (ROOT / "configs/test_grid.txt").read_text().split() + ["test_random"]

    jobs, skipped, filters_of = [], [], {}
    for s in sets:
        maps, filters = maps_and_filters(s)
        if maps:
            filters_of[s] = filters
            jobs.extend((s, i, filters) for i in maps)
        else:
            skipped.append(s)
    if not jobs:
        raise SystemExit("no map has outputs from the network runs and the MTEX filters yet")
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(score_map, jobs, chunksize=4))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "map", "method", "all_mean", "bnd_mean", "bad_mean", "gross_pct", "unfilled_px"])
        for s, i, r in results:
            for m, v in r.items():
                w.writerow([s, i, m] + [f"{v[k]:.5f}" for k in KEYS] + [v["unfilled"]])

    per_set = defaultdict(lambda: defaultdict(list))
    for s, i, r in results:
        for m, v in r.items():
            per_set[s][m].append(v)
    summary = {}
    for s in sets:
        if s not in per_set:
            continue
        row = {m: {k: nanmean([x[k] for x in vals]) for k in KEYS} for m, vals in per_set[s].items()}
        row["n"] = len(per_set[s]["ours"])
        row["unfilled"] = {f: int(sum(x["unfilled"] for x in per_set[s][f])) for f in filters_of[s]}
        # each filter: the better of plain and pre-cleaned, on ALL
        row["best"] = {}
        for b in BASE:
            cands = [f for f in (b, f"clean-{b}") if f in row]
            if cands:
                row["best"][b] = min(cands, key=lambda f: row[f]["all"])
        summary[s] = row

    methods = ["input", "ours"] + ALL_FILTERS
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "maps", "measure"] + methods)
        for s, row in summary.items():
            for k in KEYS:
                w.writerow([s, row["n"], k] + [f"{row[m][k]:.4f}" if m in row else "" for m in methods])

    bases = [b for b in BASE if any(b in row["best"] for row in summary.values())]
    print(f"Mean per-pixel error ALL (deg), all methods on the same maps; ours = mean of {len(RUNS)} runs.")
    print("Each filter: the better of plain and pre-cleaned (c = clean- version); * best MTEX result.\n")
    print(f"{'set':<16}{'maps':>5}{'input':>8}{'ours':>8}" + "".join(f"{PRETTY[b]:>16}" for b in bases))
    for s, row in summary.items():
        vals = {b: row[row["best"][b]]["all"] for b in bases if b in row["best"]}
        top = min(vals, key=vals.get)
        cells = []
        for b in bases:
            if b not in vals:
                cells.append(f"{'-':>16}")
                continue
            tag = ("c" if row["best"][b].startswith("clean-") else " ") + ("*" if b == top else " ")
            cells.append(f"{vals[b]:>14.3f}{tag}")
        print(f"{s:<16}{row['n']:>5}{row['input']['all']:>8.3f}{row['ours']['all']:>8.3f}" + "".join(cells))
    holes = {}
    for s, row in summary.items():
        unfilled = {f: n for f, n in row["unfilled"].items() if n}
        if unfilled:
            holes[s] = unfilled
    if holes:
        print(f"\nWARNING: MTEX left pixels unfilled (ignored in its scores): {holes}")
    if skipped:
        print(f"\nno MTEX outputs yet for: {', '.join(skipped)}")

    grid = [s for s in summary if level(s)[0] is not None]
    if len(grid) >= 3:
        a = np.array([summary[s]["ours"]["all"] for s in grid])
        print(f"\nPaired t-test across {len(grid)} test sets, on the per-set ALL errors (ours - MTEX):")
        competitors = {PRETTY[b]: [summary[s][summary[s]["best"][b]]["all"] for s in grid]
                       for b in bases if all(b in summary[s]["best"] for s in grid)}
        competitors["best MTEX per set"] = [min(summary[s][summary[s]["best"][b]]["all"]
                                                for b in summary[s]["best"]) for s in grid]
        for name, b in competitors.items():
            b = np.array(b)
            t, pv = stats.ttest_rel(a, b)
            print(f"  vs {name:<18} mean difference {np.mean(a - b):+.3f} deg, ours lower in "
                  f"{int((a < b).sum())}/{len(grid)} sets, t = {t:.2f}, p = {pv:.2g}")

    # ---- figure: error vs scatter, one panel per misindexing level ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, mis in zip(axes, (0, 3, 6, 9)):
        sub = sorted((level(s)[0], s) for s in grid if level(s)[1] == mis)
        if not sub:
            ax.set_visible(False)
            continue
        x = [t for t, _ in sub]
        ax.plot(x, [summary[s]["input"]["all"] for _, s in sub], color=MUTED, ls="--", lw=1.2, label="noisy input")
        for b in bases:
            y = [summary[s][summary[s]["best"][b]]["all"] if b in summary[s]["best"] else np.nan for _, s in sub]
            ax.plot(x, y, color=COLOURS[b], lw=1.3, marker="o", ms=3, label=PRETTY[b])
        ax.plot(x, [summary[s]["ours"]["all"] for _, s in sub], color=COLOURS["ours"], lw=2.4,
                marker="o", ms=5, label="ours (3 runs)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([1, 3, 5, 7, 9, 12, 15])
        ax.set_xlabel(r"maximum scatter angle $\theta_{\max}$ (deg)")
        ax.set_title(f"misindexed pixels p = {mis}%", loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("mean per-pixel error (deg)")
    axes[0].legend(frameon=False, fontsize=7.5)
    n_maps = sorted({summary[s]["n"] for s in grid})
    fig.suptitle(f"Network vs MTEX filters on the same test maps ({'/'.join(map(str, n_maps))} per set; "
                 "each filter with or without pre-cleaning, whichever is better)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "mtex_comparison.png", dpi=150)
    fig.savefig(out / "mtex_comparison.pdf")
    print(f"\nwrote {out}/per_map.csv, summary.csv, mtex_comparison.png/.pdf")


if __name__ == "__main__":
    main()
