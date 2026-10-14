#!/usr/bin/env python3
"""
Compare the network with the MTEX filters on the maps MTEX was run on.

For every test set, only the maps that have an output from every MTEX filter
AND from every network run are used, so all methods are scored on exactly the
same maps. Per map: mean per-pixel disorientation (ALL), the same over
grain-boundary pixels (BOUNDARY) and misindexed pixels (CORRUPTED), and the
percentage of pixels more than 10 deg wrong. The network value is the mean of
its runs.

Across the test sets, a paired t-test compares the network with each filter on
the per-set mean errors (n = number of sets), so even one map per set gives a
test over the 28 conditions.

Writes figures/mtex_comparison/per_map.csv, summary.csv and
mtex_comparison.png/.pdf (error vs scatter, one panel per misindexing level).

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

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (128, 128)
FILTERS = ["spline", "median", "clean-spline", "clean-median"]
RUNS = ["blind_learned_seed1", "blind_learned_seed2", "blind_learned_seed3"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLOURS = {"ours": "#2a78d6", "spline": "#eb6834", "median": "#1baf7a",
           "clean-spline": "#eda100", "clean-median": "#e87ba4"}


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
    s, i = args
    stem = f"map_{i:05d}_clean_euler_noisy"
    qc = euler_to_quat(load_euler(ROOT / f"datasets/clean_euler/map_{i:05d}_clean_euler.txt"))
    f_noisy = ROOT / f"datasets/noise_{s}/{stem}.txt"
    mask = Path(str(f_noisy) + ".badmask.npy")
    bad = np.load(mask).ravel() if mask.exists() else np.zeros(len(qc), bool)
    bnd = boundary_mask(qc, SHAPE)
    out = {"input": measures(euler_to_quat(load_euler(f_noisy)), qc, bad, bnd)}
    res = ROOT / f"datasets/mtex_out_{s}"
    runs = [measures(euler_to_quat(load_euler(res / f"{stem}_noisy__{r}.txt")), qc, bad, bnd) for r in RUNS]
    out["ours"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for f in FILTERS:
        out[f] = measures(euler_to_quat(load_euler(res / f"{stem}_noisy__{f}.txt")), qc, bad, bnd)
    return s, i, out


def common_maps(s):
    res = ROOT / f"datasets/mtex_out_{s}"
    have = lambda name: {int(m.group(1)) for p in res.glob(f"map_*__{name}.txt")
                         if (m := re.match(r"map_(\d+)", p.name))}
    maps = set.intersection(*[have(n) for n in FILTERS + RUNS]) if res.exists() else set()
    return sorted(maps)


def level(s):
    m = re.match(r"scatter(\d+)(?:_mis(\d+))?$", s)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--out", default=str(ROOT / "figures/mtex_comparison"))
    args = p.parse_args()
    sets = args.sets or (ROOT / "configs/test_grid.txt").read_text().split() + ["test_random"]

    jobs, skipped = [], []
    for s in sets:
        maps = common_maps(s)
        if maps:
            jobs.extend((s, i) for i in maps)
        else:
            skipped.append(s)
    if not jobs:
        raise SystemExit("no map has outputs from every MTEX filter and every run yet")
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(score_map, jobs, chunksize=4))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = ["input", "ours"] + FILTERS
    with open(out / "per_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "map", "method", "all_mean", "bnd_mean", "bad_mean", "gross_pct", "unfilled_px"])
        for s, i, r in results:
            for m in methods:
                w.writerow([s, i, m] + [f"{r[m][k]:.5f}" for k in ("all", "bnd", "bad", "gross")] + [r[m]["unfilled"]])

    per_set = defaultdict(lambda: defaultdict(list))
    for s, i, r in results:
        for m in methods:
            per_set[s][m].append(r[m])
    summary = {}
    for s in sets:
        if s not in per_set:
            continue
        summary[s] = {m: {k: nanmean([x[k] for x in per_set[s][m]]) for k in ("all", "bnd", "bad", "gross")}
                      for m in methods}
        summary[s]["n"] = len(per_set[s]["ours"])
        summary[s]["unfilled"] = {f: int(sum(x["unfilled"] for x in per_set[s][f])) for f in FILTERS}

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "maps", "measure"] + methods)
        for s, row in summary.items():
            for k in ("all", "bnd", "bad", "gross"):
                w.writerow([s, row["n"], k] + [f"{row[m][k]:.4f}" for m in methods])

    print(f"methods scored on the same maps; ours = mean of {len(RUNS)} runs. Mean per-pixel error ALL (deg):\n")
    print(f"{'set':<16}{'maps':>5}{'input':>8}{'ours':>8}" + "".join(f"{f:>14}" for f in FILTERS))
    for s, row in summary.items():
        best = min(FILTERS, key=lambda f: row[f]["all"])
        print(f"{s:<16}{row['n']:>5}{row['input']['all']:>8.3f}{row['ours']['all']:>8.3f}"
              + "".join(f"{row[f]['all']:>13.3f}{'*' if f == best else ' '}" for f in FILTERS))
    print("  * best MTEX filter for that set")
    holes = {}
    for s, row in summary.items():
        unfilled = {f: n for f, n in row["unfilled"].items() if n}
        if unfilled:
            holes[s] = unfilled
    if holes:
        print(f"\nWARNING: MTEX left pixels unfilled (ignored in its scores): {holes}")
    if skipped:
        print(f"\nno complete MTEX outputs yet for: {', '.join(skipped)}")

    grid = [s for s in summary if level(s)[0] is not None]
    if len(grid) >= 3:
        print(f"\nPaired t-test across {len(grid)} test sets, on the per-set ALL errors (ours - filter):")
        for f in FILTERS:
            a = np.array([summary[s]["ours"]["all"] for s in grid])
            b = np.array([summary[s][f]["all"] for s in grid])
            t, pv = stats.ttest_rel(a, b)
            print(f"  vs {f:<13} mean difference {np.mean(a - b):+.3f} deg, ours lower in "
                  f"{int((a < b).sum())}/{len(grid)} sets, t = {t:.2f}, p = {pv:.2g}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
    for ax, mis in zip(axes, (0, 3, 6, 9)):
        sub = sorted((level(s)[0], s) for s in grid if level(s)[1] == mis)
        if not sub:
            ax.set_visible(False); continue
        x = [t for t, _ in sub]
        ax.plot(x, [summary[s]["input"]["all"] for _, s in sub], color=MUTED, ls="--", lw=1.2, label="noisy input")
        for m in ["ours"] + FILTERS:
            ax.plot(x, [summary[s][m]["all"] for _, s in sub], color=COLOURS[m], lw=2, marker="o", ms=4,
                    label="ours (3 runs)" if m == "ours" else m)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([1, 3, 5, 7, 9, 12, 15])
        ax.set_xlabel(r"maximum scatter angle $\theta_{\max}$ (deg)")
        ax.set_title(f"misindexed pixels p = {mis}%", loc="left")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("mean per-pixel error (deg)")
    axes[0].legend(frameon=False, fontsize=8)
    n_maps = sorted({summary[s]["n"] for s in grid})
    fig.suptitle(f"Network vs MTEX filters on the same test maps ({'/'.join(map(str, n_maps))} per set)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "mtex_comparison.png", dpi=150)
    fig.savefig(out / "mtex_comparison.pdf")
    print(f"\nwrote {out}/per_map.csv, summary.csv, mtex_comparison.png/.pdf")


if __name__ == "__main__":
    main()
