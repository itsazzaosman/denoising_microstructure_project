#!/usr/bin/env python3
"""
One results table and one figure for a trained model over every test set.

Reads figures/<set>/scores_mean.csv (written by score_dataset.py / `make
score-grid`) for the 28 grid sets and test_random. For each set and each run
it takes the mean over the 500 test maps; across the runs (the seeds) it
reports the mean and the standard deviation, which is the run-to-run spread.

Writes
  figures/results/<prefix>_results.csv   one row per test set
  figures/results/<prefix>_results.png   error vs scatter level, one line per
                                         misindexing level, input dashed

Usage
-----
    python summarize_results.py --runs blind_learned_seed1 blind_learned_seed2 blind_learned_seed3
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
METRICS = [("all_mean", "ALL"), ("bnd_mean", "BOUNDARY"), ("bad_mean", "CORRUPTED"), ("gross_pct", ">10 deg %")]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Misindexing level is ordinal, so one hue from light to dark (palette blue ramp).
RAMP = {0: "#86b6ef", 3: "#3987e5", 6: "#1c5cab", 9: "#0d366b"}


def per_method_means(path):
    """{method: {metric: mean over maps}} from one per-map score CSV."""
    vals = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            for key, _ in METRICS:
                v = float(r[key])
                if np.isfinite(v):
                    vals[r["method"]][key].append(v)
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in vals.items()}, \
           {m: len(d["all_mean"]) for m, d in vals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="method names of the runs (seeds)")
    p.add_argument("--prefix", default="blind_learned")
    args = p.parse_args()

    sets = (ROOT / "configs/test_grid.txt").read_text().split() + ["test_random"]
    out = ROOT / "figures/results"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in sets:
        means, counts = per_method_means(ROOT / f"figures/{s}/scores_mean.csv")
        missing = [r for r in args.runs if counts.get(r, 0) < 500]
        if missing:
            raise SystemExit(f"{s}: runs {missing} have fewer than 500 scored maps")
        noise = yaml.safe_load((ROOT / f"configs/noise/{s}.yaml").read_text())["noise"]
        row = {"set": s,
               "scatter_max_deg": "0-15 (random)" if s == "test_random" else noise["scatter"],
               "misindex_pct": "0-10 (random)" if s == "test_random" else round(100 * noise["misindex"])}
        for key, _ in METRICS:
            row[f"input_{key}"] = means["noisy (no filter)"].get(key, np.nan)
            seed_vals = np.array([means[r].get(key, np.nan) for r in args.runs])
            row[f"model_{key}"] = float(np.mean(seed_vals))
            row[f"model_{key}_sd"] = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else 0.0
        row["removed_pct"] = 100 * (1 - row["model_all_mean"] / row["input_all_mean"])
        rows.append(row)

    with open(out / f"{args.prefix}_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{len(args.runs)} runs: {', '.join(args.runs)}   (model = mean over runs, +- sd across runs)\n")
    print(f"{'set':<16}{'ALL in':>8}{'ALL model':>17}{'removed':>9}{'BOUNDARY model':>19}"
          f"{'CORRUPTED in':>14}{'CORRUPTED model':>19}{'>10% in':>9}{'>10% model':>17}")
    for r in rows:
        pm = lambda k, d=3: f"{r['model_' + k]:.{d}f} ± {r['model_' + k + '_sd']:.{d}f}"
        corr_in = f"{r['input_bad_mean']:.2f}" if np.isfinite(r["input_bad_mean"]) else "-"
        corr_md = pm("bad_mean", 2) if np.isfinite(r["model_bad_mean"]) else "-"
        print(f"{r['set']:<16}{r['input_all_mean']:>8.3f}{pm('all_mean'):>17}{r['removed_pct']:>8.1f}%"
              f"{pm('bnd_mean'):>19}{corr_in:>14}{corr_md:>19}{r['input_gross_pct']:>9.2f}{pm('gross_pct'):>17}")

    # ---- figure: error vs scatter level, one line per misindexing level ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED})
    grid = [r for r in rows if r["set"] != "test_random"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [("all_mean", "ALL: mean error over all pixels (deg)", [0, 3, 6, 9]),
              ("bnd_mean", "BOUNDARY: mean error on grain-boundary pixels (deg)", [0, 3, 6, 9]),
              ("bad_mean", "CORRUPTED: mean error on misindexed pixels (deg)", [3, 6, 9])]
    for a, (key, title, levels) in zip(ax, panels):
        for m in levels:
            sub = sorted([r for r in grid if r["misindex_pct"] == m], key=lambda r: r["scatter_max_deg"])
            x = [r["scatter_max_deg"] for r in sub]
            a.plot(x, [r[f"input_{key}"] for r in sub], color=RAMP[m], lw=1.2, ls="--")
            a.errorbar(x, [r[f"model_{key}"] for r in sub], yerr=[r[f"model_{key}_sd"] for r in sub],
                       color=RAMP[m], lw=2, marker="o", ms=4, capsize=2, label=f"{m}% misindexed")
        a.set_yscale("log")
        a.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))     # 0.1 0.2 0.5 1 2 5 10 ...
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel(r"maximum scatter angle $\theta_{\max}$ (deg)")
        a.set_xticks([1, 3, 5, 7, 9, 12, 15])
        a.set_title(title, loc="left", fontsize=9.5)
        a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    ax[0].plot([], [], color=MUTED, ls="--", lw=1.2, label="noisy input")
    ax[0].plot([], [], color=MUTED, lw=2, marker="o", ms=4, label="denoised (mean ± sd, 3 seeds)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_results.png", dpi=150)
    fig.savefig(out / f"{args.prefix}_results.pdf")
    print(f"\nwrote {out / (args.prefix + '_results.csv')} and _results.png/.pdf")


if __name__ == "__main__":
    main()
