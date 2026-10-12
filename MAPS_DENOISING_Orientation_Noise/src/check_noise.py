#!/usr/bin/env python3
"""
Measure the noise the blind model will be trained and validated on.

Draws a sample of training maps through TrainMaps exactly as training does
(random level per map), reads the saved validation set, and reports what the
noise actually looks like on the pixels, not just the settings asked for:

  scatter max     2 x the median error of the non-misindexed pixels (the tilt
                  angle is uniform in [0, max], so its median is max / 2)
  misindexed %    share of pixels replaced
  boundary odds   how much more likely a boundary pixel is to be misindexed
                  than an interior one; should come out near --boundary-bias
  ALL, >10 deg    the same input statistics the scorers report

Writes figures/noise_check/summary.csv, per_map.csv and noise_check.png.

Usage
-----
    python check_noise.py --n-train 400
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import DirMaps, TrainMaps
from orientation import boundary_mask, disorientation_deg
from plot_maps import ipf_z_color

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (128, 128)


def measure(q_noisy, q_clean, bad):
    """Per-map noise statistics from (N, 4) noisy/clean quats and an (N,) mask."""
    d = disorientation_deg(q_noisy, q_clean)
    bnd = boundary_mask(q_clean, SHAPE)
    p_bad_bnd = bad[bnd].mean() if bnd.any() else np.nan
    p_bad_int = bad[~bnd].mean() if (~bnd).any() else np.nan
    return {
        "scatter_max_est": 2.0 * float(np.median(d[~bad])),
        "misindexed_pct": 100.0 * float(bad.mean()),
        # P(bad | boundary) / P(bad | interior): the boundary bias as realised.
        "boundary_ratio": (float(p_bad_bnd / p_bad_int)
                           if bad.any() and p_bad_int > 0 else np.nan),
        "all_med": float(np.median(d)),
        "gross_pct": 100.0 * float((d > 10).mean()),
    }


def as_numpy(sample):
    q_noisy, q_clean, bad = (t.numpy() for t in sample[:3])
    return (q_noisy.reshape(-1, 4).astype(np.float64),
            q_clean.reshape(-1, 4).astype(np.float64), bad.reshape(-1))


def summarise(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    return {"min": v.min(), "p25": np.percentile(v, 25), "median": np.median(v),
            "p75": np.percentile(v, 75), "max": v.max(), "mean": v.mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=400, help="training maps to sample")
    p.add_argument("--scatter", type=float, default=9.0)
    p.add_argument("--misindex", type=float, default=0.10)
    p.add_argument("--boundary-bias", type=float, default=4.0)
    p.add_argument("--val-clean-dir", default=str(ROOT / "datasets/splits/val/clean_euler"))
    p.add_argument("--val-noisy-dir", default=str(ROOT / "datasets/splits/val/noise_random"))
    p.add_argument("--out", default=str(ROOT / "figures/noise_check"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    noise = dict(scatter_deg=args.scatter, misindex=args.misindex,
                 boundary_bias=args.boundary_bias, random=True)

    # fixed_noise gives every map a seeded draw, so this check is repeatable.
    # Training itself draws unseeded, fresh noise every epoch from the same ranges.
    train = TrainMaps(split="train", val_fraction=0, augment=False, noise=noise,
                      fixed_noise=True)
    pick = np.linspace(0, len(train) - 1, args.n_train).astype(int)
    rows, examples = [], []
    for k in pick:
        qn, qc, bad = as_numpy(train[k])
        rows.append({"split": "train", "map": int(train.ids[k]), **measure(qn, qc, bad)})
        examples.append((rows[-1], qn, qc, bad))

    val = DirMaps(args.val_clean_dir, args.val_noisy_dir)
    for k in range(len(val)):
        qn, qc, bad = as_numpy(val[k])
        rows.append({"split": "val", "map": int(val.ids[k]), **measure(qn, qc, bad)})

    keys = ["scatter_max_est", "misindexed_pct", "boundary_ratio", "all_med", "gross_pct"]
    with open(out / "per_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "map", *keys])
        w.writeheader()
        w.writerows(rows)

    print(f"train sample: {sum(r['split'] == 'train' for r in rows)} maps   "
          f"val: {sum(r['split'] == 'val' for r in rows)} maps\n")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "quantity", "min", "p25", "median", "p75", "max", "mean"])
        print(f"{'':<6}{'quantity':<18}{'min':>8}{'p25':>8}{'median':>8}{'p75':>8}{'max':>8}{'mean':>8}")
        for split in ("train", "val"):
            sub = [r for r in rows if r["split"] == split]
            for key in keys:
                s = summarise(sub, key)
                w.writerow([split, key, *(f"{s[c]:.4f}" for c in s)])
                print(f"{split:<6}{key:<18}" + "".join(f"{s[c]:>8.2f}" for c in s))
            print()

    # ---- figure: level distributions + example maps from low to high noise ----
    fig = plt.figure(figsize=(13, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for col, (key, label, top) in enumerate([("scatter_max_est", "scatter max (deg)", args.scatter),
                                             ("misindexed_pct", "misindexed pixels (%)", 100 * args.misindex)]):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        bins = np.linspace(0, top * 1.05, 22)
        for split, colour in (("train", "#4C72B0"), ("val", "#DD8452")):
            v = [r[key] for r in rows if r["split"] == split]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=colour,
                    label=f"{split} (n={len(v)})")
        ax.set_xlabel(label); ax.set_ylabel("density"); ax.legend(frameon=False)
        ax.set_title(f"{label.split(' (')[0]}: drawn uniformly from 0 to {top:g}")

    examples.sort(key=lambda e: e[0]["scatter_max_est"] + 50 * e[0]["misindexed_pct"] / 100)
    chosen = [examples[int(i)] for i in np.linspace(0, len(examples) - 1, 4)]
    for col, (r, qn, qc, bad) in enumerate(chosen):
        for row, (img, title) in enumerate([
                (ipf_z_color(qc).reshape(*SHAPE, 3), f"map {r['map']}: clean"),
                (ipf_z_color(qn).reshape(*SHAPE, 3),
                 f"noisy: scatter max {r['scatter_max_est']:.1f} deg,\n"
                 f"{r['misindexed_pct']:.1f}% misindexed"),
                (bad.reshape(SHAPE), "misindexed pixels (white)")], start=1):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
            ax.set_title(title, fontsize=9); ax.axis("off")
    fig.savefig(out / "noise_check.png", dpi=130)
    print(f"wrote {out}/noise_check.png, summary.csv, per_map.csv")


if __name__ == "__main__":
    main()
