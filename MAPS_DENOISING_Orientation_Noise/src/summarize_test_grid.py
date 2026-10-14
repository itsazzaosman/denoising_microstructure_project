#!/usr/bin/env python3
"""
Measure the input noise of every test set in configs/test_grid.txt.

Per set, over all its maps (per-map mean, then the mean across maps, as
score_dataset.py does):

  ALL         mean disorientation of the noisy input to the truth, deg
  INTERIOR    the same over pixels that were not misindexed: the scatter alone
  BOUNDARY    the same over grain-boundary pixels
  misindexed  share of pixels replaced, %
  >10 deg     share of pixels grossly wrong, %

This is the "noisy (no filter)" row of every score table, gathered into one
table that describes the test grid in the paper. Writes
figures/test_grid/input_noise.csv.

Usage
-----
    python summarize_test_grid.py
"""

import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from orientation import boundary_mask, disorientation_deg, euler_to_quat, load_euler
from paths import CLEAN_DIR, noise_dir

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (128, 128)
MAP_ID = re.compile(r"map_(\d+)")


def one_map(args):
    clean_path, noisy_path = args
    qc = euler_to_quat(load_euler(clean_path))
    qn = euler_to_quat(load_euler(noisy_path))
    mask_path = Path(str(noisy_path) + ".badmask.npy")
    bad = (np.load(mask_path).ravel() if mask_path.exists()
           else np.zeros(len(qc), dtype=bool))
    d = disorientation_deg(qn, qc)
    bnd = boundary_mask(qc, SHAPE)
    return (d.mean(), d[~bad].mean(), d[bnd].mean(),
            100 * bad.mean(), 100 * (d > 10).mean())


def main():
    names = (ROOT / "configs/test_grid.txt").read_text().split()
    out = ROOT / "figures/test_grid"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name in names:
            cfg = yaml.safe_load((ROOT / f"configs/noise/{name}.yaml").read_text())["noise"]
            noisy = sorted(noise_dir(name).glob("map_*_noisy.txt"))
            ids = [MAP_ID.search(f.name).group(1) for f in noisy]
            jobs = [(CLEAN_DIR / f"map_{i}_clean_euler.txt", f)
                    for i, f in zip(ids, noisy)]
            stats = np.array(list(pool.map(one_map, jobs, chunksize=8)))
            avg = stats.mean(axis=0)
            rows.append({"set": name, "scatter_max_deg": cfg["scatter"],
                         "misindex_pct": round(100 * cfg["misindex"]), "maps": len(noisy),
                         "all": avg[0], "interior": avg[1], "boundary": avg[2],
                         "misindexed_pct": avg[3], "gross_pct": avg[4]})
            r = rows[-1]
            print(f"{name:<16} {r['maps']:>4} maps   ALL {r['all']:6.3f}  INTERIOR {r['interior']:6.3f}  "
                  f"BOUNDARY {r['boundary']:6.3f}  misindexed {r['misindexed_pct']:5.2f}%  "
                  f">10deg {r['gross_pct']:5.2f}%", flush=True)

    with open(out / "input_noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {out / 'input_noise.csv'}")


if __name__ == "__main__":
    main()
