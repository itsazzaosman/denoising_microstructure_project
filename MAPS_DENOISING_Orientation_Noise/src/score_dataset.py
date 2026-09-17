#!/usr/bin/env python3
"""
Score denoised orientation maps over a whole dataset, not just one map.

score_filter_denoising.py answers "how did each method do on map 1?". This
answers "how does each method do across 500 maps?", which is the number a
baseline claim actually rests on. Single-map medians move around by a few
percent from map to map, so a difference between two methods on one map is not
evidence of anything.

Reported per method:

    ALL        median disorientation over every pixel
    CORRUPTED  median over the pixels the noise generator actually replaced
    BOUNDARY   median over pixels touching a grain boundary
    >10 deg    share of pixels still grossly wrong

Each is computed per map first, then aggregated across maps, so every map
carries equal weight regardless of how many pixels it has. The spread shown is
the interquartile range across maps: if two methods differ by less than that,
you have not separated them.

By default only maps where EVERY method produced an output are scored, so the
columns are directly comparable. --all-maps relaxes that and reports the per
method map count instead.

Usage
-----
    python score_dataset.py \
        --clean-dir datasets/clean_euler \
        --noisy-dir datasets/noisy_mis05 \
        --results   mtex_out/ \
        --shape 128 128

    # a subset, and a machine-readable dump
    python score_dataset.py ... --limit 50 --csv scores.csv
"""

import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orientation import (boundary_mask, disorientation_deg, euler_to_quat,
                         load_euler)

MAP_ID = re.compile(r"map_(\d+)")


def map_id(path):
    """The numeric map id embedded in a filename, or None."""
    m = MAP_ID.search(Path(path).name)
    return int(m.group(1)) if m else None


def index_dataset(clean_dir, noisy_dir, results_dir):
    """
    Pair up clean / noisy / mask / per-method result files by map id.

    Pairing is by map id rather than by exact stem because the stems drift:
    a noisy .txt is map_00001_clean_euler_noisy.txt but the MTEX outputs are
    named after the .ang, giving map_00001_clean_euler_noisy_noisy__method.txt.
    """
    clean = {map_id(f): f for f in Path(clean_dir).glob("map_*.txt") if map_id(f)}

    results = defaultdict(dict)
    for f in Path(results_dir).glob("*__*.txt"):
        mid = map_id(f)
        if mid is not None:
            results[mid][f.stem.split("__")[-1]] = f

    jobs = []
    for f in sorted(Path(noisy_dir).glob("map_*_noisy.txt"), key=lambda p: map_id(p) or 0):
        mid = map_id(f)
        if mid is None or mid not in clean:
            continue
        mask = Path(str(f) + ".badmask.npy")
        jobs.append({
            "id": mid,
            "clean": clean[mid],
            "noisy": f,
            "mask": mask if mask.exists() else None,
            "results": results.get(mid, {}),
        })
    return jobs


def score_one(job, shape):
    """All metrics for one map. Returns {method: {metric: value}} or None."""
    n = int(np.prod(shape))
    try:
        e_clean = load_euler(job["clean"])
        if e_clean.shape[0] != n:
            return None
        q_clean = euler_to_quat(e_clean)
    except Exception as exc:
        print(f"  [skip] map {job['id']}: clean unreadable ({exc})", flush=True)
        return None

    bad = np.load(job["mask"]).ravel().astype(bool) if job["mask"] else None
    bnd = boundary_mask(q_clean, shape)

    def metrics(q_test):
        d = disorientation_deg(q_clean, q_test)
        return {
            "all": float(np.median(d)),
            "bad": float(np.median(d[bad])) if bad is not None and bad.any() else np.nan,
            "bnd": float(np.median(d[bnd])) if bnd.any() else np.nan,
            "gross": float(100.0 * (d > 10.0).mean()),
        }

    out = {}
    try:
        out["noisy (no filter)"] = metrics(euler_to_quat(load_euler(job["noisy"])))
    except Exception as exc:
        print(f"  [skip] map {job['id']}: noisy unreadable ({exc})", flush=True)
        return None

    for method, path in job["results"].items():
        try:
            e = load_euler(path)
            if e.shape[0] != n:
                continue
            out[method] = metrics(euler_to_quat(e))
        except Exception as exc:
            print(f"  [skip] map {job['id']} {method}: {exc}", flush=True)
    return out


def _worker(args):
    job, shape = args
    return job["id"], score_one(job, shape)


def aggregate(per_map, methods):
    """Median and IQR across maps, per method per metric."""
    rows = {}
    for method in methods:
        vals = defaultdict(list)
        for scores in per_map.values():
            if method in scores:
                for k, v in scores[method].items():
                    if not np.isnan(v):
                        vals[k].append(v)
        row = {"n": max((len(v) for v in vals.values()), default=0)}
        for k, v in vals.items():
            a = np.asarray(v)
            row[k] = float(np.median(a))
            row[k + "_iqr"] = float(np.percentile(a, 75) - np.percentile(a, 25))
        rows[method] = row
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--noisy-dir", required=True)
    p.add_argument("--results", required=True,
                   help="folder of denoised .txt files named <stem>__<method>.txt")
    p.add_argument("--shape", nargs=2, type=int, required=True, metavar=("NY", "NX"))
    p.add_argument("--limit", type=int, default=0, help="score only the first N maps")
    p.add_argument("--jobs", type=int, default=8, help="parallel worker processes")
    p.add_argument("--all-maps", action="store_true",
                   help="score every map even where some methods are missing "
                        "(default: only maps where all methods ran)")
    p.add_argument("--csv", default=None, help="also write per-map scores here")
    args = p.parse_args()

    shape = tuple(args.shape)
    jobs = index_dataset(args.clean_dir, args.noisy_dir, args.results)
    if not jobs:
        sys.exit("no clean/noisy pairs found -- check --clean-dir and --noisy-dir")

    all_methods = sorted({m for j in jobs for m in j["results"]})
    if not all_methods:
        sys.exit(f"no *__*.txt result files in {args.results}")

    if not args.all_maps:
        complete = [j for j in jobs if len(j["results"]) == len(all_methods)]
        dropped = len(jobs) - len(complete)
        if dropped:
            print(f"note: {dropped} of {len(jobs)} maps lack some methods and are "
                  f"excluded; pass --all-maps to include them")
        jobs = complete
        if not jobs:
            sys.exit("no map has results from every method -- rerun with --all-maps")

    if args.limit:
        jobs = jobs[:args.limit]

    n_mask = sum(j["mask"] is not None for j in jobs)
    print(f"maps       : {len(jobs)}  (ids {jobs[0]['id']}-{jobs[-1]['id']})")
    print(f"methods    : {len(all_methods)}  ({', '.join(all_methods)})")
    print(f"badmasks   : {n_mask}/{len(jobs)}"
          + ("" if n_mask == len(jobs) else "   <- CORRUPTED column is partial"))
    print(f"scoring with {args.jobs} workers...", flush=True)

    per_map = {}
    payload = [(j, shape) for j in jobs]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for i, (mid, scores) in enumerate(pool.map(_worker, payload, chunksize=4), 1):
                if scores:
                    per_map[mid] = scores
                if i % 50 == 0 or i == len(payload):
                    print(f"  {i}/{len(payload)}", flush=True)
    else:
        for i, item in enumerate(payload, 1):
            mid, scores = _worker(item)
            if scores:
                per_map[mid] = scores
            if i % 50 == 0 or i == len(payload):
                print(f"  {i}/{len(payload)}", flush=True)

    if not per_map:
        sys.exit("every map failed to score")

    methods = ["noisy (no filter)"] + all_methods
    rows = aggregate(per_map, methods)

    print(f"\nscored {len(per_map)} maps")
    print("=" * 96)
    print(f"{'method':<20}{'ALL':>16}{'CORRUPTED':>16}{'BOUNDARY':>16}{'>10 deg':>14}{'maps':>7}")
    print(f"{'':<20}{'med (IQR) deg':>16}{'med (IQR) deg':>16}{'med (IQR) deg':>16}{'med %':>14}{'':>7}")
    print("-" * 96)
    for m in methods:
        r = rows.get(m, {})
        if not r.get("n"):
            continue
        cell = lambda k, w=16: f"{r[k]:.3f} ({r[k+'_iqr']:.3f})".rjust(w) if k in r else "-".rjust(w)
        print(f"{m:<20}{cell('all')}{cell('bad')}{cell('bnd')}"
              f"{r.get('gross', float('nan')):>14.2f}{r['n']:>7}")
    print("=" * 96)

    print("\nHow to read this")
    print("  every figure is the median ACROSS MAPS of that map's own median")
    print("  (IQR) is the spread across maps -- a gap smaller than this is not a result")
    print("  ALL       flattered by the 95% of pixels that were only lightly perturbed")
    print("  CORRUPTED did the method actually repair the pixels that were replaced?")
    print("  BOUNDARY  filters smear here; watch this when a method wins on ALL")
    print("  >10 deg   share of pixels still grossly wrong -- the headline number")

    ranked = [(m, rows[m]) for m in all_methods if rows.get(m, {}).get("n")]
    if ranked:
        base = rows["noisy (no filter)"]
        best_gross = min(ranked, key=lambda kv: kv[1]["gross"])
        best_all = min(ranked, key=lambda kv: kv[1]["all"])
        print(f"\n  fewest gross errors: {best_gross[0]} "
              f"({base['gross']:.2f}% -> {best_gross[1]['gross']:.2f}%)")
        print(f"  best on ALL        : {best_all[0]} "
              f"({base['all']:.3f} -> {best_all[1]['all']:.3f} deg, "
              f"boundary {best_all[1].get('bnd', float('nan')):.3f} deg)")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("map,method,all_med,bad_med,bnd_med,gross_pct\n")
            for mid in sorted(per_map):
                for method, s in sorted(per_map[mid].items()):
                    f.write(f"{mid},{method},{s['all']:.6f},{s['bad']:.6f},"
                            f"{s['bnd']:.6f},{s['gross']:.4f}\n")
        print(f"\nwrote {args.csv}  ({len(per_map)} maps x {len(methods)} methods)")


if __name__ == "__main__":
    main()
