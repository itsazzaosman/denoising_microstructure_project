#!/usr/bin/env python3
"""
Score denoised orientation maps against the clean truth.

Reports the disorientation delta three ways, because a single whole-map number
is misleading:

  ALL         every pixel. Flattering: if only 5% of pixels were corrupted,
              95% of this number comes from pixels that were never broken.
  CORRUPTED   only the pixels the noise actually damaged (needs the badmask).
              This is "did the method fix the damage?"
  BOUNDARY    only pixels next to a grain boundary. This is where filters
              smear and where twins live, so it is the number that predicts
              behaviour on the real LabDCT problem.

A method can look excellent on ALL and still be useless, by smoothing grain
interiors while leaving every misindexed pixel untouched.

Usage
-----
    python score_denoising.py --clean clean.txt --noisy noisy.txt \
        --mask noisy.txt.badmask.npy --results mtex_out/ --shape 128 128

The results folder is scanned for files named <something>__<method>.txt, which
is what run_mtex_baseline.m writes. The part after the double underscore is
used as the method name in the table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# orientation maths (cubic m-3m)
# --------------------------------------------------------------------------


def cubic_sym():
    h, r = 0.5, np.sqrt(2.0) / 2.0
    return np.array([
        (1, 0, 0, 0),
        (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
        (r, r, 0, 0), (r, -r, 0, 0), (r, 0, r, 0), (r, 0, -r, 0),
        (r, 0, 0, r), (r, 0, 0, -r),
        (0, r, r, 0), (0, r, -r, 0), (0, r, 0, r), (0, r, 0, -r),
        (0, 0, r, r), (0, 0, r, -r),
        (h, h, h, h), (h, -h, -h, -h), (h, h, -h, h), (h, -h, h, -h),
        (h, -h, h, h), (h, h, -h, -h), (h, -h, -h, h), (h, h, h, -h),
    ], dtype=np.float64)


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_to_quat(e):
    phi1, Phi, phi2 = e[:, 0], e[:, 1], e[:, 2]
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack([c * np.cos(sigma), -s * np.cos(delta),
                  -s * np.sin(delta), -c * np.sin(sigma)], axis=-1)
    q[q[:, 0] < 0] *= -1.0
    return q


def disorientation_deg(q1, q2, sym):
    """Minimised over all 24x24 cubic symmetry combinations."""
    m = qmul(q2, qconj(q1))
    best = np.zeros(m.shape[0])
    for si in sym:
        left = qmul(si[None, :], m)
        for sj in sym:
            np.maximum(best, np.abs(qmul(left, sj[None, :])[:, 0]), out=best)
    np.clip(best, -1.0, 1.0, out=best)
    return np.degrees(2.0 * np.arccos(best))


# --------------------------------------------------------------------------


def load_euler(path):
    with open(path) as f:
        first = f.readline()
    try:
        [float(t) for t in first.split()]
        skip = 0
    except ValueError:
        skip = 1
    d = np.loadtxt(path, skiprows=skip)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    return d[:, :3]


def boundary_mask(q_clean, shape):
    """True where a pixel touches a neighbour with a different clean orientation."""
    ny, nx = shape
    g = q_clean.reshape(ny, nx, 4)
    same = lambda a, b: np.all(np.isclose(a, b, atol=1e-6), axis=-1)
    m = np.zeros((ny, nx), dtype=bool)
    dh = ~same(g[:, :-1], g[:, 1:])
    dv = ~same(g[:-1, :], g[1:, :])
    m[:, :-1] |= dh
    m[:, 1:] |= dh
    m[:-1, :] |= dv
    m[1:, :] |= dv
    return m.ravel()


def score(q_clean, q_test, sym, bad, bnd):
    d = disorientation_deg(q_clean, q_test, sym)
    row = {
        "all_med": np.median(d),
        "all_mean": d.mean(),
        "bad_med": np.median(d[bad]) if bad is not None and bad.any() else np.nan,
        "bad_mean": d[bad].mean() if bad is not None and bad.any() else np.nan,
        "bnd_med": np.median(d[bnd]) if bnd.any() else np.nan,
        "still_bad": 100.0 * (d > 10.0).mean(),
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True, help="clean Euler .txt (ground truth)")
    p.add_argument("--noisy", required=True, help="noisy Euler .txt (the input)")
    p.add_argument("--mask", default=None, help="the .badmask.npy for this map")
    p.add_argument("--results", required=True,
                   help="folder of denoised .txt files, named <stem>__<method>.txt")
    p.add_argument("--shape", nargs=2, type=int, required=True, metavar=("NY", "NX"))
    args = p.parse_args()

    shape = tuple(args.shape)
    n = int(np.prod(shape))
    sym = cubic_sym()

    e_clean = load_euler(args.clean)
    if e_clean.shape[0] != n:
        sys.exit(f"clean has {e_clean.shape[0]} rows, --shape implies {n}")
    q_clean = euler_to_quat(e_clean)

    bad = None
    if args.mask:
        bad = np.load(args.mask).ravel().astype(bool)
    bnd = boundary_mask(q_clean, shape)

    rows = []
    rows.append(("noisy (no filter)", score(q_clean, euler_to_quat(load_euler(args.noisy)),
                                            sym, bad, bnd)))

    files = sorted(Path(args.results).glob("*__*.txt"))
    if not files:
        print(f"warning: no *__*.txt files in {args.results}")
    for f in files:
        method = f.stem.split("__")[-1]
        try:
            e = load_euler(f)
            if e.shape[0] != n:
                print(f"  [skip] {f.name}: {e.shape[0]} rows")
                continue
            rows.append((method, score(q_clean, euler_to_quat(e), sym, bad, bnd)))
        except Exception as exc:
            print(f"  [skip] {f.name}: {exc}")

    print(f"\nmap        : {Path(args.clean).name}")
    print(f"pixels     : {n}")
    if bad is not None:
        print(f"corrupted  : {bad.sum()} ({100*bad.mean():.1f}%)")
    print(f"boundary   : {bnd.sum()} ({100*bnd.mean():.1f}%)")

    print("\n" + "=" * 78)
    print(f"{'method':<20} {'ALL':>12} {'CORRUPTED':>12} {'BOUNDARY':>12} {'>10 deg':>10}")
    print(f"{'':<20} {'median deg':>12} {'median deg':>12} {'median deg':>12} {'% px':>10}")
    print("-" * 78)
    for name, r in rows:
        print(f"{name:<20} {r['all_med']:>12.3f} {r['bad_med']:>12.3f} "
              f"{r['bnd_med']:>12.3f} {r['still_bad']:>10.2f}")
    print("=" * 78)

    base = rows[0][1]
    best = min(rows[1:], key=lambda kv: kv[1]["all_med"]) if len(rows) > 1 else None
    print("\nHow to read this")
    print("  ALL       lower is better, but flattered by untouched pixels")
    print("  CORRUPTED did the method actually repair the damaged pixels?")
    print("  BOUNDARY  filters smear here; this predicts twin-boundary behaviour")
    print("  >10 deg   share of pixels still grossly wrong after filtering")
    if best:
        print(f"\n  best on ALL: {best[0]} "
              f"({base['all_med']:.3f} -> {best[1]['all_med']:.3f} deg)")
        print(f"  that method on CORRUPTED: "
              f"{base['bad_med']:.3f} -> {best[1]['bad_med']:.3f} deg")


if __name__ == "__main__":
    main()