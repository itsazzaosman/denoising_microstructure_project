#!/usr/bin/env python3
"""
Verify the magnitude and granularity of DREAM.3D's Add Orientation Noise filter.

Reads two Euler-angle files exported by 'Export ASCII Data' (clean and noisy),
computes the per-pixel disorientation delta under cubic (m-3m) symmetry, and
reports:

  1. mean / median / percentiles of delta, to check against the magnitude you
     set in the filter;
  2. whether the perturbation is per-voxel or per-grain, inferred from how many
     distinct delta values appear and how often neighbouring pixels share one.

Usage:
    python verify_orientation_noise.py clean.txt noisy.txt --shape 128 128

Assumes Bunge ZXZ Euler angles, three columns per row, one row per voxel in
DREAM.3D's raster order (x fastest).
"""

import argparse
import sys

import numpy as np


# ----------------------------------------------------------------------------
# cubic symmetry operators as quaternions, (w, x, y, z)
# ----------------------------------------------------------------------------
def cubic_symmetry_quaternions():
    """The 24 proper rotations of the cube, as unit quaternions."""
    h = 0.5
    r = np.sqrt(2.0) / 2.0
    q = [
        # identity
        (1, 0, 0, 0),
        # 180 deg about <100>
        (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
        # 90 deg about <100>
        (r, r, 0, 0), (r, -r, 0, 0),
        (r, 0, r, 0), (r, 0, -r, 0),
        (r, 0, 0, r), (r, 0, 0, -r),
        # 180 deg about <110>
        (0, r, r, 0), (0, r, -r, 0),
        (0, r, 0, r), (0, r, 0, -r),
        (0, 0, r, r), (0, 0, r, -r),
        # 120 deg about <111>
        (h, h, h, h), (h, -h, -h, -h),
        (h, h, -h, h), (h, -h, h, -h),
        (h, -h, h, h), (h, h, -h, -h),
        (h, -h, -h, h), (h, h, h, -h),
    ]
    return np.array(q, dtype=np.float64)


def quat_multiply(a, b):
    """Hamilton product. a, b broadcastable arrays with last axis = 4."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conjugate(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_bunge_to_quaternion(euler):
    """Bunge ZXZ (phi1, Phi, phi2) in radians -> unit quaternion (w,x,y,z)."""
    phi1, Phi, phi2 = euler[:, 0], euler[:, 1], euler[:, 2]
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    c = np.cos(0.5 * Phi)
    s = np.sin(0.5 * Phi)
    q = np.stack([
        c * np.cos(sigma),
        -s * np.cos(delta),
        -s * np.sin(delta),
        -c * np.sin(sigma),
    ], axis=-1)
    # enforce positive scalar part for a canonical representative
    flip = q[:, 0] < 0
    q[flip] *= -1.0
    return q


def disorientation_deg(q1, q2, sym):
    """
    Per-row disorientation angle in degrees between two sets of orientations,
    minimised over cubic symmetry on both sides plus the switching symmetry.
    """
    m = quat_multiply(q2, quat_conjugate(q1))          # (N, 4)
    best = np.zeros(m.shape[0], dtype=np.float64)      # track max |scalar|
    for si in sym:
        left = quat_multiply(si[None, :], m)           # (N, 4)
        for sj in sym:
            full = quat_multiply(left, sj[None, :])    # (N, 4)
            np.maximum(best, np.abs(full[:, 0]), out=best)
    np.clip(best, -1.0, 1.0, out=best)
    return np.degrees(2.0 * np.arccos(best))


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_euler(path):
    """Load a DREAM.3D ASCII export, skipping a header row if present."""
    with open(path) as f:
        first = f.readline()
    try:
        [float(t) for t in first.split()]
        skip = 0
    except ValueError:
        skip = 1

    data = np.loadtxt(path, skiprows=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        sys.exit(f"{path}: expected >= 3 columns, found {data.shape[1]}. "
                 "Export EulerAngles only, or edit the column selection below.")
    euler = data[:, :3]

    # DREAM.3D writes radians; detect degrees just in case
    if np.nanmax(np.abs(euler)) > 2.0 * np.pi + 1e-6:
        print(f"  note: {path} looks like degrees, converting to radians")
        euler = np.radians(euler)
    return euler


# ----------------------------------------------------------------------------
# granularity test
# ----------------------------------------------------------------------------
def report_granularity(q_clean, q_noisy, delta, shape, sym):
    """
    Decide per-voxel vs per-grain by looking at pairs of ADJACENT voxels that
    share the same clean orientation (i.e. lie inside one grain), and asking
    whether their NOISY orientations still agree.

      per-grain  -> the whole grain is rotated rigidly, neighbours stay equal
      per-voxel  -> each voxel moves independently, neighbours now disagree

    This does not depend on the spread of the perturbation magnitude, which is
    why it is preferred over counting distinct delta values.
    """
    print("\nGranularity")
    if shape is None or int(np.prod(shape)) != delta.size:
        print("  need a matching --shape to run this test; skipping")
        return

    ny, nx = shape
    qc = q_clean.reshape(ny, nx, 4)
    qn = q_noisy.reshape(ny, nx, 4)

    # candidate neighbour pairs, horizontal and vertical
    pairs_c = [(qc[:, :-1].reshape(-1, 4), qc[:, 1:].reshape(-1, 4)),
               (qc[:-1, :].reshape(-1, 4), qc[1:, :].reshape(-1, 4))]
    pairs_n = [(qn[:, :-1].reshape(-1, 4), qn[:, 1:].reshape(-1, 4)),
               (qn[:-1, :].reshape(-1, 4), qn[1:, :].reshape(-1, 4))]

    within = []
    for (ca, cb), (na, nb) in zip(pairs_c, pairs_n):
        d_clean = disorientation_deg(ca, cb, sym)
        same_grain = d_clean < 1e-4          # identical clean orientation
        if same_grain.any():
            within.append(disorientation_deg(na[same_grain],
                                             nb[same_grain], sym))
    if not within:
        print("  found no adjacent voxel pairs sharing a clean orientation;")
        print("  the map may be single-voxel grains. Cannot decide.")
        return

    d_within = np.concatenate(within)
    print(f"  intra-grain neighbour pairs         : {d_within.size}")
    print(f"  mean delta between them (noisy)     : {d_within.mean():.4f} deg")
    print(f"  fraction still identical (<0.01 deg): "
          f"{(d_within < 0.01).mean():.3f}")

    print("\nVerdict")
    if (d_within < 0.01).mean() > 0.95:
        print("  PER-GRAIN. Neighbours inside a grain remain identical, so")
        print("  whole grains are rotated rigidly. There is no within-grain")
        print("  signal to average, which makes this a different and harder")
        print("  problem than angular scatter. Flag before the full run.")
    elif d_within.mean() > 0.5 * delta.mean():
        print("  PER-VOXEL. Neighbours inside a grain now disagree by an")
        print("  amount comparable to the noise magnitude, so each voxel was")
        print("  perturbed independently. This is the angular scatter case:")
        print("  grain interiors are recoverable by local averaging, so report")
        print("  delta at boundaries separately.")
    else:
        print("  AMBIGUOUS / MIXED. Intra-grain scatter is non-zero but small")
        print("  relative to the total. Inspect the delta map before the run.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clean", help="Euler angles exported before Add Orientation Noise")
    p.add_argument("noisy", help="Euler angles exported after Add Orientation Noise")
    p.add_argument("--shape", nargs=2, type=int, default=None,
                   metavar=("NY", "NX"), help="map shape, e.g. --shape 128 128")
    p.add_argument("--save-map", default=None,
                   help="optional .npy path to save the per-pixel delta map")
    args = p.parse_args()

    print("Loading")
    e1 = load_euler(args.clean)
    e2 = load_euler(args.noisy)
    if e1.shape != e2.shape:
        sys.exit(f"row count differs: {e1.shape[0]} vs {e2.shape[0]}")
    print(f"  {e1.shape[0]} voxels from each file")

    if np.allclose(e1, e2):
        sys.exit("\nThe two files are identical. Add Orientation Noise did not "
                 "modify the array -- check the filter's Euler Angles input and "
                 "that filter 12 sits AFTER filter 11.")

    sym = cubic_symmetry_quaternions()
    q1 = euler_bunge_to_quaternion(e1)
    q2 = euler_bunge_to_quaternion(e2)

    print("Computing disorientation (576 symmetry combinations)")
    delta = disorientation_deg(q1, q2, sym)

    print(f"\nDisorientation, degrees")
    print(f"  mean       : {delta.mean():.4f}")
    print(f"  median     : {np.median(delta):.4f}")
    print(f"  std        : {delta.std():.4f}")
    print(f"  min / max  : {delta.min():.4f} / {delta.max():.4f}")
    for q in (5, 25, 75, 95, 99):
        print(f"  p{q:<9d}: {np.percentile(delta, q):.4f}")

    shape = tuple(args.shape) if args.shape else None
    report_granularity(q1, q2, delta, shape, sym)

    if args.save_map and shape and int(np.prod(shape)) == delta.size:
        np.save(args.save_map, delta.reshape(shape))
        print(f"\nSaved delta map to {args.save_map}")


if __name__ == "__main__":
    main()