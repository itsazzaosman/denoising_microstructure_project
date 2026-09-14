#!/usr/bin/env python3
"""
Add realistic EBSD noise to a clean DREAM.3D Euler-angle map, and export to
.ang so MTEX can read the same data.

Two noise types, applied together, because real maps have both:

  1. SCATTER   every pixel is nudged by a small random rotation (< a few deg).
               This is angular uncertainty in indexing. DREAM.3D's
               "Add Orientation Noise" filter does this; included here so the
               whole noise model lives in one place.

  2. MISINDEX  a fraction p of pixels are REPLACED with a completely random
               orientation (tens of degrees wrong). This is what actually
               breaks real maps: the pattern was too weak to index and the
               software picked the wrong solution. Salt-and-pepper, but in
               orientation space instead of colour space.

Misindexing can be spread uniformly, or concentrated near grain boundaries
(--boundary-bias), which is where it happens in reality: a boundary pixel's
pattern is a mix of two grains, so it indexes badly.

Usage
-----
    # one map
    python add_ebsd_noise.py clean.txt noisy.txt --shape 128 128 \
        --scatter 1.0 --misindex 0.05 --boundary-bias 4

    # a folder of maps: first 500 only
    python add_ebsd_noise.py clean_euler/ noisy_mis05/ --shape 128 128 \
        --limit 500 --scatter 1.0 --misindex 0.05 --boundary-bias 4 --save-mask

    # also write .ang files for MTEX
    python add_ebsd_noise.py clean_euler/ noisy_mis05/ --shape 128 128 \
        --limit 500 --misindex 0.05 --ang

Pass a file for single-map mode, or a folder for batch mode. In batch mode each
map gets seed+index, so maps differ from one another but a rerun reproduces the
same corruption. Outputs that already exist are skipped, so an interrupted run
resumes.

Euler angles are Bunge ZXZ in radians throughout, matching DREAM.3D.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# quaternion helpers  (w, x, y, z)
# ---------------------------------------------------------------------------


def euler_to_quat(euler):
    """Bunge ZXZ (phi1, Phi, phi2) radians -> unit quaternion."""
    phi1, Phi, phi2 = euler[:, 0], euler[:, 1], euler[:, 2]
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack([c * np.cos(sigma), -s * np.cos(delta),
                  -s * np.sin(delta), -c * np.sin(sigma)], axis=-1)
    q[q[:, 0] < 0] *= -1.0
    return q


def quat_to_euler(q):
    """Unit quaternion -> Bunge ZXZ (phi1, Phi, phi2) radians in [0, 2pi)."""
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    q03, q12 = q0**2 + q3**2, q1**2 + q2**2
    chi = np.sqrt(q03 * q12)
    phi1 = np.zeros_like(q0)
    Phi = np.zeros_like(q0)
    phi2 = np.zeros_like(q0)

    a = chi > 1e-12
    phi1[a] = np.arctan2((-q0[a] * q2[a] + q1[a] * q3[a]) / chi[a],
                         (-q0[a] * q1[a] - q2[a] * q3[a]) / chi[a])
    Phi[a] = np.arctan2(2 * chi[a], q03[a] - q12[a])
    phi2[a] = np.arctan2((q0[a] * q2[a] + q1[a] * q3[a]) / chi[a],
                         (-q0[a] * q1[a] + q2[a] * q3[a]) / chi[a])

    b = (~a) & (q12 < 1e-12)
    phi1[b] = np.arctan2(-2 * q0[b] * q3[b], q0[b]**2 - q3[b]**2)

    c = (~a) & (q03 < 1e-12)
    phi1[c] = np.arctan2(2 * q1[c] * q2[c], q1[c]**2 - q2[c]**2)
    Phi[c] = np.pi

    return np.stack([phi1 % (2 * np.pi), Phi, phi2 % (2 * np.pi)], axis=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def random_quats(n, rng):
    """Uniformly distributed random orientations (Shoemake's method)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    return np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=-1)


def small_rotations(n, max_deg, rng):
    """Random-axis rotations, angle uniform in [0, max_deg] (matches DREAM.3D)."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.radians(rng.random(n) * max_deg) / 2.0
    return np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


def boundary_distance_weight(q, shape, bias):
    """
    Weight each pixel by how close it is to a grain boundary, so misindexing
    lands where it lands in reality. A pixel is 'on a boundary' if any of its
    4 neighbours has a different orientation.

    Returns probabilities that sum to 1. bias=1 means uniform (no bias);
    higher bias concentrates misindexing on boundary pixels.
    """
    ny, nx = shape
    g = q.reshape(ny, nx, 4)
    same = lambda a, b: np.all(np.isclose(a, b, atol=1e-6), axis=-1)

    on_boundary = np.zeros((ny, nx), dtype=bool)
    on_boundary[:, :-1] |= ~same(g[:, :-1], g[:, 1:])
    on_boundary[:, 1:] |= ~same(g[:, :-1], g[:, 1:])
    on_boundary[:-1, :] |= ~same(g[:-1, :], g[1:, :])
    on_boundary[1:, :] |= ~same(g[:-1, :], g[1:, :])

    w = np.ones((ny, nx), dtype=float)
    w[on_boundary] = float(bias)
    return (w / w.sum()).ravel(), on_boundary.ravel()


def add_noise(euler, shape, scatter_deg, misindex_frac, boundary_bias, rng):
    q = euler_to_quat(euler)
    n = q.shape[0]

    # 1. small-angle scatter on every pixel
    if scatter_deg > 0:
        q = quat_mul(small_rotations(n, scatter_deg, rng), q)

    # 2. misindexing: replace a fraction of pixels outright
    n_bad = int(round(misindex_frac * n))
    bad_idx = np.array([], dtype=int)
    if n_bad > 0:
        if boundary_bias and boundary_bias > 1 and shape is not None:
            p, _ = boundary_distance_weight(euler_to_quat(euler), shape, boundary_bias)
            bad_idx = rng.choice(n, size=n_bad, replace=False, p=p)
        else:
            bad_idx = rng.choice(n, size=n_bad, replace=False)
        q[bad_idx] = random_quats(n_bad, rng)

    return quat_to_euler(q), bad_idx


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

HEADER = "EulerAngles_0 EulerAngles_1 EulerAngles_2"


def load_euler(path):
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
        sys.exit(f"{path}: expected >=3 columns, got {data.shape[1]}")
    return data[:, :3]


def save_euler(path, euler):
    np.savetxt(path, euler, header=HEADER, comments="", fmt="%.8g")


def save_ang(path, euler, shape, step=1.0, phase_name="Nickel", lattice=(3.524, 3.524, 3.524)):
    """
    Write a minimal TSL .ang file that MTEX reads with:
        ebsd = EBSD.load('map.ang', CS, 'convertSpatial2EulerReferenceFrame')

    Columns: phi1 PHI phi2 x y IQ CI phase_index detector_intensity fit
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx.ravel() * step).astype(float)
    y = (yy.ravel() * step).astype(float)
    n = euler.shape[0]

    head = [
        "# TEM_PIXperUM          1.000000",
        "# x-star                0.000000",
        "# y-star                0.000000",
        "# z-star                0.000000",
        "# WorkingDistance       15.000000",
        "#",
        f"# Phase 1",
        f"# MaterialName  \t{phase_name}",
        f"# Formula     \t{phase_name}",
        "# Info",
        "# Symmetry              43",
        f"# LatticeConstants      {lattice[0]:.3f} {lattice[1]:.3f} {lattice[2]:.3f}"
        "  90.000  90.000  90.000",
        "# NumberFamilies        0",
        "#",
        "# GRID: SqrGrid",
        f"# XSTEP: {step:.6f}",
        f"# YSTEP: {step:.6f}",
        f"# NCOLS_ODD: {nx}",
        f"# NCOLS_EVEN: {nx}",
        f"# NROWS: {ny}",
        "#",
        "# OPERATOR: synthetic",
        "# SAMPLEID:",
        "# SCANID:",
        "#",
    ]

    body = np.column_stack([
        euler[:, 0], euler[:, 1], euler[:, 2],
        x, y,
        np.full(n, 100.0),   # IQ  (image quality, constant for synthetic data)
        np.full(n, 0.8),     # CI  (confidence index)
        np.ones(n),          # phase index
        np.full(n, 100.0),   # detector intensity
        np.zeros(n),         # fit
    ])

    with open(path, "w") as f:
        f.write("\n".join(head) + "\n")
        np.savetxt(f, body, fmt="%9.5f %9.5f %9.5f %12.5f %12.5f %8.1f %6.3f %2d %8.1f %6.3f")


def natural_key(s):
    """Sort map_2 before map_10, which plain sorted() gets wrong."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def process_one(in_path, out_path, shape, args, rng, quiet=False):
    """Noise a single map. Returns the number of corrupted pixels."""
    euler = load_euler(in_path)
    if euler.shape[0] != int(np.prod(shape)):
        raise ValueError(f"{euler.shape[0]} rows but --shape implies {int(np.prod(shape))}")

    noisy, bad_idx = add_noise(euler, shape, args.scatter, args.misindex,
                               args.boundary_bias, rng)
    save_euler(out_path, noisy)

    if args.save_mask:
        mask = np.zeros(euler.shape[0], dtype=bool)
        mask[bad_idx] = True
        np.save(str(out_path) + ".badmask.npy", mask.reshape(shape))
    if args.ang:
        base = str(out_path).rsplit(".", 1)[0]
        save_ang(base + "_clean.ang", euler, shape, args.step)
        save_ang(base + "_noisy.ang", noisy, shape, args.step)

    if not quiet:
        print(f"  {os.path.basename(str(in_path))} -> {out_path}  "
              f"({len(bad_idx)} px misindexed)")
    return len(bad_idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clean", help="clean Euler .txt, OR a folder of them")
    p.add_argument("noisy", help="output .txt, OR an output folder if input is a folder")
    p.add_argument("--shape", nargs=2, type=int, required=True, metavar=("NY", "NX"))
    p.add_argument("--limit", type=int, default=500,
                   help="folder mode: process only the first N maps (0 = all)")
    p.add_argument("--start", type=int, default=0,
                   help="folder mode: skip the first N maps")
    p.add_argument("--glob", default="*.txt",
                   help="folder mode: which files to pick up")
    p.add_argument("--scatter", type=float, default=1.0,
                   help="max small-angle scatter in degrees, applied to every pixel "
                        "(0 to disable; mean is half this)")
    p.add_argument("--misindex", type=float, default=0.05,
                   help="fraction of pixels replaced with a random orientation")
    p.add_argument("--boundary-bias", type=float, default=1.0,
                   help="how much more likely a boundary pixel is to be misindexed "
                        "(1 = uniform, 4 = 4x more likely on boundaries)")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; in folder mode each map uses seed+index, so maps "
                        "differ from each other but a rerun reproduces them exactly")
    p.add_argument("--save-mask", action="store_true",
                   help="save which pixels were corrupted (needed for masked scoring)")
    p.add_argument("--ang", action="store_true", help="also write .ang files for MTEX")
    p.add_argument("--step", type=float, default=1.0, help="pixel size in microns")
    args = p.parse_args()

    shape = tuple(args.shape)
    in_path = Path(args.clean)

    # ---------- single file ----------
    if in_path.is_file():
        rng = np.random.default_rng(args.seed)
        n_bad = process_one(in_path, Path(args.noisy), shape, args, rng, quiet=True)
        print(f"scatter     : 0-{args.scatter} deg on every pixel")
        print(f"misindexed  : {n_bad} px ({args.misindex:.2%})")
        print(f"wrote       : {args.noisy}")
        return

    # ---------- folder ----------
    if not in_path.is_dir():
        sys.exit(f"not found: {in_path}")

    files = sorted(in_path.glob(args.glob), key=natural_key)
    files = [f for f in files if not f.name.startswith(".")]
    if not files:
        sys.exit(f"no files matching {args.glob} in {in_path}")

    selected = files[args.start:]
    if args.limit > 0:
        selected = selected[:args.limit]

    outdir = Path(args.noisy)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"input       : {in_path}  ({len(files)} files found)")
    print(f"processing  : {len(selected)} maps")
    print(f"noise       : scatter 0-{args.scatter} deg, misindex {args.misindex:.1%}, "
          f"boundary bias {args.boundary_bias}x")
    print(f"output      : {outdir.resolve()}\n")

    done = skipped = failed = 0
    for i, f in enumerate(selected):
        out_path = outdir / f"{f.stem}_noisy.txt"

        # Only skip when every output this run was asked to produce already
        # exists. Checking the .txt alone would skip maps that need masks or
        # .ang files added by a later run.
        wanted = [out_path]
        if args.save_mask:
            wanted.append(Path(str(out_path) + ".badmask.npy"))
        if args.ang:
            base = str(out_path).rsplit(".", 1)[0]
            wanted += [Path(base + "_clean.ang"), Path(base + "_noisy.ang")]

        if all(w.exists() for w in wanted):
            skipped += 1
            continue
        try:
            rng = np.random.default_rng(args.seed + args.start + i)
            process_one(f, out_path, shape, args, rng, quiet=True)
            done += 1
        except Exception as e:
            failed += 1
            print(f"  [skip] {f.name}: {e}")

        if (i + 1) % 50 == 0 or i + 1 == len(selected):
            print(f"  {i + 1}/{len(selected)}  ok={done} skipped={skipped} failed={failed}")

    print(f"\nDone. {done} written, {skipped} already existed, {failed} failed.")


if __name__ == "__main__":
    main()