#!/usr/bin/env python3
"""
Converts a batch of Dream3D Euler angle exports into EMsoft angle files.

Dream3D writes one Euler triplet per grid pixel, in radians, with a plain
text column header - not usable by EMEBSD's `anglefile` as-is. EMsoft wants
a first line naming the angle type ('eu'), a count line, then that many
triplets in degrees - same format as 10_orientation_map/map_angles_40k.txt,
matched here for consistency with the rest of this repo.

Batch mode: converts every Euler_<index>.txt file in IN_DIR (produced by
11_batch_generation/generate_microstructures_windows.py and pulled over from
the Windows side) into a matching angles_<index>.txt file in OUT_DIR, keeping
the same index so each map's angle file can be paired back up with it later.
Resumable - a map that's already been converted is skipped.

Does not check/adjust Euler angle convention (TSL vs HKL, axis order) -
verify the simulated patterns visually against ground truth before trusting
the convention assumption implicit in EMEBSD_ni.nml's `eulerconvention`.

Run with:
    python3 convert_dream3d_angles.py
"""

import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "11_batch_generation", "euler_maps")
OUT_DIR = os.path.join(HERE, "..", "11_batch_generation", "angles")

IN_PATTERN = os.path.join(IN_DIR, "Euler_*.txt")
INDEX_RE = re.compile(r"Euler_(\d+)\.txt$")


def convert_one(in_file, out_file):
    euler_rad = np.loadtxt(in_file, skiprows=1)
    euler_deg = np.degrees(euler_rad)

    with open(out_file, "w") as f:
        f.write("eu\n")
        f.write(f"{len(euler_deg)}\n")
        for phi1, PHI, phi2 in euler_deg:
            f.write(f"{phi1:12.6f} {PHI:12.6f} {phi2:12.6f}\n")

    return len(euler_deg)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    in_files = sorted(glob.glob(IN_PATTERN))
    if not in_files:
        print(f"No files matching {IN_PATTERN} - nothing to convert.")
        return

    converted = skipped = failed = 0
    for in_file in in_files:
        match = INDEX_RE.search(os.path.basename(in_file))
        if not match:
            print(f"[SKIP] {in_file} - filename doesn't match Euler_<index>.txt")
            skipped += 1
            continue

        index = match.group(1)
        out_file = os.path.join(OUT_DIR, f"angles_{index}.txt")

        if os.path.exists(out_file):
            skipped += 1
            continue

        try:
            convert_one(in_file, out_file)
        except Exception as e:
            print(f"[FAILED] {in_file}: {e}")
            failed += 1
            continue

        converted += 1
        if converted % 100 == 0:
            print(f"{converted}/{len(in_files)} converted...")

    print(
        f"Done. {converted} converted, {skipped} skipped (already done), "
        f"{failed} failed, out of {len(in_files)} input files found."
    )


if __name__ == "__main__":
    main()
