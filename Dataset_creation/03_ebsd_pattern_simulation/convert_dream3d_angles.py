#!/usr/bin/env python3
"""
Converts a Dream3D Euler angle export into an EMsoft angle file.

Dream3D writes one Euler triplet per grid pixel, in radians, with a plain
text column header - not usable by EMEBSD's `anglefile` as-is. EMsoft wants
a first line naming the angle type ('eu'), a count line, then that many
triplets in degrees - same format as 10_orientation_map/map_angles_40k.txt,
matched here for consistency with the rest of this repo.

Does not check/adjust Euler angle convention (TSL vs HKL, axis order) -
verify the simulated patterns visually against ground truth before trusting
the convention assumption implicit in EMEBSD_ni.nml's `eulerconvention`.

Run with:
    python3 convert_dream3d_angles.py
"""

import os
import numpy as np

IN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Nickel_100x100_Euler_v2.txt")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Nickel_100x100_angles_v2.txt")

euler_rad = np.loadtxt(IN_FILE, skiprows=1)
euler_deg = np.degrees(euler_rad)

with open(OUT_FILE, "w") as f:
    f.write("eu\n")
    f.write(f"{len(euler_deg)}\n")
    for phi1, PHI, phi2 in euler_deg:
        f.write(f"{phi1:12.6f} {PHI:12.6f} {phi2:12.6f}\n")

print(f"read {IN_FILE}  ({len(euler_rad)} orientations, radians)")
print(f"wrote {OUT_FILE}  ({len(euler_deg)} orientations, degrees)")
