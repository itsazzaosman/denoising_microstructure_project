#!/usr/bin/env python3
"""
STEP 1 of the synthetic-orientation pipeline.

Generates a list of uniformly random crystal orientations and writes them as an
EMsoft angle file. These drive EMEBSD pattern simulation.

Why synthetic rather than orientations from a real scan: the denoiser only ever
sees the diffraction pattern, never the orientation, so all it needs is variety.
A real scan is a poor source of variety because neighbouring points share grains
(our 4125-point scan contained only 3490 distinct orientations) and a textured
sample leaves whole regions of orientation space unsampled. Uniform sampling has
neither problem.

Uniform sampling of SO(3) is equivalent to uniform sampling of the cubic
fundamental zone, since symmetrically equivalent orientations produce identical
patterns. So no explicit fundamental-zone reduction is needed here.

Run with the environment that has orix:
    ~/.ebsd/bin/python 01_generate_angles.py
"""

import numpy as np
from orix.quaternion import Rotation

# ======================================================================
# SETTINGS
# ======================================================================
N_ORIENTATIONS = 30000
OUT_FILE = "/home/azza_osman/EMsoftData_work/08_synthetic_training_data/ni_angles_30k.txt"
SEED = 1234
# ======================================================================

rng_rotations = Rotation.random(N_ORIENTATIONS)

euler = np.asarray(rng_rotations.to_euler()).reshape(-1, 3)

# orix has returned radians historically, but this has moved between versions,
# so detect rather than assume.
if np.nanmax(np.abs(euler)) <= 2 * np.pi + 1e-6:
    print("converting radians to degrees")
    euler = np.rad2deg(euler)

euler[:, 0] %= 360.0
euler[:, 1] %= 180.0
euler[:, 2] %= 360.0

print(f"{len(euler)} orientations")
print(f"  phi1 : {euler[:,0].min():7.2f} to {euler[:,0].max():7.2f}")
print(f"  PHI  : {euler[:,1].min():7.2f} to {euler[:,1].max():7.2f}")
print(f"  phi2 : {euler[:,2].min():7.2f} to {euler[:,2].max():7.2f}")
print(f"  distinct to 0.1 deg: {len(np.unique(np.round(euler, 1), axis=0))}")

with open(OUT_FILE, "w") as f:
    f.write("eu\n")
    f.write(f"{len(euler)}\n")
    for phi1, PHI, phi2 in euler:
        f.write(f"{phi1:12.6f} {PHI:12.6f} {phi2:12.6f}\n")

print(f"\nwrote {OUT_FILE}")
print("\nNext: point EMEBSD at this file (see the pipeline instructions).")