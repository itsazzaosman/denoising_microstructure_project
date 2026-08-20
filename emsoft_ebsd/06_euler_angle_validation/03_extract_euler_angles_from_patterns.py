#!/usr/bin/env python3
"""
patterns.h5  ->  ni_angles.txt  (EMsoft angle file)

Extracts the 4125 indexed orientations, converts them to degrees, and writes the
angle file EMEBSD expects. Order is preserved exactly, so line i of the angle
file corresponds to scan point (row = i // nx, col = i % nx).

Run:
    python3 03_extract_euler_angles_from_patterns.py
    python3 03_extract_euler_angles_from_patterns.py --file patterns.h5 --out ni_angles.txt
"""

import argparse
import numpy as np
import kikuchipy as kp

p = argparse.ArgumentParser()
p.add_argument("--file", default="patterns.h5")
p.add_argument("--out", default="ni_angles.txt")
p.add_argument("--map-out", default="ni_angles_map.csv")
args = p.parse_args()

# ----------------------------------------------------------------------
print(f"Loading {args.file}")
s = kp.load(args.file)
xmap = s.xmap
ny, nx = xmap.shape
print(f"  scan grid: {ny} x {nx} = {ny * nx} points")
print(f"  phase: {xmap.phases[0].name}, point group {xmap.phases[0].point_group.name}")

# ----------------------------------------------------------------------
# Euler angles -> degrees, Bunge convention
# ----------------------------------------------------------------------
eu = np.asarray(xmap.rotations.to_euler()).reshape(-1, 3)

if np.nanmax(np.abs(eu)) <= 2 * np.pi + 1e-6:
    print("  angles are in radians -> converting to degrees")
    eu = np.rad2deg(eu)
else:
    print("  angles already in degrees")

eu[:, 0] %= 360.0
eu[:, 1] %= 180.0
eu[:, 2] %= 360.0

# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------
print("\nSanity checks")
print(f"  count            : {len(eu)}   (expect {ny * nx})")
print(f"  phi1 range [deg] : {eu[:,0].min():8.3f} to {eu[:,0].max():8.3f}   (expect ~0-360)")
print(f"  PHI  range [deg] : {eu[:,1].min():8.3f} to {eu[:,1].max():8.3f}   (expect ~0-180)")
print(f"  phi2 range [deg] : {eu[:,2].min():8.3f} to {eu[:,2].max():8.3f}   (expect ~0-360)")
print(f"  any NaN          : {np.isnan(eu).any()}")

n_unique = len(np.unique(np.round(eu, 1), axis=0))
print(f"  distinct to 0.1 deg : {n_unique} of {len(eu)}  "
      f"(fewer than total = neighbouring points share grains, as expected)")

# ----------------------------------------------------------------------
# Write EMsoft angle file
# ----------------------------------------------------------------------
with open(args.out, "w") as f:
    f.write("eu\n")
    f.write(f"{len(eu)}\n")
    for a, b, c in eu:
        f.write(f"{a:12.6f} {b:12.6f} {c:12.6f}\n")

print(f"\nwrote {args.out}")
with open(args.out) as f:
    head = [next(f) for _ in range(5)]
print("  first lines:")
for line in head:
    print("   ", line.rstrip())

# ----------------------------------------------------------------------
# Row/col mapping, so patterns can be put back on the grid
# ----------------------------------------------------------------------
with open(args.map_out, "w") as f:
    f.write("angle_line,row,col,phi1_deg,PHI_deg,phi2_deg\n")
    for i, (a, b, c) in enumerate(eu):
        f.write(f"{i},{i // nx},{i % nx},{a:.6f},{b:.6f},{c:.6f}\n")
print(f"wrote {args.map_out}")

# ----------------------------------------------------------------------
# Detector geometry -> EMEBSD.nml
# ----------------------------------------------------------------------
det = s.detector
print("\nDetector geometry of the source scan:")
print(f"  {det}")

try:
    pc_em = np.atleast_2d(det.pc_emsoft())[0]
    xpc, ypc, L = pc_em[0], pc_em[1], pc_em[2]
    pc_line = f"  xpc = {xpc:.4f},  ypc = {ypc:.4f},  L = {L:.2f} um"
except Exception as e:
    pc_line = f"  (PC conversion to EMsoft convention failed: {e})"
print(pc_line)

print(f"""
--------------------------------------------------------------
Set these in EMEBSD.nml
--------------------------------------------------------------
 anglefile       = '{args.out}',
 anglefiletype   = 'orientations',
 eulerconvention = 'tsl',
 masterfile      = '02_master_pattern/Ni_master_hires.h5',
 datafile        = '03_ebsd_pattern_simulation/Ni_EBSD_sim.h5',
 numsx           = {det.shape[1]},        ! match the source scan, or raise
 numsy           = {det.shape[0]},        ! both if you want bigger patterns
 thetac          = {det.tilt},
 energymin       = 10.0,       ! must be >= your Monte Carlo Ehistmin
 energymax       = 20.0,
 nthreads        = 16,
{pc_line}

Reassemble the simulated patterns onto the scan grid afterwards with:
    sim.reshape({ny}, {nx}, numsy, numsx)
--------------------------------------------------------------
""")