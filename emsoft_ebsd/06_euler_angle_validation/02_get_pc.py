#!/usr/bin/env python3
"""
STEP 1 -- run this NOW, before EMEBSD.

Prints the pattern centre of the source scan in EMsoft's convention, so you can
fill in xpc, ypc, L and delta in EMEBSD.nml.

Run from ~/EMsoftData_work/06_euler_angle_validation with the venv active:
    python3 02_get_pc.py
"""

import numpy as np
import kikuchipy as kp

s = kp.load("patterns.h5")
det = s.detector

print("Detector as recorded in the dataset")
print(f"  shape (Ny, Nx)     : {det.shape}")
print(f"  pc (PCx, PCy, PCz) : {det.pc_average}")
print(f"  sample_tilt        : {det.sample_tilt}")
print(f"  detector tilt      : {det.tilt}")
print(f"  binning            : {det.binning}")
print(f"  px_size            : {det.px_size}")

numsy, numsx = det.shape

# ----------------------------------------------------------------------
# EMsoft convention, straight from kikuchipy
# ----------------------------------------------------------------------
print("\nEMsoft convention via pc_emsoft()")
pc_em = None
for kwargs in ({"version": 5}, {}):
    try:
        pc_em = np.asarray(det.pc_emsoft(**kwargs)).ravel()
        print(f"  (called with {kwargs or 'no arguments'})")
        break
    except Exception as e:
        print(f"  attempt with {kwargs or 'no arguments'} failed: {e}")

if pc_em is not None and pc_em.size >= 3:
    xpc, ypc, L = (float(pc_em[0]), float(pc_em[1]), float(pc_em[2]))
    print(f"  xpc   = {xpc:.4f}")
    print(f"  ypc   = {ypc:.4f}")
    print(f"  L     = {L:.4f}   (in units of px_size = {det.px_size})")
    print(f"  delta = {float(det.px_size):.4f}   <- must match the L above")
else:
    print("  pc_emsoft() unavailable -- use the manual values below instead")

# ----------------------------------------------------------------------
# Manual derivation, using a realistic 50 um pixel
# ----------------------------------------------------------------------
pcx, pcy, pcz = (float(v) for v in np.asarray(det.pc_average).ravel()[:3])
delta = 50.0
L_manual = pcz * numsx * delta
xpc_manual = (pcx - 0.5) * numsx
ypc_manual = (pcy - 0.5) * numsy

print(f"\nManual derivation with delta = {delta} um")
print(f"  xpc   = {xpc_manual:.4f}")
print(f"  ypc   = {ypc_manual:.4f}")
print(f"  L     = {L_manual:.4f}")
print(f"  delta = {delta}")

print(f"""
Only the ratio L / (numsx * delta) matters -- that is PCz = {pcz:.4f}.
Pick ONE of the two blocks above and use all four numbers from it together.
The sign of ypc and the exact xpc offset differ between conventions, which is
why the visual check in step 3 exists.

Paste into EMEBSD.nml:
 numsx  = {numsx},
 numsy  = {numsy},
 thetac = {det.tilt},
""")