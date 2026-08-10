#!/usr/bin/env python3
"""
STEP 4 -- run this AFTER EMEBSD has produced Ni_EBSD_sim.h5.

Compares simulated patterns against the real ones at the same scan points.
If the pattern centre is right, bands sit in roughly the same places.

Run from ~/EMsoftData_work/06_euler_angle_validation with the venv active:
    python3 04_pc_check.py
    python3 04_pc_check.py --sim ../03_ebsd_pattern_simulation/Ni_EBSD_sim.h5
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import kikuchipy as kp

p = argparse.ArgumentParser()
p.add_argument("--real", default="patterns.h5")
p.add_argument("--sim", default="../03_ebsd_pattern_simulation/Ni_EBSD_sim.h5")
p.add_argument("--out", default="outputs/pc_check.png")
args = p.parse_args()

# ----------------------------------------------------------------------
print(f"Loading real scan: {args.real}")
s = kp.load(args.real)
ny, nx = s.axes_manager.navigation_shape[::-1]
s.remove_static_background()
s.remove_dynamic_background()
real = np.asarray(s.data)
print(f"  {ny} x {nx} points, patterns {real.shape[2]} x {real.shape[3]}")

print(f"Loading simulated patterns: {args.sim}")
try:
    with h5py.File(args.sim, "r") as f:
        found = []
        f.visititems(lambda n, o: found.append((n, o.shape))
                     if isinstance(o, h5py.Dataset) and o.ndim >= 3 else None)
        print("  3-D datasets present:")
        for n, sh in found:
            print(f"    {n}  {sh}")
        key = "EMData/EBSD/EBSDPatterns"
        if key not in f:
            if not found:
                sys.exit("No 3-D dataset found in the simulated file.")
            key = found[0][0]
            print(f"  falling back to {key}")
        sim = f[key][:]
except FileNotFoundError:
    sys.exit(f"{args.sim} not found -- run EMEBSD first.")

sim = np.squeeze(sim)
print(f"  simulated array: {sim.shape}")

if sim.shape[0] != ny * nx:
    print(f"  WARNING: {sim.shape[0]} simulated patterns but {ny*nx} scan points."
          " Index mapping may be wrong.")

# ----------------------------------------------------------------------
picks = [(9, 12), (18, 37), (27, 25), (36, 50)]
fig, ax = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 6.6))
for j, (r, c) in enumerate(picks):
    i = r * nx + c
    ax[0, j].imshow(real[r, c], cmap="gray")
    ax[0, j].set_title(f"real ({r},{c})", fontsize=10)
    ax[0, j].axis("off")
    if i < sim.shape[0]:
        ax[1, j].imshow(sim[i], cmap="gray")
        ax[1, j].set_title(f"simulated, line {i}", fontsize=10)
    else:
        ax[1, j].text(0.5, 0.5, "out of range", ha="center", va="center")
    ax[1, j].axis("off")

fig.suptitle("Pattern centre check: bands should sit in the same places",
             fontsize=13)
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {args.out}")
print("""
Compare column by column:
  bands in roughly the same places  -> PC is right, continue
  everything shifted the same way   -> xpc / ypc offset wrong
  vertically mirrored               -> ypc sign convention wrong
  same shapes, wrong scale          -> L / delta ratio wrong
  unrelated patterns                -> wrong ordering or wrong master pattern
""")