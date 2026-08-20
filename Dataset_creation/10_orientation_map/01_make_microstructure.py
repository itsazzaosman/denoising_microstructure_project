#!/usr/bin/env python3
"""
STEP 1 of the orientation-map pipeline.

Invents a fake piece of nickel with grains in it, and writes out one crystal
orientation per map pixel as an EMsoft angle file. EMEBSD then turns that
angle file into one diffraction pattern per pixel (STEP 2).

The important difference from 08_synthetic_training_data/01_generate_30k_angles.py:
there, every orientation was drawn independently, because the denoiser only ever
looks at one pattern at a time and just wants variety. Here the pixels have to
behave like a real sample surface -- neighbouring pixels usually sit inside the
same grain and therefore share an orientation. That shared-orientation structure
IS the microstructure, and it is what makes the result a map rather than a pile
of unrelated patterns.

Run with the environment that has orix:
    venv/bin/python 10_orientation_map/01_make_microstructure.py
"""

import os

import matplotlib
matplotlib.use("Agg")          # no display under WSL; save figures instead
import matplotlib.pyplot as plt
import numpy as np
from orix.quaternion import Rotation, Orientation
from orix.quaternion.symmetry import Oh          # m-3m, nickel's point group
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d

# ======================================================================
# SETTINGS
# ======================================================================
MAP_NY, MAP_NX = 200, 200      # 200 x 200 = 40,000 pixels
N_GRAINS       = 400           # ~100 pixels per grain -> ~11 pixels across
SPREAD_DEG     = 0.3           # gentle orientation drift inside each grain
SEED           = 4242

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
ANGLE_FILE = os.path.join(OUT_DIR, "map_angles_40k.txt")
TRUTH_FILE = os.path.join(OUT_DIR, "map_ground_truth.npz")
# ======================================================================

N_PIXELS = MAP_NY * MAP_NX
rng = np.random.default_rng(SEED)
np.random.seed(SEED)           # orix's Rotation.random() draws from numpy's global stream


# ----------------------------------------------------------------------
# 1. Cut the map up into grains
# ----------------------------------------------------------------------
# Scatter N_GRAINS seed points at random, then hand every pixel to whichever
# seed is closest. Each seed's territory becomes one grain. Crude, but it is
# the standard way to fake a polycrystal and the grain shapes look plausible.
seeds_y = rng.uniform(0, MAP_NY, N_GRAINS)
seeds_x = rng.uniform(0, MAP_NX, N_GRAINS)

yy, xx = np.indices((MAP_NY, MAP_NX))
d2 = (yy[..., None] - seeds_y) ** 2 + (xx[..., None] - seeds_x) ** 2
grain_id = np.argmin(d2, axis=-1)                      # (MAP_NY, MAP_NX)

sizes = np.bincount(grain_id.ravel(), minlength=N_GRAINS)
print(f"map          : {MAP_NY} x {MAP_NX} = {N_PIXELS} pixels")
print(f"grains       : {N_GRAINS}")
print(f"grain size   : {sizes.min()} to {sizes.max()} pixels, mean {sizes.mean():.0f}")


# ----------------------------------------------------------------------
# 2. Give each GRAIN one orientation, then copy it to that grain's pixels
# ----------------------------------------------------------------------
# This one line is the whole difference from the stage-08 generator:
# N_GRAINS random draws, not N_PIXELS. The duplication that follows is
# deliberate -- it is what makes neighbours agree.
grain_rot = Rotation.random(N_GRAINS)
pixel_rot = grain_rot[grain_id.ravel()]                # row-major: row 0 left->right, then row 1, ...


# ----------------------------------------------------------------------
# 3. Add a little drift inside each grain
# ----------------------------------------------------------------------
# Real grains are not perfectly uniform -- the lattice bends slightly, and the
# camera adds its own scatter. In the real 4125-point scan, neighbouring pixels
# inside a grain differ by about 0.4 degrees rather than exactly 0. Without this
# the map is suspiciously perfect.
axes = rng.normal(size=(N_PIXELS, 3))
axes /= np.linalg.norm(axes, axis=1, keepdims=True)
angles = np.deg2rad(rng.normal(0.0, SPREAD_DEG, N_PIXELS))
wobble = Rotation.from_axes_angles(Vector3d(axes), angles)
pixel_rot = pixel_rot * wobble


# ----------------------------------------------------------------------
# 4. Convert to Euler angles and write the EMsoft angle file
# ----------------------------------------------------------------------
euler = np.asarray(pixel_rot.to_euler()).reshape(-1, 3)

# orix has returned radians historically, but this has moved between versions,
# so detect rather than assume (same guard as the stage-08 generator).
if np.nanmax(np.abs(euler)) <= 2 * np.pi + 1e-6:
    print("converting radians to degrees")
    euler = np.rad2deg(euler)

euler[:, 0] %= 360.0
euler[:, 1] %= 180.0
euler[:, 2] %= 360.0

with open(ANGLE_FILE, "w") as f:
    f.write("eu\n")
    f.write(f"{len(euler)}\n")
    for phi1, PHI, phi2 in euler:
        f.write(f"{phi1:12.6f} {PHI:12.6f} {phi2:12.6f}\n")

print(f"\nwrote {ANGLE_FILE}  ({len(euler)} orientations)")
print("     line order is row-major: pixel (row, col) is on line 3 + row*MAP_NX + col")


# ----------------------------------------------------------------------
# 5. Save the answer key, so the map can be checked later
# ----------------------------------------------------------------------
np.savez_compressed(
    TRUTH_FILE,
    euler_deg=euler,                     # (N_PIXELS, 3), same order as the angle file
    grain_id=grain_id,                   # (MAP_NY, MAP_NX)
    map_shape=np.array([MAP_NY, MAP_NX]),
)
print(f"wrote {TRUTH_FILE}")


# ----------------------------------------------------------------------
# 6. Sanity check: do neighbours actually agree?
# ----------------------------------------------------------------------
# The number to watch. Inside a grain it should be well under 1 degree; across
# a boundary, tens of degrees. If the median comes out near 40 degrees, the
# grain structure did not survive and the map is just noise.
o = Orientation.from_euler(np.deg2rad(euler), symmetry=Oh).reshape(MAP_NY, MAP_NX)
right = o[:, :-1].angle_with(o[:, 1:], degrees=True).ravel()
down = o[:-1, :].angle_with(o[1:, :], degrees=True).ravel()
neigh = np.concatenate([right, down])

print(f"\nneighbour disorientation: median {np.median(neigh):.2f} deg, "
      f"{np.mean(neigh < 1):.1%} under 1 deg, {np.mean(neigh > 10):.1%} over 10 deg")
print("  (real 4125-point scan, for comparison: median 0.38 deg, 83% under 1 deg)")


# ----------------------------------------------------------------------
# 7. Picture of the answer key
# ----------------------------------------------------------------------
ckey = IPFColorKeyTSL(Oh)
rgb = ckey.orientation2color(o.flatten()).reshape(MAP_NY, MAP_NX, 3)

fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
ax[0].imshow(rgb)
ax[0].set_title(f"ground-truth orientation map\n{MAP_NY}x{MAP_NX}, {N_GRAINS} grains")
ax[1].imshow(grain_id, cmap="tab20", interpolation="nearest")
ax[1].set_title("grain ID (which grain each pixel belongs to)")
for a in ax:
    a.set_xlabel("x (pixels)")
    a.set_ylabel("y (pixels)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "map_ground_truth.png"), dpi=150)
print(f"\nwrote {os.path.join(OUT_DIR, 'map_ground_truth.png')}")
print("\nNext: run EMEBSD_map.nml (STEP 2).")
