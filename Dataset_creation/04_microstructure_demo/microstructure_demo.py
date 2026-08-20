#!/usr/bin/env python3
"""
Synthetic microstructure -> simulated EBSD patterns -> dictionary indexing -> IPF grain map

Demonstrates the full EBSD loop using an EMsoft master pattern:

  1. Invent a polycrystal (Voronoi grains, random orientation per grain)
  2. Simulate one EBSD pattern per map pixel from the master pattern
  3. Add noise so it resembles real data
  4. Recover the orientations by dictionary indexing
  5. Plot ground-truth and recovered grain maps side by side

Run from the folder containing your EMsoft master pattern file.

    /usr/bin/python3 microstructure_demo.py

Dependencies:
    pip install kikuchipy orix
    (numpy, matplotlib, dask come along with it)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# SETTINGS  -- start small, increase once it works
# ----------------------------------------------------------------------
MASTER_FILE = "../02_master_pattern/Ni_master_hires.h5"

MAP_NX, MAP_NY = 40, 40      # map size in pixels -> 1600 patterns
N_GRAINS       = 12          # number of grains to generate
DET_SHAPE      = (50, 50)    # detector resolution per simulated pattern
ENERGY         = 20          # keV, must exist in your master pattern
DI_RESOLUTION  = 4.0         # degrees; SMALLER = bigger dictionary = slower
NOISE_LEVEL    = 0.05        # fraction of pattern std added as Gaussian noise
SEED           = 42

# Detector geometry -- keep consistent between the "experiment" and dictionary
SAMPLE_TILT = 70.0
DET_TILT    = 10.0
PC          = (0.5, 0.5, 0.5)   # pattern centre, Bruker convention

# ----------------------------------------------------------------------

try:
    import kikuchipy as kp
    from orix.quaternion import Rotation
    from orix.sampling import get_sample_fundamental
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n\nInstall with:\n    pip install kikuchipy orix")

rng = np.random.default_rng(SEED)


# ======================================================================
# STEP 1 -- Build a synthetic microstructure
# ======================================================================
def make_voronoi_grains(nx, ny, n_grains, rng):
    """Assign every pixel to its nearest seed point. Produces convex grains
    with straight boundaries -- a crude but standard synthetic microstructure."""
    seeds_x = rng.uniform(0, nx, n_grains)
    seeds_y = rng.uniform(0, ny, n_grains)

    yy, xx = np.indices((ny, nx))
    d2 = (xx[..., None] - seeds_x) ** 2 + (yy[..., None] - seeds_y) ** 2
    return np.argmin(d2, axis=-1)          # (ny, nx) grain id per pixel


print("Step 1: building synthetic microstructure")
grain_id = make_voronoi_grains(MAP_NX, MAP_NY, N_GRAINS, rng)
print(f"  {MAP_NY} x {MAP_NX} pixels, {N_GRAINS} grains")

# One random orientation per grain, then expand to per-pixel
grain_rots = Rotation.random(N_GRAINS)
rot_true = grain_rots[grain_id.ravel()]
print(f"  {rot_true.size} pixel orientations")


# ======================================================================
# STEP 2 -- Load the EMsoft master pattern
# ======================================================================
print(f"\nStep 2: loading master pattern {MASTER_FILE}")
try:
    mp = kp.load(
        MASTER_FILE,
        energy=ENERGY,
        projection="lambert",
        hemisphere="upper",
    )
except Exception as e:
    sys.exit(
        f"Could not load master pattern: {e}\n\n"
        "Check the filename, and that the file was produced by EMEBSDmaster.\n"
        "Inspect its structure with:\n"
        f"    h5ls -r {MASTER_FILE}"
    )

phase = mp.phase
print(f"  phase: {phase.name}, point group: {phase.point_group.name}")

detector = kp.detectors.EBSDDetector(
    shape=DET_SHAPE,
    pc=PC,
    sample_tilt=SAMPLE_TILT,
    tilt=DET_TILT,
    convention="bruker",
)
print(f"  detector: {DET_SHAPE[0]}x{DET_SHAPE[1]} px, sample tilt {SAMPLE_TILT}deg")


# ======================================================================
# STEP 3 -- Simulate the "experimental" patterns
# ======================================================================
print("\nStep 3: simulating one pattern per pixel (this is the slow part)")
sim = mp.get_patterns(
    rotations=rot_true,
    detector=detector,
    energy=ENERGY,
    compute=True,
)

pats = np.asarray(sim.data, dtype=np.float32)
pats = pats.reshape(MAP_NY, MAP_NX, *DET_SHAPE)

# Add noise so dictionary indexing has something realistic to chew on
pats += rng.normal(0.0, pats.std() * NOISE_LEVEL, pats.shape).astype(np.float32)

s = kp.signals.EBSD(pats)
s.rescale_intensity(dtype_out=np.float32)
print(f"  simulated {MAP_NY * MAP_NX} patterns of {DET_SHAPE}")

# Save a few example patterns so you can see what was "measured"
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for i, (r, c) in enumerate([(5, 5), (5, 30), (30, 5), (30, 30)]):
    ax[i].imshow(pats[r, c], cmap="gray")
    ax[i].set_title(f"pixel ({r}, {c}), grain {grain_id[r, c]}")
    ax[i].axis("off")
plt.tight_layout()
plt.savefig("micro_example_patterns.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("  wrote micro_example_patterns.png")


# ======================================================================
# STEP 4 -- Dictionary indexing: recover orientations from patterns
# ======================================================================
print(f"\nStep 4: building dictionary at {DI_RESOLUTION} deg resolution")
rot_dict = get_sample_fundamental(
    resolution=DI_RESOLUTION,
    point_group=phase.point_group,
)
print(f"  {rot_dict.size} candidate orientations")

sim_dict = mp.get_patterns(
    rotations=rot_dict,
    detector=detector,
    energy=ENERGY,
    compute=True,
)

print("  indexing (comparing every pattern against every candidate)...")
xmap = s.dictionary_indexing(sim_dict, metric="ncc", keep_n=1)
print(f"  done. mean correlation score: {xmap.scores.mean():.3f}")


# ======================================================================
# STEP 5 -- Colour by orientation and plot
# ======================================================================
print("\nStep 5: generating IPF colour maps")

ckey = IPFColorKeyTSL(phase.point_group, direction=Vector3d.zvector())

rgb_true = ckey.orientation2color(rot_true).reshape(MAP_NY, MAP_NX, 3)
rgb_found = ckey.orientation2color(xmap.rotations.reshape(-1)).reshape(MAP_NY, MAP_NX, 3)
scores = xmap.scores.reshape(MAP_NY, MAP_NX)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].imshow(rgb_true)
ax[0].set_title("Ground truth\n(orientations we invented)")
ax[0].axis("off")

ax[1].imshow(rgb_found)
ax[1].set_title(f"Recovered by dictionary indexing\n({DI_RESOLUTION} deg dictionary)")
ax[1].axis("off")

im = ax[2].imshow(scores, cmap="viridis")
ax[2].set_title("Match quality (NCC)\nlow at grain boundaries")
ax[2].axis("off")
plt.colorbar(im, ax=ax[2], fraction=0.046)

plt.tight_layout()
plt.savefig("micro_grain_map.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  wrote micro_grain_map.png")

# The IPF colour key itself -- explains what the colours mean
fig_key = ckey.plot(return_figure=True)
fig_key.savefig("micro_ipf_colorkey.png", dpi=150, bbox_inches="tight")
plt.close(fig_key)
print("  wrote micro_ipf_colorkey.png")

print("\nDone. Open the PNGs with:  explorer.exe .")