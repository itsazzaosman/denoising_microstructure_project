#!/usr/bin/env python3
"""
STEP 3 of the orientation-map pipeline -- run this after EMEBSD finishes.

Confirms the 40,000 patterns EMEBSD wrote really do line up with the map, by
checking the one thing that would silently ruin everything: pattern ordering.
If line N of the angle file did not become pattern N in the HDF5 file, the map
is scrambled and no later analysis can detect it.

The test: patterns from the same grain should look nearly identical, and
patterns from different grains should look clearly different. This script picks
one such pair of each and reports the correlation.

Run with the environment that has h5py:
    venv/bin/python 10_orientation_map/02_check_map.py
"""

import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
PATTERN_FILE = os.path.join(OUT_DIR, "Ni_EBSD_map_40k.h5")
TRUTH_FILE   = os.path.join(OUT_DIR, "map_ground_truth.npz")

truth = np.load(TRUTH_FILE)
MAP_NY, MAP_NX = truth["map_shape"]
grain_id = truth["grain_id"]

# ----------------------------------------------------------------------
# Find where EMEBSD put the patterns. The dataset name has moved between
# EMsoft versions, so search rather than hardcode.
# ----------------------------------------------------------------------
with h5py.File(PATTERN_FILE, "r") as f:
    found = []
    f.visititems(lambda name, obj: found.append((name, obj.shape))
                 if isinstance(obj, h5py.Dataset) and obj.ndim == 3 else None)
    if not found:
        raise SystemExit(f"no 3-D pattern dataset found in {PATTERN_FILE}\n"
                         f"inspect it with: h5ls -r {PATTERN_FILE}")
    path, shape = max(found, key=lambda t: np.prod(t[1]))
    print(f"pattern dataset : {path}  shape {shape}")
    pats = f[path][()]

n_pat = pats.shape[0]
print(f"patterns        : {n_pat}   expected {MAP_NY * MAP_NX}")
if n_pat != MAP_NY * MAP_NX:
    raise SystemExit("count mismatch -- EMEBSD did not simulate every pixel")

# Reshape flat list -> map. Row-major, matching how the angle file was written.
pmap = pats.reshape(MAP_NY, MAP_NX, *pats.shape[1:])
print(f"reshaped to     : {pmap.shape}  (y, x, detector rows, detector cols)")


# ----------------------------------------------------------------------
# The ordering test
# ----------------------------------------------------------------------
# Compare two populations of pattern pairs:
#   - pairs of neighbouring pixels that sit inside the SAME grain
#   - pairs of pixels picked at random from anywhere on the map
# If the ordering is right, the first group is strongly correlated and the
# second is not. If the patterns got scrambled, both groups look alike.
def ncc(a, b):
    """How similar two patterns are: 1.0 = identical, 0.0 = unrelated."""
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


rng = np.random.default_rng(0)
N_SAMPLE = 300

same, cross, rand = [], [], []
while len(same) < N_SAMPLE or len(cross) < N_SAMPLE:
    y, x = rng.integers(MAP_NY), rng.integers(MAP_NX - 1)
    score = ncc(pmap[y, x], pmap[y, x + 1])
    if grain_id[y, x] == grain_id[y, x + 1]:
        if len(same) < N_SAMPLE:
            same.append(score)                 # neighbours, inside one grain
    elif len(cross) < N_SAMPLE:
        cross.append(score)                    # neighbours, across a boundary

for _ in range(N_SAMPLE):
    rand.append(ncc(pmap[rng.integers(MAP_NY), rng.integers(MAP_NX)],
                    pmap[rng.integers(MAP_NY), rng.integers(MAP_NX)]))

same, cross, rand = map(np.asarray, (same, cross, rand))
print(f"\npattern similarity (1.0 = identical, 0.0 = unrelated):")
print(f"  neighbours in the same grain  : median {np.median(same):6.3f}")
print(f"  neighbours across a boundary  : median {np.median(cross):6.3f}")
print(f"  two pixels picked at random   : median {np.median(rand):6.3f}")

if np.median(same) > 0.5 and np.median(same) > np.median(rand) + 0.4:
    print("\nPASS -- ordering is correct. Patterns follow the grain structure: "
          "same-grain neighbours match, everything else does not.")
else:
    print("\nFAIL -- patterns do not follow the grain structure. The pattern "
          "order in the HDF5 file probably does not match the angle file.")

# Example pixels for the figure: two neighbours in the biggest grain, plus one
# pixel from a different grain.
biggest = np.bincount(grain_id.ravel()).argmax()
ys, xs = np.where(grain_id == biggest)
same_a = (int(ys[0]), int(xs[0]))
nb = [(ys[k], xs[k]) for k in range(len(ys))
      if xs[k] == same_a[1] + 1 and ys[k] == same_a[0]]
same_b = (int(same_b_[0]), int(same_b_[1])) if (same_b_ := (nb[0] if nb else (ys[1], xs[1]))) else None
oy, ox = np.where(grain_id != biggest)
diff = (int(oy[0]), int(ox[0]))

# ----------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
for a, (pt, lab) in zip(ax, [
    (same_a, f"grain {biggest} at {same_a}"),
    (same_b, f"grain {biggest} at {same_b}\n(same grain -> nearly identical)"),
    (diff,   f"grain {grain_id[diff]} at {diff}\n(other grain -> looks different)"),
]):
    a.imshow(pmap[pt], cmap="gray")
    a.set_title(lab, fontsize=9)
    a.axis("off")

# A quick "is there structure in space" view: mean brightness per pixel. Grain
# outlines showing up here is independent proof the map is not scrambled.
ax[3].imshow(pmap.reshape(MAP_NY, MAP_NX, -1).mean(axis=2), cmap="gray")
ax[3].set_title("mean pattern brightness per map pixel\n(grains should be visible)", fontsize=9)
ax[3].axis("off")

fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(OUT_DIR, "map_check.png")
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
