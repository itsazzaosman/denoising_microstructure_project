#!/usr/bin/env python3
"""
Stage 2 -- indexing accuracy, the paper's actual claim.

Stage 1 showed the denoiser improves PSNR/NCC against a simulated clean
target. That's necessary but not the real claim: Andrews et al. (2023)
argue denoising improves *indexing*. This script tests that directly by
Hough-indexing the same three pattern sets from Stage 1 -- noisy,
denoised, clean -- and comparing each result's orientations against the
known ground truth by symmetry-aware disorientation.

Requires pyebsdindex (`pip install pyebsdindex`) on top of kikuchipy and
orix -- kept in a separate ".ebsd" environment from the TensorFlow one
used for training/prediction, which is also why this script reads
Stage 1's saved pattern arrays instead of loading the Keras model itself.

Run:
    ~/.ebsd/bin/python 02_indexing_accuracy.py

Answers: does denoising improve indexing, and by how much? Success looks
like the "result" (denoised) map sitting between "baseline" (noisy) and
"upper bound" (clean).
"""

import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")          # no display under WSL; save figures instead
import matplotlib.pyplot as plt
import numpy as np
import kikuchipy as kp
from diffpy.structure import Lattice, Structure
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Orientation
from orix.quaternion.symmetry import Oh          # m-3m
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d

# ======================================================================
# SETTINGS
# ======================================================================
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(OUT_DIR)
PATTERNS_FILE = f"{OUT_DIR}/stage1_denoised_patterns.h5"      # noisy/denoised/clean, from Stage 1
ANGLES_FILE   = os.path.join(REPO_ROOT, "03_ebsd_pattern_simulation", "ni_angles.txt")

SCAN_SHAPE = (55, 75)     # (ny, nx) -- must match the order patterns/angles were written in

DETECTOR_SHAPE = (60, 60)
PC             = (0.423, 0.214, 0.502)
SAMPLE_TILT    = 70.0
DETECTOR_TILT  = 0.0

PHASE_NAME       = "nickel"
SPACE_GROUP      = 225
LATTICE_A_NM     = 0.3524
LATTICE_A_ANGSTROM = LATTICE_A_NM * 10   # diffpy.structure/orix expect angstrom, not nm

CHUNKSIZE = 528     # patterns indexed at a time; kikuchipy default
# ======================================================================


ny, nx = SCAN_SHAPE
h, w = DETECTOR_SHAPE


# ----------------------------------------------------------------------
# 1. pip install pyebsdindex -- checked here so the error is obvious
#    rather than surfacing deep inside kikuchipy's hough_indexing()
# ----------------------------------------------------------------------
try:
    import pyebsdindex  # noqa: F401
except ImportError:
    raise SystemExit(
        "pyebsdindex is not installed in this environment.\n"
        "    pip install pyebsdindex"
    )
print(f"pyebsdindex {pyebsdindex.__version__}")


# ----------------------------------------------------------------------
# 2. Reshape all three pattern sets onto the scan grid
# ----------------------------------------------------------------------
print(f"\nLoading {PATTERNS_FILE}")
with h5py.File(PATTERNS_FILE, "r") as f:
    noisy    = f["noisy"][:]
    denoised = f["denoised"][:]
    clean    = f["clean"][:]

n_patterns = noisy.shape[0]
assert n_patterns == ny * nx, (
    f"{n_patterns} patterns but scan grid is {ny}x{nx}={ny * nx} -- "
    "SCAN_SHAPE doesn't match this dataset"
)

patterns = {
    "baseline (noisy)":    noisy.reshape(ny, nx, h, w),
    "result (denoised)":   denoised.reshape(ny, nx, h, w),
    "upper bound (clean)": clean.reshape(ny, nx, h, w),
}
print(f"  reshaped {n_patterns} patterns of {h}x{w} to {ny}x{nx} scan grid, all three sets")


# ----------------------------------------------------------------------
# 3. Build the detector
# ----------------------------------------------------------------------
detector = kp.detectors.EBSDDetector(
    shape=DETECTOR_SHAPE,
    pc=PC,
    sample_tilt=SAMPLE_TILT,
    tilt=DETECTOR_TILT,
)
print(f"\nDetector: {detector}")


# ----------------------------------------------------------------------
# 4. Define the phase
# ----------------------------------------------------------------------
phase = Phase(
    name=PHASE_NAME,
    space_group=SPACE_GROUP,
    structure=Structure(lattice=Lattice(
        LATTICE_A_ANGSTROM, LATTICE_A_ANGSTROM, LATTICE_A_ANGSTROM, 90, 90, 90,
    )),
)
phase_list = PhaseList(phase)
print(f"Phase: {phase_list}")

indexer = detector.get_indexer(phase_list)


# ----------------------------------------------------------------------
# 5-7. Hough-index all three sets
# ----------------------------------------------------------------------
xmaps = {}
for label, pats in patterns.items():
    print(f"\nHough-indexing {label} ({n_patterns} patterns)...")
    s = kp.signals.EBSD(pats)
    xmaps[label] = s.hough_indexing(
        phase_list=phase_list, indexer=indexer, chunksize=CHUNKSIZE, verbose=1,
    )


# ----------------------------------------------------------------------
# 8. Load ground truth
# ----------------------------------------------------------------------
print(f"\nLoading ground truth: {ANGLES_FILE}")
gt_euler = np.loadtxt(ANGLES_FILE, skiprows=2)     # (N, 3) phi1, PHI, phi2 in degrees
assert len(gt_euler) == n_patterns, \
    f"{len(gt_euler)} ground-truth angles but {n_patterns} patterns"
gt_orientation = Orientation.from_euler(gt_euler, symmetry=Oh, degrees=True)
print(f"  {gt_orientation.size} ground-truth orientations, symmetry {gt_orientation.symmetry.name}")


# ----------------------------------------------------------------------
# 9-10. Symmetry-aware disorientation from truth, per map, plus Hough's
#        own fit and confidence index (CI)
# ----------------------------------------------------------------------
report = {}
disorientation_maps = {}   # label -> (N,) degrees, kept for the figures below
for label, xmap in xmaps.items():
    orientations = xmap.orientations       # Orientation, symmetry already m-3m from the phase
    disorientation = gt_orientation.angle_with(orientations, degrees=True)
    disorientation = np.asarray(disorientation).ravel()
    disorientation_maps[label] = disorientation

    fit = np.asarray(xmap.prop["fit"]).ravel()
    cm  = np.asarray(xmap.prop["cm"]).ravel()

    stats = {
        "median_disorientation_deg": float(np.nanmedian(disorientation)),
        "mean_disorientation_deg":   float(np.nanmean(disorientation)),
        "fraction_under_1deg":       float(np.mean(disorientation < 1.0)),
        "fraction_under_2deg":       float(np.mean(disorientation < 2.0)),
        "hough_fit_mean":            float(np.nanmean(fit)),
        "hough_cm_mean":             float(np.nanmean(cm)),
    }
    report[label] = stats
    print(f"\n{label}:")
    print(f"  median disorientation : {stats['median_disorientation_deg']:.3f} deg")
    print(f"  mean disorientation   : {stats['mean_disorientation_deg']:.3f} deg")
    print(f"  fraction < 1 deg      : {stats['fraction_under_1deg']:.3f}")
    print(f"  fraction < 2 deg      : {stats['fraction_under_2deg']:.3f}")
    print(f"  Hough fit (mean)      : {stats['hough_fit_mean']:.3f}")
    print(f"  Hough CI (mean)       : {stats['hough_cm_mean']:.3f}")

baseline_med = report["baseline (noisy)"]["median_disorientation_deg"]
result_med   = report["result (denoised)"]["median_disorientation_deg"]
upper_med    = report["upper bound (clean)"]["median_disorientation_deg"]
improved = result_med < baseline_med
between  = upper_med <= result_med <= baseline_med

print(f"\nDoes denoising improve indexing? "
      f"{'YES' if improved else 'NO'} "
      f"(median disorientation {baseline_med:.3f} deg -> {result_med:.3f} deg, "
      f"upper bound {upper_med:.3f} deg)")
print(f"Result sits between baseline and upper bound? {'YES' if between else 'NO'}")

report["summary"] = {
    "improved": improved,
    "result_between_baseline_and_upper_bound": between,
    "median_disorientation_gain_deg": baseline_med - result_med,
}
with open(f"{OUT_DIR}/stage2_indexing_accuracy.json", "w") as f:
    json.dump(report, f, indent=2)
print("saved stage2_indexing_accuracy.json")


# ----------------------------------------------------------------------
# 11. Plot the three IPF maps side by side with ground truth
# ----------------------------------------------------------------------
ipf_key = IPFColorKeyTSL(Oh, direction=Vector3d.zvector())

panels = [("ground truth", gt_orientation)] + [
    (label, xmap.orientations) for label, xmap in xmaps.items()
]

fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
for a, (label, orientation) in zip(ax, panels):
    rgb = ipf_key.orientation2color(orientation).reshape(ny, nx, 3)
    a.imshow(np.clip(rgb, 0, 1))
    a.set_title(label, fontsize=11)
    a.axis("off")
fig.suptitle("IPF-Z maps: ground truth vs. Hough-indexed noisy / denoised / clean", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage2_ipf_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved stage2_ipf_maps.png")


# ----------------------------------------------------------------------
# 12. Disorientation heatmaps -- same scan, three runs, colored by how
#     WRONG each point is rather than by its orientation. This is what
#     actually differs visually: the IPF panels above look identical
#     because all three runs already agree with truth to a fraction of a
#     degree, far below what IPF coloring can show; a magnitude scale
#     zoomed into that same fraction-of-a-degree range can.
# ----------------------------------------------------------------------
vmax = max(d.max() for d in disorientation_maps.values())
print(f"\nShared disorientation color scale: 0 to {vmax:.2f} deg")

fig, ax = plt.subplots(1, len(disorientation_maps), figsize=(4.6 * len(disorientation_maps), 4.4))
im = None
for a, (label, disorientation) in zip(ax, disorientation_maps.items()):
    im = a.imshow(disorientation.reshape(ny, nx), cmap="viridis", vmin=0, vmax=vmax)
    a.set_title(f"{label}\nmedian {np.median(disorientation):.3f}°", fontsize=11)
    a.axis("off")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("disorientation from truth (deg)")
fig.suptitle("Per-point indexing error, same color scale across all three", fontsize=12)
plt.savefig(f"{OUT_DIR}/stage2_disorientation_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved stage2_disorientation_heatmaps.png")


# ----------------------------------------------------------------------
# 13. Disorientation CDF -- the standard indexing-accuracy figure: for
#     every possible threshold, what fraction of points are indexed at
#     or below it. This is what "result sits between baseline and upper
#     bound" actually looks like as a picture.
# ----------------------------------------------------------------------
LINE_STYLE = {
    "baseline (noisy)":    dict(color="#E69F00", linestyle="-",  linewidth=1.8),  # Okabe-Ito orange
    "result (denoised)":   dict(color="#0072B2", linestyle="-",  linewidth=2.6),  # Okabe-Ito blue -- the finding
    "upper bound (clean)": dict(color="#7F7F7F", linestyle="--", linewidth=1.8),  # neutral, dashed = reference
}

fig, a = plt.subplots(figsize=(6.4, 4.8))
x_max = max(1.5, vmax)
xs = np.linspace(0, x_max, 400)
for label, disorientation in disorientation_maps.items():
    cdf = [np.mean(disorientation <= x) for x in xs]
    a.plot(xs, cdf, label=label, **LINE_STYLE[label])

for threshold in (1.0, 2.0):
    if threshold <= x_max:
        a.axvline(threshold, color="0.85", linewidth=1, zorder=0)
        a.text(threshold, 0.02, f"{threshold:g}°", color="0.5", fontsize=9,
               ha="center", va="bottom")

a.set_xlim(0, x_max)
a.set_ylim(0, 1.02)
a.set_xlabel("disorientation from truth (degrees)")
a.set_ylabel("fraction of points at or below")
a.set_title("Indexing accuracy vs. disorientation threshold")
a.legend(loc="lower right", frameon=False)
a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stage2_disorientation_cdf.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved stage2_disorientation_cdf.png")


# ----------------------------------------------------------------------
# Persist the raw per-pixel disorientation arrays, so these figures can
# be redrawn or restyled without rerunning Hough indexing.
# ----------------------------------------------------------------------
npz_keys = {label: label.replace(" ", "_").replace("(", "").replace(")", "")
            for label in disorientation_maps}
np.savez(
    f"{OUT_DIR}/stage2_disorientation_maps.npz",
    **{npz_keys[label]: arr for label, arr in disorientation_maps.items()},
)
print("saved stage2_disorientation_maps.npz")
