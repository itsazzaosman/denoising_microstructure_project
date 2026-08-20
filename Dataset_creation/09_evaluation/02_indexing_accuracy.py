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
    ~/.ebsd/bin/python 02_indexing_accuracy.py --tag harsh

Answers: does denoising improve indexing, and by how much? Success looks
like the "result" (denoised) map sitting between "baseline" (noisy) and
"upper bound" (clean).
"""

import argparse
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
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(OUT_DIR)

cli = argparse.ArgumentParser()
cli.add_argument("--tag", default="",
                  help="must match the --tag used in 01_pattern_quality_holdout.py, "
                       "e.g. 'harsh' -> reads stage1_denoised_patterns_harsh.h5 and "
                       "writes stage2_..._harsh.*; empty (default) reproduces the "
                       "original, unsuffixed filenames")
args = cli.parse_args()
SUFFIX = f"_{args.tag}" if args.tag else ""

PATTERNS_FILE = f"{OUT_DIR}/stage1_denoised_patterns{SUFFIX}.h5"   # noisy/denoised/clean, from Stage 1
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
disorientation_maps = {}   # label -> (N,) degrees, NaN where Hough failed to index at all
for label, xmap in xmaps.items():
    # At harsh noise levels, Hough indexing can fail outright on some points --
    # those get phase "not_indexed" and xmap.orientations (single-phase only)
    # would raise. Score them as NaN: correctly excluded by nanmedian/nanmean
    # below, and correctly counted as misses by the "< 1 deg"/"< 2 deg" and CDF
    # comparisons, since NaN comparisons are always False in numpy.
    indexed_mask = xmap.is_indexed
    fraction_indexed = float(indexed_mask.mean())

    disorientation = np.full(n_patterns, np.nan)
    if indexed_mask.any():
        disorientation[indexed_mask] = gt_orientation[indexed_mask].angle_with(
            xmap[indexed_mask].orientations, degrees=True)
    disorientation_maps[label] = disorientation

    fit = np.where(indexed_mask, np.asarray(xmap.prop["fit"]).ravel(), np.nan)
    cm  = np.where(indexed_mask, np.asarray(xmap.prop["cm"]).ravel(), np.nan)

    stats = {
        "fraction_indexed":          fraction_indexed,
        "median_disorientation_deg": float(np.nanmedian(disorientation)),
        "mean_disorientation_deg":   float(np.nanmean(disorientation)),
        "fraction_under_1deg":       float(np.mean(disorientation < 1.0)),
        "fraction_under_2deg":       float(np.mean(disorientation < 2.0)),
        "hough_fit_mean":            float(np.nanmean(fit)),
        "hough_cm_mean":             float(np.nanmean(cm)),
    }
    report[label] = stats
    print(f"\n{label}:")
    print(f"  fraction indexed      : {stats['fraction_indexed']:.3f}"
          + ("" if fraction_indexed == 1.0 else "  <-- some points failed to index at all"))
    print(f"  median disorientation : {stats['median_disorientation_deg']:.3f} deg (among indexed points)")
    print(f"  mean disorientation   : {stats['mean_disorientation_deg']:.3f} deg (among indexed points)")
    print(f"  fraction < 1 deg      : {stats['fraction_under_1deg']:.3f}  (of all points; not-indexed counts as a miss)")
    print(f"  fraction < 2 deg      : {stats['fraction_under_2deg']:.3f}  (of all points; not-indexed counts as a miss)")
    print(f"  Hough fit (mean)      : {stats['hough_fit_mean']:.3f}")
    print(f"  Hough CI (mean)       : {stats['hough_cm_mean']:.3f}")

baseline_med = report["baseline (noisy)"]["median_disorientation_deg"]
result_med   = report["result (denoised)"]["median_disorientation_deg"]
upper_med    = report["upper bound (clean)"]["median_disorientation_deg"]
improved = result_med < baseline_med
between  = upper_med <= result_med <= baseline_med

baseline_frac = report["baseline (noisy)"]["fraction_indexed"]
result_frac   = report["result (denoised)"]["fraction_indexed"]
upper_frac    = report["upper bound (clean)"]["fraction_indexed"]
indexed_more  = result_frac > baseline_frac

print(f"\nDoes denoising improve indexing? "
      f"{'YES' if improved else 'NO'} "
      f"(median disorientation {baseline_med:.3f} deg -> {result_med:.3f} deg, "
      f"upper bound {upper_med:.3f} deg)")
print(f"Result sits between baseline and upper bound? {'YES' if between else 'NO'}")
print(f"Fraction indexed at all: baseline {baseline_frac:.3f} -> "
      f"denoised {result_frac:.3f}  (upper bound {upper_frac:.3f})"
      + ("  <-- denoising rescues points that failed to index outright" if indexed_more else ""))

report["summary"] = {
    "improved": improved,
    "result_between_baseline_and_upper_bound": between,
    "median_disorientation_gain_deg": baseline_med - result_med,
    "fraction_indexed_gain": result_frac - baseline_frac,
}
report_path = f"{OUT_DIR}/stage2_indexing_accuracy{SUFFIX}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"saved {os.path.basename(report_path)}")


# ----------------------------------------------------------------------
# 11. Plot the three IPF maps side by side with ground truth
# ----------------------------------------------------------------------
ipf_key = IPFColorKeyTSL(Oh, direction=Vector3d.zvector())

# Not-indexed points have no orientation to color -- shown black, same
# convention as MTEX/OIM/DREAM.3D use for "no solution".
rgb_panels = [("ground truth", ipf_key.orientation2color(gt_orientation).reshape(ny, nx, 3))]
for label, xmap in xmaps.items():
    indexed_mask = xmap.is_indexed
    rgb_flat = np.zeros((n_patterns, 3))
    if indexed_mask.any():
        rgb_flat[indexed_mask] = ipf_key.orientation2color(xmap[indexed_mask].orientations)
    rgb_panels.append((label, rgb_flat.reshape(ny, nx, 3)))

fig, ax = plt.subplots(1, len(rgb_panels), figsize=(4 * len(rgb_panels), 4.4))
for a, (label, rgb) in zip(ax, rgb_panels):
    a.imshow(np.clip(rgb, 0, 1))
    a.set_title(label, fontsize=11)
    a.axis("off")
fig.suptitle("IPF-Z maps: ground truth vs. Hough-indexed noisy / denoised / clean"
             + (f" [{args.tag}]" if args.tag else ""), fontsize=12)
plt.tight_layout()
ipf_path = f"{OUT_DIR}/stage2_ipf_maps{SUFFIX}.png"
plt.savefig(ipf_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {os.path.basename(ipf_path)}")


# ----------------------------------------------------------------------
# 12. Disorientation heatmaps -- same scan, three runs, colored by how
#     WRONG each point is rather than by its orientation. This is what
#     actually differs visually: the IPF panels above look identical
#     because all three runs already agree with truth to a fraction of a
#     degree, far below what IPF coloring can show; a magnitude scale
#     zoomed into that same fraction-of-a-degree range can.
# ----------------------------------------------------------------------
vmax = max(np.nanmax(d) for d in disorientation_maps.values())
print(f"\nShared disorientation color scale: 0 to {vmax:.2f} deg")

# Black = failed to index at all, not "zero disorientation" -- distinct from
# the viridis scale itself so a bad point can never be mistaken for a good one.
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad(color="black")

any_failures = any(report[label]["fraction_indexed"] < 1.0 for label in disorientation_maps)

fig, ax = plt.subplots(1, len(disorientation_maps), figsize=(4.6 * len(disorientation_maps), 4.6),
                        constrained_layout=True)
im = None
for a, (label, disorientation) in zip(ax, disorientation_maps.items()):
    im = a.imshow(disorientation.reshape(ny, nx), cmap=cmap, vmin=0, vmax=vmax)
    title_lines = [label, f"median {np.nanmedian(disorientation):.3f}°"]
    a.set_title("\n".join(title_lines), fontsize=10.5, linespacing=1.5)
    a.axis("off")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("disorientation from truth (deg)")
subtitle = "black = failed to index" if any_failures else None
suptitle_text = "Per-point indexing error, same color scale across all three"
if args.tag:
    suptitle_text += f" [{args.tag}]"
fig.suptitle(suptitle_text + (f"\n({subtitle})" if subtitle else ""), fontsize=12)
heatmap_path = f"{OUT_DIR}/stage2_disorientation_heatmaps{SUFFIX}.png"
plt.savefig(heatmap_path, dpi=150)
plt.close(fig)
print(f"saved {os.path.basename(heatmap_path)}")


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
a.set_title("Indexing accuracy vs. disorientation threshold" + (f" [{args.tag}]" if args.tag else ""))
a.legend(loc="lower right", frameon=False)
a.spines["top"].set_visible(False)
a.spines["right"].set_visible(False)
plt.tight_layout()
cdf_path = f"{OUT_DIR}/stage2_disorientation_cdf{SUFFIX}.png"
plt.savefig(cdf_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {os.path.basename(cdf_path)}")


# ----------------------------------------------------------------------
# Persist the raw per-pixel disorientation arrays, so these figures can
# be redrawn or restyled without rerunning Hough indexing.
# ----------------------------------------------------------------------
npz_keys = {label: label.replace(" ", "_").replace("(", "").replace(")", "")
            for label in disorientation_maps}
npz_path = f"{OUT_DIR}/stage2_disorientation_maps{SUFFIX}.npz"
np.savez(
    npz_path,
    **{npz_keys[label]: arr for label, arr in disorientation_maps.items()},
)
print(f"saved {os.path.basename(npz_path)}")
