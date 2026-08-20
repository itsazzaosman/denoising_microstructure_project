#!/usr/bin/env python3
"""
STEP 3 of the synthetic-orientation pipeline.

Adds noise to the clean simulated patterns at a FIXED signal-to-noise ratio.

Why fixed rather than calibrated: calibration measured the noise level of the
real scan by differencing adjacent same-grain patterns. That gave SNR 1.27. Those
synthetic orientations have no real counterparts to difference, so we reuse the
number measured earlier rather than re-deriving it.

Run with the environment that has h5py and numpy:
    ~/.ebsd/bin/python 03_make_noisy_fixed.py
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# SETTINGS
# ======================================================================
BASE = "/home/azza_osman/EMsoftData_work"

SIM_FILE = f"{BASE}/08_synthetic_training_data/Ni_EBSD_sim_30k.h5"
OUT_FILE = f"{BASE}/08_synthetic_training_data/training_pairs_30k.h5"
FIG_FILE = f"{BASE}/08_synthetic_training_data/noise_check_30k.png"

TARGET_SNR    = 1.27     # measured from the real scan earlier
VAL_FRACTION  = 0.15
SEED          = 42
# ======================================================================

rng = np.random.default_rng(SEED)

print(f"Loading {SIM_FILE}")
with h5py.File(SIM_FILE, "r") as f:
    clean_raw = f["EMData/EBSD/EBSDPatterns"][:]
clean_raw = np.squeeze(clean_raw).astype(np.float32)
n_patterns, height, width = clean_raw.shape
print(f"  {n_patterns} patterns of {height} x {width}")
print(f"  range {clean_raw.min():.0f} to {clean_raw.max():.0f}")

# Standardise each pattern to zero mean and unit standard deviation, so that
# "what the pattern looks like" is separated from "how strong it is".
mean = clean_raw.mean(axis=(1, 2), keepdims=True)
std = clean_raw.std(axis=(1, 2), keepdims=True)
clean_z = (clean_raw - mean) / np.maximum(std, 1e-6)

# clean_z has unit standard deviation, so noise of std 1/SNR gives the target SNR
noise_level = 1.0 / TARGET_SNR
print(f"\nAdding Gaussian noise: std {noise_level:.4f}  (SNR {TARGET_SNR})")

noisy_z = clean_z + rng.normal(0.0, noise_level, clean_z.shape).astype(np.float32)


def to_unit(a):
    """Map each pattern to [0, 1], clipping the extreme tails."""
    lo = np.percentile(a, 0.5, axis=(1, 2), keepdims=True)
    hi = np.percentile(a, 99.5, axis=(1, 2), keepdims=True)
    return np.clip((a - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


clean_unit = to_unit(clean_z)
noisy_unit = to_unit(noisy_z)

print(f"  clean: mean {clean_unit.mean():.3f}, std {clean_unit.std():.3f}")
print(f"  noisy: mean {noisy_unit.mean():.3f}, std {noisy_unit.std():.3f}")

# ----------------------------------------------------------------------
# Train / validation split
# ----------------------------------------------------------------------
perm = rng.permutation(n_patterns)
n_val = int(round(VAL_FRACTION * n_patterns))
val_idx = np.sort(perm[:n_val])
train_idx = np.sort(perm[n_val:])
print(f"\nSplit: {len(train_idx)} train, {len(val_idx)} validation")

steps_per_epoch = int(np.ceil(len(train_idx) / 64))
print(f"  at batch 64 that is {steps_per_epoch} steps per epoch")
print(f"  10 epochs -> {steps_per_epoch * 10} gradient steps (paper: ~5800)")

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
print(f"\nWriting {OUT_FILE} (this takes a minute at this size)")
with h5py.File(OUT_FILE, "w") as f:
    f.create_dataset("clean", data=clean_unit, compression="gzip", compression_opts=4)
    f.create_dataset("noisy", data=noisy_unit, compression="gzip", compression_opts=4)
    f.create_dataset("train_idx", data=train_idx)
    f.create_dataset("val_idx", data=val_idx)
    f.attrs["mode"] = "fixed_snr"
    f.attrs["snr"] = TARGET_SNR
    f.attrs["noise_level"] = noise_level
    f.attrs["seed"] = SEED
    f.attrs["source_sim"] = SIM_FILE
    f.attrs["orientations"] = "uniform random SO(3)"
print("done")

# ----------------------------------------------------------------------
# Visual check
# ----------------------------------------------------------------------
picks = rng.choice(n_patterns, 4, replace=False)
fig, ax = plt.subplots(2, len(picks), figsize=(3.1 * len(picks), 6.4))
for column, index in enumerate(picks):
    ax[0, column].imshow(clean_unit[index], cmap="gray", vmin=0, vmax=1)
    ax[0, column].set_title(f"clean target\n#{index}", fontsize=9)
    ax[1, column].imshow(noisy_unit[index], cmap="gray", vmin=0, vmax=1)
    ax[1, column].set_title("noisy input", fontsize=9)
    for row in range(2):
        ax[row, column].axis("off")
fig.suptitle(f"Synthetic training pairs, SNR {TARGET_SNR}", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_FILE, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG_FILE}")