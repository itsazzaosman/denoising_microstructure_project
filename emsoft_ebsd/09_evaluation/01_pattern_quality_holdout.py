#!/usr/bin/env python3
"""
Stage 1 -- pattern quality on held-out simulation.

The model trained in 08_synthetic_training_data/train_autoencoder_synthetic.py
only ever saw 30,000 patterns at synthetic, randomly-sampled orientations.
This script scores it against a completely different dataset it has never
touched: the 4,125 patterns simulated at the real scan's orientations
(05_noising_training_data/training_pairs.h5) -- a different EMEBSD run, a
different angle file, a different noise draw.

Answers: did the model generalise to orientations it never trained on, or
did it just memorise the 30k synthetic set? Pass --tag/--holdout-file to
run this same check against a different noise level (e.g. a harsher one
than the model trained on) without overwriting the original results.

Run:
    ~/.dl/bin/python 01_pattern_quality_holdout.py
    ~/.dl/bin/python 01_pattern_quality_holdout.py --tag harsh \\
        --holdout-file ../05_noising_training_data/training_pairs_harsh.h5
"""

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")          # no display under WSL; save figures instead
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ======================================================================
# SETTINGS
# ======================================================================
# Paths are relative to this file's position in the repo, so this script
# runs unchanged wherever the repo lives (local machine, cluster, ...).
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(OUT_DIR)

cli = argparse.ArgumentParser()
cli.add_argument("--model-file", default=os.path.join(
    REPO_ROOT, "08_synthetic_training_data", "denoiser_30.keras"))
cli.add_argument("--holdout-file", default=os.path.join(
    REPO_ROOT, "05_noising_training_data", "training_pairs.h5"))
cli.add_argument("--tag", default="",
                  help="suffix for output filenames, e.g. 'harsh' -> stage1_..._harsh.*; "
                       "empty (default) reproduces the original, unsuffixed filenames")
args = cli.parse_args()

MODEL_FILE   = args.model_file
HOLDOUT_FILE = args.holdout_file
SUFFIX       = f"_{args.tag}" if args.tag else ""

BATCH_SIZE  = 64
EXAMPLE_IDX = 0     # which holdout pattern to show in the figure
# ======================================================================


def psnr(a, b):
    """Peak signal-to-noise ratio, averaged over patterns. Higher is better."""
    mse = np.mean((a - b) ** 2, axis=(1, 2))
    return float(np.mean(10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))))


def ncc(a, b):
    """Normalised cross-correlation, averaged over patterns. 1.0 is identical."""
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return float(np.mean((a * b).mean(axis=1) /
                         np.maximum(a.std(axis=1) * b.std(axis=1), 1e-12)))


# ----------------------------------------------------------------------
# 1. Load the trained model
# ----------------------------------------------------------------------
print(f"Loading model: {MODEL_FILE}")
autoencoder = tf.keras.models.load_model(MODEL_FILE)
_, model_h, model_w, _ = autoencoder.input_shape
print(f"  expects input {model_h} x {model_w} x 1")


# ----------------------------------------------------------------------
# 2. Load the held-out set -- 4,125 patterns at real-scan orientations,
#    never used anywhere in this model's training
# ----------------------------------------------------------------------
print(f"\nLoading holdout set: {HOLDOUT_FILE}")
with h5py.File(HOLDOUT_FILE, "r") as f:
    clean = f["clean"][:]      # (N, 60, 60) float32 in [0, 1]
    noisy = f["noisy"][:]      # (N, 60, 60) float32 in [0, 1]
    meta  = dict(f.attrs)

n_patterns, height, width = clean.shape
print(f"  {n_patterns} pairs of {height} x {width}")
print(f"  noising mode: {meta.get('mode')}, SNR {float(meta.get('snr', 0)):.3f}")
print("  this file's own train/val split is irrelevant here -- the model")
print("  trained on a different file entirely, so every pattern below is unseen")


# ----------------------------------------------------------------------
# 3. Verify preprocessing matches what the model was trained on
# ----------------------------------------------------------------------
assert clean.dtype == np.float32 and noisy.dtype == np.float32, \
    f"expected float32, got clean={clean.dtype}, noisy={noisy.dtype}"
assert clean.min() >= -1e-6 and clean.max() <= 1 + 1e-6, \
    f"clean out of [0, 1]: [{clean.min():.3f}, {clean.max():.3f}]"
assert noisy.min() >= -1e-6 and noisy.max() <= 1 + 1e-6, \
    f"noisy out of [0, 1]: [{noisy.min():.3f}, {noisy.max():.3f}]"
assert (height, width) == (model_h, model_w), \
    f"holdout patterns are {height}x{width}, model expects {model_h}x{model_w}"

x_holdout = noisy[..., None]   # (N, H, W) -> (N, H, W, 1)
y_holdout = clean[..., None]
print(f"  preprocessing OK: {x_holdout.dtype}, shape {x_holdout.shape}, "
      f"range [{x_holdout.min():.3f}, {x_holdout.max():.3f}]")


# ----------------------------------------------------------------------
# 4. Predict
# ----------------------------------------------------------------------
print(f"\nDenoising {n_patterns} holdout patterns...")
denoised = autoencoder.predict(x_holdout, batch_size=BATCH_SIZE, verbose=0)[..., 0]
target   = y_holdout[..., 0]
noisy_in = x_holdout[..., 0]


# ----------------------------------------------------------------------
# 5 & 6. PSNR / NCC for noisy vs. target and denoised vs. target,
#        and the improvement between them
# ----------------------------------------------------------------------
psnr_noisy,    ncc_noisy    = psnr(noisy_in, target), ncc(noisy_in, target)
psnr_denoised, ncc_denoised = psnr(denoised, target), ncc(denoised, target)
psnr_gain = psnr_denoised - psnr_noisy
ncc_gain  = ncc_denoised - ncc_noisy

print("\nHeld-out simulation, measured against the clean target:")
print(f"  noisy input : PSNR {psnr_noisy:6.2f} dB   NCC {ncc_noisy:.4f}")
print(f"  denoised    : PSNR {psnr_denoised:6.2f} dB   NCC {ncc_denoised:.4f}")
print(f"  improvement : PSNR {psnr_gain:+6.2f} dB   NCC {ncc_gain:+.4f}")

generalised = psnr_gain > 0 and ncc_gain > 0
print(f"\nGeneralised to unseen, real-derived orientations? "
      f"{'YES' if generalised else 'NO'} -- denoised {'beats' if generalised else 'does not beat'} "
      f"noisy on both metrics")

results = {
    "model_file": MODEL_FILE,
    "holdout_file": HOLDOUT_FILE,
    "n_patterns": int(n_patterns),
    "psnr_noisy_db": psnr_noisy,
    "psnr_denoised_db": psnr_denoised,
    "psnr_gain_db": psnr_gain,
    "ncc_noisy": ncc_noisy,
    "ncc_denoised": ncc_denoised,
    "ncc_gain": ncc_gain,
    "generalised": generalised,
}
results_path = f"{OUT_DIR}/stage1_pattern_quality_holdout{SUFFIX}.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {os.path.basename(results_path)}")

# Persist the three full pattern sets so Stage 2 (indexing accuracy) can
# Hough-index them without needing TensorFlow -- that script runs in the
# separate .ebsd environment (pyebsdindex/kikuchipy/orix only).
patterns_path = f"{OUT_DIR}/stage1_denoised_patterns{SUFFIX}.h5"
with h5py.File(patterns_path, "w") as f:
    f.create_dataset("noisy", data=noisy_in, compression="gzip")
    f.create_dataset("denoised", data=denoised, compression="gzip")
    f.create_dataset("clean", data=target, compression="gzip")
print(f"saved {os.path.basename(patterns_path)}")


# ----------------------------------------------------------------------
# 7. One-example figure: noisy, denoised, clean
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
ax[0].imshow(noisy_in[EXAMPLE_IDX], cmap="gray", vmin=0, vmax=1)
ax[0].set_title("noisy input", fontsize=10)
ax[1].imshow(denoised[EXAMPLE_IDX], cmap="gray", vmin=0, vmax=1)
ax[1].set_title("denoised output", fontsize=10)
ax[2].imshow(target[EXAMPLE_IDX], cmap="gray", vmin=0, vmax=1)
ax[2].set_title("clean target", fontsize=10)
for a in ax:
    a.axis("off")
fig.suptitle(f"Holdout pattern {EXAMPLE_IDX} -- real-derived orientation, unseen in training"
             + (f" [{args.tag}]" if args.tag else ""),
             fontsize=10)
plt.tight_layout()
example_path = f"{OUT_DIR}/stage1_holdout_example{SUFFIX}.png"
plt.savefig(example_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {os.path.basename(example_path)}")
