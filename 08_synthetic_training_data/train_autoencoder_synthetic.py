#!/usr/bin/env python3
"""
Train the denoising autoencoder.

Replicates Andrews et al. (2023), Ultramicroscopy 253, 113810.

Architecture is verbatim from the authors' repository
(caldrews/ebsd_denoising_autoencoder, test_model_architecture.py), with only the
input shape changed from 236x236 to match our patterns.

Hyperparameters are the ones stated in the paper. The authors' train_model.py is
referenced in their README but is not actually in the repository, so the training
loop here is reconstructed from the paper's description.

Edit the SETTINGS block below, then:
    ~/.dl/bin/python train_autoencoder.py
"""

import json
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")          # no display under WSL; save figures instead
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# ======================================================================
# SETTINGS
# ======================================================================
# Resolved from the script's own location, so this file runs unchanged
# whether it's at ~/EMsoftData_work/08_synthetic_training_data (local) or
# wherever it lands after being copied to the cluster.
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = f"{OUT_DIR}/training_pairs_30k.h5"
MODEL_OUT = f"{OUT_DIR}/denoiser_30.keras"

EPOCHS     = 100                    # paper: 10
BATCH_SIZE = 64                     # paper: 64
                                    # optimizer adam and MSE loss are also
                                    # from the paper, set further down

EARLY_STOP = True                 # paper used none; set True for long runs
PATIENCE   = 20                 # only used when EARLY_STOP is True

SEED       = 42
# ======================================================================


tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TensorFlow {tf.__version__}")
print(f"GPUs visible: {tf.config.list_physical_devices('GPU') or 'none (CPU only)'}")


# ----------------------------------------------------------------------
# Load the training pairs
# ----------------------------------------------------------------------
print(f"\nLoading {DATA_FILE}")
with h5py.File(DATA_FILE, "r") as f:
    clean     = f["clean"][:]          # (N, 60, 60) float32 in [0, 1]
    noisy     = f["noisy"][:]          # (N, 60, 60) float32 in [0, 1]
    train_idx = f["train_idx"][:]
    val_idx   = f["val_idx"][:]
    meta      = dict(f.attrs)

n_patterns, height, width = clean.shape

print(f"  {n_patterns} pairs of {height} x {width}")
print(f"  noising mode: {meta.get('mode')}, SNR {float(meta.get('snr', 0)):.3f}")
print(f"  clean range {clean.min():.3f} to {clean.max():.3f}")
print(f"  noisy range {noisy.min():.3f} to {noisy.max():.3f}")

# The sigmoid output layer can only produce values in [0, 1],
# so the targets must live in that range too.
assert clean.max() <= 1 + 1e-6, "targets must be in [0, 1]"

# Keras convolutions expect a channel axis: (N, H, W) -> (N, H, W, 1)
x_train = noisy[train_idx][..., None]
y_train = clean[train_idx][..., None]
x_val   = noisy[val_idx][..., None]
y_val   = clean[val_idx][..., None]

print(f"  train {len(x_train)}, validation {len(x_val)}")

steps = int(np.ceil(len(train_idx) / BATCH_SIZE)) * EPOCHS
print(f"  gradient steps this run: {steps}  (the paper had roughly 5800)")
if steps < 2000:
    print("  NOTE: far fewer steps than the paper, so expect underfitting.")
    print("        Raise EPOCHS, or use a larger source scan such as ni_gain(0).")


# ----------------------------------------------------------------------
# Build the model
# ----------------------------------------------------------------------
# Two 2x downsamples then two 2x upsamples, so the size must divide by 4.
#   original: 236 -> 118 -> 59 -> 118 -> 236
#   ours:      60 ->  30 -> 15 ->  30 ->  60
if height % 4 or width % 4:
    raise SystemExit(f"{height}x{width} is not divisible by 4; pad or crop first")

inputs = layers.Input(shape=(height, width, 1))

# Encoder
x = layers.Conv2D(64,  (1, 1), activation="relu", padding="same")(inputs)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(128, (5, 5), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64,  (1, 1), strides=2, activation="relu", padding="same")(x)
outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.summary()


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
callbacks = []
if EARLY_STOP:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, PATIENCE // 3)),
    ]

print(f"\nTraining: {EPOCHS} epochs, batch {BATCH_SIZE}, adam, MSE")
history = autoencoder.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=2,
)

print(f"\nfinal training loss   : {history.history['loss'][-1]:.6f}")
print(f"final validation loss : {history.history['val_loss'][-1]:.6f}")
print(f"paper reported        : 0.0022 (on their data, not directly comparable)")

autoencoder.save(MODEL_OUT)
print(f"saved {MODEL_OUT}")

with open(f"{OUT_DIR}/training_history_30k.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
              f, indent=2)
print("saved training_history_30k.json")


# ----------------------------------------------------------------------
# How much did it actually help?
# ----------------------------------------------------------------------
denoised = autoencoder.predict(x_val, batch_size=BATCH_SIZE, verbose=0)[..., 0]
target   = y_val[..., 0]
noisy_in = x_val[..., 0]


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


print("\nValidation set, measured against the clean target:")
print(f"  noisy input : PSNR {psnr(noisy_in, target):6.2f} dB   NCC {ncc(noisy_in, target):.4f}")
print(f"  denoised    : PSNR {psnr(denoised, target):6.2f} dB   NCC {ncc(denoised, target):.4f}")
print("  denoised should beat noisy on both, otherwise something is wrong")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
idx = 0     # which validation pattern to show
fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
ax[0].imshow(noisy_in[idx], cmap="gray", vmin=0, vmax=1)
ax[0].set_title("noisy input", fontsize=10)
ax[1].imshow(denoised[idx], cmap="gray", vmin=0, vmax=1)
ax[1].set_title("denoised output", fontsize=10)
ax[2].imshow(target[idx], cmap="gray", vmin=0, vmax=1)
ax[2].set_title("target/ground_truth", fontsize=10)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/denoise_examples_30k.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote denoise_examples_30k.png")

n_show = min(6, len(denoised))
fig, ax = plt.subplots(3, n_show, figsize=(2.6 * n_show, 8.2))
for j in range(n_show):
    ax[0, j].imshow(noisy_in[j], cmap="gray", vmin=0, vmax=1)
    ax[0, j].set_title("noisy input", fontsize=9)
    ax[1, j].imshow(denoised[j], cmap="gray", vmin=0, vmax=1)
    ax[1, j].set_title("denoised output", fontsize=9)
    ax[2, j].imshow(target[j], cmap="gray", vmin=0, vmax=1)
    ax[2, j].set_title("target/ground_truth", fontsize=9)
    for i in range(3):
        ax[i, j].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/denoise_examples_30k.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote denoise_examples_30k.png")

print("""
Remember: MSE is not the result. The paper's claim is that denoising improves
indexing. Test that by running Hough indexing on the noisy patterns and on the
denoised ones, then comparing both against ni_angles.txt by disorientation.
""")