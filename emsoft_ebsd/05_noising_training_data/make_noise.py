#!/usr/bin/env python3
"""
STEP 4 -- make the noisy training inputs.

Takes the clean EMsoft-simulated patterns and produces a matched noisy copy of
each one, giving the (noisy -> clean) pairs a denoising autoencoder trains on.

Two modes:

  --mode calibrated   (default) measures the real scan's noise level and matches
                      it. Uses the fact that adjacent scan points inside one
                      grain have the same true signal, so their difference is
                      pure noise.

  --mode paper        Gaussian noise at a fixed sigma plus contrast reduction,
                      following Andrews et al. (sigma=25 on 8-bit, contrast
                      reduced to resemble the as-collected patterns).

Run from ~/EMsoftData_work:
    python3 make_noisy.py
    python3 make_noisy.py --mode paper --sigma 25 --contrast 0.5
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

p = argparse.ArgumentParser()
p.add_argument("--sim", default="../03_ebsd_pattern_simulation/Ni_EBSD_sim.h5", help="clean simulated patterns")
p.add_argument("--real", default="../06_euler_angle_validation/patterns.h5", help="real scan")
p.add_argument("--out", default="training_pairs.h5")
p.add_argument("--mode", default="calibrated", choices=["calibrated", "paper"])
p.add_argument("--sigma", type=float, default=25.0,
               help="paper mode: Gaussian sigma on the 0-255 scale")
p.add_argument("--contrast", type=float, default=0.5,
               help="paper mode: contrast factor, <1 reduces contrast")
p.add_argument("--val-fraction", type=float, default=0.15)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

rng = np.random.default_rng(args.seed)

# ----------------------------------------------------------------------
# Load the clean simulated patterns
# ----------------------------------------------------------------------
print(f"Loading clean patterns: {args.sim}")
with h5py.File(args.sim, "r") as f:
    clean_raw = f["EMData/EBSD/EBSDPatterns"][:]
clean_raw = np.squeeze(clean_raw).astype(np.float32)
n, sy, sx = clean_raw.shape
print(f"  {n} patterns of {sy} x {sx}, range {clean_raw.min():.0f}-{clean_raw.max():.0f}")

# Per-pattern standardisation: zero mean, unit standard deviation.
# This separates "what the pattern looks like" from "how strong it is",
# so noise can be added at a controlled signal-to-noise ratio.
mu = clean_raw.mean(axis=(1, 2), keepdims=True)
sd = clean_raw.std(axis=(1, 2), keepdims=True)
clean_z = (clean_raw - mu) / np.maximum(sd, 1e-6)

# ----------------------------------------------------------------------
# Work out how much noise to add
# ----------------------------------------------------------------------
if args.mode == "calibrated":
    print(f"\nCalibrating against the real scan: {args.real}")
    try:
        import kikuchipy as kp
        from orix.quaternion import Orientation
    except ImportError as e:
        sys.exit(f"Calibrated mode needs kikuchipy and orix: {e}")

    s = kp.load(args.real)
    ny, nx = s.axes_manager.navigation_shape[::-1]
    s.remove_static_background()
    s.remove_dynamic_background()
    real = np.asarray(s.data, dtype=np.float32)

    xmap = s.xmap
    pg = xmap.phases[0].point_group
    o = Orientation(xmap.rotations.data, symmetry=pg).reshape(ny, nx)

    # Horizontally adjacent pairs, keep only those within the same grain
    dis = np.asarray(o[:, :-1].angle_with(o[:, 1:], degrees=True)).reshape(ny, nx - 1)
    same = dis < 0.5
    print(f"  {same.sum()} adjacent pairs within 0.5 deg (same grain)")

    if same.sum() < 50:
        sys.exit("Too few same-grain pairs to calibrate -- use --mode paper")

    rows, cols = np.nonzero(same)
    # difference of two same-signal patterns is sqrt(2) x the noise
    diffs = real[rows, cols] - real[rows, cols + 1]
    noise_std = float(diffs.std() / np.sqrt(2.0))

    total_std = float(real.std())
    signal_var = max(total_std ** 2 - noise_std ** 2, 1e-9)
    signal_std = float(np.sqrt(signal_var))
    snr = signal_std / noise_std

    print(f"  real total std   : {total_std:.4f}")
    print(f"  estimated noise  : {noise_std:.4f}")
    print(f"  estimated signal : {signal_std:.4f}")
    print(f"  signal-to-noise  : {snr:.3f}")

    # clean_z has unit std, so noise std of 1/snr reproduces that ratio
    noise_level = 1.0 / snr
    signal_scale = 1.0

else:
    print(f"\nPaper mode: contrast {args.contrast}, sigma {args.sigma} on 0-255")
    # A standardised pattern spread over 0-255 has std of roughly 255/6.
    signal_scale = args.contrast
    noise_level = args.sigma / (255.0 / 6.0)
    snr = signal_scale / noise_level
    print(f"  implied signal-to-noise: {snr:.3f}")

print(f"\nAdding noise: signal x {signal_scale:.3f}, noise std {noise_level:.4f}")

# ----------------------------------------------------------------------
# Build the noisy patterns
# ----------------------------------------------------------------------
noisy_z = signal_scale * clean_z + rng.normal(0.0, noise_level, clean_z.shape).astype(np.float32)


def to_unit(a):
    """Map to [0, 1] per pattern, clipping the extreme tails."""
    lo = np.percentile(a, 0.5, axis=(1, 2), keepdims=True)
    hi = np.percentile(a, 99.5, axis=(1, 2), keepdims=True)
    return np.clip((a - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)


clean_u = to_unit(clean_z)
noisy_u = to_unit(noisy_z)

print(f"  clean: mean {clean_u.mean():.3f}, std {clean_u.std():.3f}")
print(f"  noisy: mean {noisy_u.mean():.3f}, std {noisy_u.std():.3f}")

# ----------------------------------------------------------------------
# Train / validation split
# ----------------------------------------------------------------------
perm = rng.permutation(n)
n_val = int(round(args.val_fraction * n))
val_idx = np.sort(perm[:n_val])
train_idx = np.sort(perm[n_val:])
print(f"\nSplit: {len(train_idx)} train, {len(val_idx)} validation")

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
with h5py.File(args.out, "w") as f:
    f.create_dataset("clean", data=clean_u, compression="gzip", compression_opts=4)
    f.create_dataset("noisy", data=noisy_u, compression="gzip", compression_opts=4)
    f.create_dataset("train_idx", data=train_idx)
    f.create_dataset("val_idx", data=val_idx)
    f.attrs["mode"] = args.mode
    f.attrs["signal_scale"] = signal_scale
    f.attrs["noise_level"] = noise_level
    f.attrs["snr"] = snr
    f.attrs["seed"] = args.seed
    f.attrs["source_sim"] = args.sim
    if args.mode == "paper":
        f.attrs["sigma_8bit"] = args.sigma
        f.attrs["contrast"] = args.contrast

print(f"wrote {args.out}")

# ----------------------------------------------------------------------
# Visual check: does the synthetic noise resemble the real thing?
# ----------------------------------------------------------------------
picks = [9 * 75 + 12, 18 * 75 + 37, 27 * 75 + 25, 36 * 75 + 50]
rows = 3 if args.mode == "calibrated" else 2
fig, ax = plt.subplots(rows, len(picks), figsize=(3.1 * len(picks), 3.2 * rows))
ax = np.atleast_2d(ax)

for j, i in enumerate(picks):
    ax[0, j].imshow(clean_u[i], cmap="gray")
    ax[0, j].set_title(f"clean (target)\nline {i}", fontsize=9)
    ax[1, j].imshow(noisy_u[i], cmap="gray")
    ax[1, j].set_title("noisy (input)", fontsize=9)
    if rows == 3:
        r, c = divmod(i, 75)
        ax[2, j].imshow(real[r, c], cmap="gray")
        ax[2, j].set_title(f"real ({r},{c})", fontsize=9)
    for k in range(rows):
        ax[k, j].axis("off")

fig.suptitle(f"Training pairs -- {args.mode} mode, SNR {snr:.2f}", fontsize=13)
plt.tight_layout()
plt.savefig("noise_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote noise_check.png")

print("""
Look at noise_check.png: the middle row is what the network sees, the top row is
what it must reproduce. If calibration worked, the middle row should look about
as degraded as the bottom row -- not obviously cleaner or noisier.

If the noisy row looks too clean, lower the SNR:
    python3 make_noisy.py --mode paper --contrast 0.35 --sigma 30
""")