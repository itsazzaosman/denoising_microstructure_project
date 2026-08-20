# 05 — Noising training data

**Role:** turn clean simulated patterns into matched noisy/clean pairs, at a
noise level *measured from the real detector* — the training data for the
denoising autoencoder in [`07_training_pipeline`](07_training_pipeline.md).

**Input:** [`03_ebsd_pattern_simulation/Ni_EBSD_sim.h5`](03_ebsd_pattern_simulation.md)
(clean) + [`06_euler_angle_validation/patterns.h5`](06_euler_angle_validation.md)
(real scan, used only to measure the noise level, not as training data itself)
**Output:** `training_pairs.h5` → consumed by
[`07_training_pipeline`](07_training_pipeline.md) and
[`09_evaluation`](09_evaluation.md) (as the held-out generalization test set)

This folder was originally named `05_denoising_training_data`; it was
renamed to `05_noising_training_data` partway through the project (the
script inside, `make_noise.py`, keeps its original name — worth knowing if
older notes reference the old folder name).

## The noise model: additive Gaussian, two ways to set its strength

`make_noise.py` standardizes each clean pattern to zero mean / unit standard
deviation, adds i.i.d. Gaussian noise, then rescales back to `[0, 1]` per
pattern (clipping the 0.5–99.5 percentile tails). The actual noise-adding
line is:

```python
noisy_z = signal_scale * clean_z + rng.normal(0.0, noise_level, clean_z.shape)
```

This is a simplified stand-in for real EBSD detector noise, which is
typically Poisson (shot) noise — signal-dependent, from counting individual
electrons — not signal-independent Gaussian noise. The simplification is
deliberate: it's the same one Andrews et al. (2023) used, and this pipeline
follows their method rather than inventing its own noise model.

Two modes set `noise_level` differently:

- **`--mode calibrated`** (default) — measures real noise directly from
  `patterns.h5`. It finds pairs of horizontally-adjacent scan points that
  sit inside the same grain (disorientation < 0.5°, so their true signal is
  identical), takes the pixel-wise difference between them — since the true
  signal cancels out, that difference is pure noise — and uses its standard
  deviation (divided by √2, since it's a difference of two independent noisy
  samples) as the real noise level. This produced **SNR = 1.27**, the number
  used throughout the rest of the project as "the real detector's actual
  noise level."
- **`--mode paper`** — a fixed Gaussian sigma (`--sigma`, default 25, on an
  8-bit 0–255 scale) plus a contrast-reduction factor (`--contrast`, default
  0.5) taken directly from Andrews et al.'s own paper, rather than measured.

## Two files, two noise levels

| File | Mode | Effective SNR | Purpose |
|---|---|---|---|
| `training_pairs.h5` | `calibrated` | 1.27 | Original training data for [`07_training_pipeline`](07_training_pipeline.md); later reused in [`09_evaluation`](09_evaluation.md) as the "calibrated" held-out test |
| `training_pairs_harsh.h5` | `paper --contrast 0.35 --sigma 30` | ≈0.50 | Generated later, specifically for [`09_evaluation`](09_evaluation.md)'s harsher-noise indexing test — see that page for why the calibrated level turned out too easy to show a real effect |

Both files hold the same 4,125 patterns (from the same `Ni_EBSD_sim.h5`) —
only the noise draw differs.

## Visual check

`noise_check.png` shows real vs. clean-simulated vs. noisy-simulated
patterns side by side, to sanity-check that the synthetic noise actually
resembles what the real detector produces. The script's own guidance: the
noisy row should look "about as degraded" as the real row — not obviously
cleaner or noisier.
