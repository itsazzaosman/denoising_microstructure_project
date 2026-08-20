# 08 — Synthetic training data

**Role:** fix the data-scarcity problem found in
[`07_training_pipeline`](07_training_pipeline.md) by training on 30,000
*synthetic* orientations instead of reusing the same 4,125-point real scan.
This is the model actually evaluated in [`09_evaluation`](09_evaluation.md).

**Input:** [`02_master_pattern/Ni_master_hires.h5`](02_master_pattern.md)
(same master pattern as everywhere else) + 30,000 randomly-generated
orientations
**Output:** `denoiser_30.keras` — the model `09_evaluation` tests

## Why synthetic orientations, not a bigger real scan

The obvious first move — find a bigger real, indexed Ni dataset — hit a dead
end (`kikuchipy.data.ni_gain()` has no ground-truth orientations at all; see
[`06_euler_angle_validation`](06_euler_angle_validation.md#the-ni_gain-dead-end)
for the full story). The reframe that unblocked this: the denoiser only ever
sees the *diffraction pattern image*, never the orientation that produced
it — all it needs is pattern variety, not real-world orientation
authenticity. A real scan is actually a *poor* source of that variety:
neighboring points share grains (the 4,125-point real scan contains only
~3,490 distinct orientations), and a textured sample leaves whole regions of
orientation space completely unsampled. Uniform random sampling of
orientation space has neither problem, and — because symmetrically
equivalent orientations produce identical diffraction patterns — uniform
sampling of the full rotation group SO(3) is already equivalent to uniform
sampling of the cubic fundamental zone, so no explicit symmetry reduction
is needed.

## Building the 30k-pattern dataset

1. **`01_generate_30k_angles.py`** — `orix.quaternion.Rotation.random(30000)`
   draws 30,000 uniformly random orientations, converted to Bunge Euler
   angles and written as `ni_angles_30k.txt` in EMsoft's angle-file format.
2. **`EMEBSD_30k.nml`** — the *same* detector geometry as
   [`03_ebsd_pattern_simulation/EMEBSD_ni.nml`](03_ebsd_pattern_simulation.md)
   (60×60, PC `(4.337, 17.273, 1500.8)`, same master pattern) — only the
   angle file changed, from the real 4,125-orientation list to this
   synthetic 30,000-orientation one. Run with `EMEBSD EMEBSD_30k.nml`,
   producing `Ni_EBSD_sim_30k.h5` (108 MB, ~7x the size of the original
   `Ni_EBSD_sim.h5`).
3. **`02_make_noisy_data.py`** — adds noise at a **fixed** SNR of 1.27 (the
   same number measured from the real scan in
   [`05_noising_training_data`](05_noising_training_data.md)), rather than
   recalibrating — there's no real scan to difference against for synthetic
   orientations, so the previously-measured number is reused directly.
   Produces `training_pairs_30k.h5` (736 MB, clean/noisy pairs + an 85/15
   train/val split) and a `noise_check_30k.png` visual sanity check.

## Training run

`train_autoencoder_synthetic.py` — identical architecture to
[`07_training_pipeline`](07_training_pipeline.md), pointed at this larger
dataset. Paths are resolved from the script's own location rather than
hardcoded, so the same file runs unchanged locally or after being copied to
a cluster.

| | Value |
|---|---|
| Training patterns | ~25,500 (30,000 minus 15% val split) |
| Epochs actually run | 10 |
| Gradient steps | ~3,990 (vs. the paper's ~5,800 — closer than either run in `07` managed, from more data per epoch, not more epochs) |
| Final train / val loss | 0.01079 / 0.01070 |

Note this loss isn't directly comparable to `07`'s 50-epoch run
(0.00940/0.00954) — they're measured against *different* noise draws on
*different* patterns, so a lower number doesn't mean "worse," just
"different dataset." The point of this dataset isn't a lower training loss;
it's a model that has actually seen 30,000 distinct patterns rather than
~3,490 repeated ones, which is what [`09_evaluation`](09_evaluation.md)
tests for directly.

`submit_jobs.sh` (repo root) exists to run this same training script for
many more epochs (100) on a SLURM GPU cluster — set up but not what actually
produced the `denoiser_30.keras` evaluated in `09`, which was this quick
10-epoch local run. Running the full cluster job is a natural next step if
more training time is wanted.

## Outputs

- `denoiser_30.keras` — the trained model
- `training_history_30k.json` — per-epoch loss/val_loss
- `denoise_examples_30k.png` — noisy/denoised/clean example grid
