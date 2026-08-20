# EMsoftData_work — documentation index

This is the narrative companion to the root [`README.md`](../README.md). The
root file is a file-by-file reference table; these pages explain *why* each
stage exists, *how* it actually works, and the dead ends and gotchas hit
along the way — the things a table of filenames can't carry.

A visual sketch of the whole pipeline (the same one used in the project
meeting) is published here: **[Ni EBSD Pipeline — Sketch](https://claude.ai/code/artifact/bdc08c8d-a46f-4635-b69f-d5fef70c1bd9)**.

## Reading order

The pipeline runs in a straight line except for one loop. Read these in
order the first time:

| Stage | Doc | One line |
|---|---|---|
| 00 | [crystal_structure.md](00_crystal_structure.md) | The Ni lattice definition everything else starts from |
| 01 | [monte_carlo.md](01_monte_carlo.md) | Simulate electrons scattering inside the crystal |
| 02 | [master_pattern.md](02_master_pattern.md) | Turn that into the full-sphere diffraction pattern |
| 03 | [ebsd_pattern_simulation.md](03_ebsd_pattern_simulation.md) | Project it onto a detector at chosen orientations |
| 04 | [microstructure_demo.md](04_microstructure_demo.md) | Standalone synthetic grain-map illustration (dead end) |
| 05 | [noising_training_data.md](05_noising_training_data.md) | Build noisy/clean training pairs, calibrated to the real detector |
| 06 | [euler_angle_validation.md](06_euler_angle_validation.md) | **The loop**: supplies real orientations before 03 runs, checks the result after |
| 07 | [training_pipeline.md](07_training_pipeline.md) | First denoiser training attempt — bottlenecked by dataset size |
| 08 | [synthetic_training_data.md](08_synthetic_training_data.md) | Fix: train on 30,000 synthetic orientations instead of 4,125 real ones |
| 09 | [evaluation.md](09_evaluation.md) | Does it actually work? Two independent tests, **with results** |

Also in `docs/`: [`EMsoft-WSL-Guide.md`](EMsoft-WSL-Guide.md), the standalone
build/install guide for EMsoft itself on WSL2 — unrelated to the pipeline
narrative above, kept for reference.

## The headline finding

Skip ahead to [evaluation.md](09_evaluation.md#results) if you just want the
punchline: at the noise level actually measured from the real detector
(SNR 1.27), Hough indexing is already ~99% accurate even on raw noisy
patterns, so denoising has almost no visible room to help. Push the noise
level roughly 2.5x harsher than that, and a real, measurable gap opens up
between noisy, denoised, and clean — evidence the denoiser works, but only
once the input is worse than what this particular detector/sample actually
produces.

## Three environments, on purpose

Different stages need conflicting dependencies, so three separate Python
environments exist rather than one that fights itself:

| Environment | Has | Used by |
|---|---|---|
| `venv/` (repo root) | kikuchipy, orix, hyperspy, EMsoft-adjacent tooling — Python 3.14 | Stages 00–06, `09_evaluation`'s Stage 2 (indexing) |
| `~/.dl` | TensorFlow (Python 3.13 — TF has no 3.14 wheels yet) | `07_training_pipeline`, `08_synthetic_training_data`, `09_evaluation`'s Stage 1 (prediction) |
| `~/.ebsd` | kikuchipy + orix + pyebsdindex, deliberately separate from TensorFlow | `09_evaluation`'s Stage 2 (indexing) |

In practice the root `venv` turned out to already have everything `.ebsd`
does (plus pyebsdindex), so either works for Stage 2 — but scripts are
written to run correctly regardless of which one invokes them, since paths
are resolved relative to each script's own location, not hardcoded to one
machine or one user.
