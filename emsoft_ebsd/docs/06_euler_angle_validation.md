# 06 — Euler angle validation

**Role:** the one loop in an otherwise linear pipeline. This folder runs
*before* [`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md), to
extract real ground-truth orientations for it to simulate, and *after* it,
to check whether the simulation actually agrees with reality.

**Input:** an independently-downloaded real Ni EBSD scan (`patterns.h5`) —
this is the true starting point of the whole validation chain, obtained
outside of EMsoft entirely
**Output:** `ni_angles.txt` (fed forward into `03`) + a set of comparison
figures (fed nothing forward — diagnostic only)

Scripts are numbered `01`–`08` in actual run order (not folder-creation
order), documented in this folder's own `README.md`. All generated PNGs
live in `outputs/`; `patterns.h5`/`ni_angles.txt`/`ni_angles_map.csv` stay
at the folder's top level since they're pipeline data, not diagnostics.

## Before EMsoft's EMEBSD step

1. **`01_download_data_patternsh5.py`** — downloads a real, already-indexed
   Ni scan via kikuchipy (`kikuchipy.data.ni_gain(1, allow_download=True)`,
   saved as `patterns.h5`). See **"The ni_gain dead end"** below for why this
   specific function call matters more than it looks.
2. **`02_get_pc.py`** — reads the real scan's detector geometry and prints
   the pattern center in EMsoft's convention. Pasted manually into
   `03_ebsd_pattern_simulation/EMEBSD_ni.nml`.
3. **`03_extract_euler_angles_from_patterns.py`** — extracts every indexed
   orientation from `patterns.h5`, writes `ni_angles.txt` (EMsoft's angle
   file format) and `ni_angles_map.csv` (row/col lookup). **This file gets
   copied into `03_ebsd_pattern_simulation/`, replacing the version there**
   — that's the actual hand-off point from "real data" into "simulation."

## External step (not a script in this folder)

```bash
cd ../03_ebsd_pattern_simulation
EMEBSD EMEBSD_ni.nml
```

Produces `Ni_EBSD_sim.h5`, which every script below compares against.

## After EMEBSD has produced `Ni_EBSD_sim.h5`

4. **`04_pc_check.py`** — real vs. simulated patterns at the same scan
   points, to confirm the pattern center puts bands in the right place.
5. **`05_side_by_side.py`** — IPF-colored maps from both real and simulated
   orientations; max color difference should be ~0 since both come from the
   same underlying orientations.
6. **`06_diagnose_match.py`** — if patterns visually disagree, tests whether
   a flip/rotation/transpose of the simulated pattern fixes it (EMsoft and
   kikuchipy don't always agree on "which way is up").

## General visualization (needs only `patterns.h5`)

7. **`07_visualize_ebsps.py`** — four diagnostic views of the raw real
   patterns (montage, raw-vs-background-removed, full mosaic, IPF map with
   sample patterns pinned to their locations).
8. **`08_visualize_IPF_colored_map.py`** — a single IPF-Z map, real or
   simulated depending on which `kp.load(...)` line is active.

## The `ni_gain` dead end

Partway through the project, more training data was needed (see
[`07_training_pipeline`](07_training_pipeline.md)), and the obvious move
seemed to be swapping `kikuchipy.data.nickel_ebsd_large()` (4,125 points)
for a bigger dataset in the same family, `kikuchipy.data.ni_gain(1)`
(29,800 points). It failed with:

```
AttributeError: 'NoneType' object has no attribute 'rotations'
```

The root cause, found by reading kikuchipy's own source: `nickel_ebsd_large()`
loads kikuchipy's own HDF5 format, which bundles a pre-computed
`CrystalMap` (indexed orientations) directly in the file — that's a
convenience baked in by whoever packaged that specific tutorial dataset.
`ni_gain()` loads a raw NORDIF `.dat` pattern file instead — literally just
detector output from a gain-comparison study, with **no indexing done at
all**. `EBSD.xmap` is `None` for it. There is no larger pre-indexed Ni
dataset kikuchipy ships at this scale; `nickel_ebsd_large` is the only one.

This is why [`08_synthetic_training_data`](08_synthetic_training_data.md)
takes a completely different approach to getting more data: instead of
finding a bigger *real, indexed* scan, it generates **synthetic**
orientations directly (uniform random sampling of SO(3)) and simulates
patterns for those with EMsoft — sidestepping the need for real indexed
ground truth entirely, since the denoiser only needs pattern *diversity*,
not real-world orientation authenticity.
