# 06_euler_angle_validation — run order

Scripts are numbered in the order you actually run them. All generated
figures land in `outputs/`; `patterns.h5`, `ni_angles.txt`, and
`ni_angles_map.csv` stay at this folder's top level because they're pipeline
data (consumed by `03_ebsd_pattern_simulation/`), not diagnostic images.

Run everything from inside this folder, with the venv active.

## Before EMsoft's EMEBSD step

1. **`01_download_data_patternsh5.py`** — downloads the real experimental
   scan from kikuchipy and saves it as `patterns.h5`. Only needs to be run
   once (or again if you switch data source, e.g. `ni_gain(0)`/`ni_gain(1)`
   for more scan points).
2. **`02_get_pc.py`** — reads `patterns.h5`'s detector geometry and prints
   the pattern center (`xpc`, `ypc`, `L`, `delta`) in EMsoft's convention.
   Paste these into `../03_ebsd_pattern_simulation/EMEBSD_ni.nml`.
3. **`03_extract_euler_angles_from_patterns.py`** — extracts every indexed
   orientation from `patterns.h5` and writes `ni_angles.txt` (the EMsoft
   angle file) and `ni_angles_map.csv` (row/col lookup). **Copy the new
   `ni_angles.txt` into `../03_ebsd_pattern_simulation/`, replacing the old
   one** — `EMEBSD_ni.nml` reads its `anglefile` from there, not from here.

## External step — not a script in this folder

Run EMsoft itself from `03_ebsd_pattern_simulation/`:

```
cd ../03_ebsd_pattern_simulation
EMEBSD EMEBSD_ni.nml
```

This produces/overwrites `Ni_EBSD_sim.h5`, which every script below compares
against.

## After EMEBSD has produced `Ni_EBSD_sim.h5`

4. **`04_pc_check.py`** — quick visual: real vs. simulated patterns at the
   same scan points, to confirm the pattern center is right. → `outputs/pc_check.png`
5. **`05_side_by_side.py`** — builds IPF-colored orientation maps from both
   real and simulated data; the max color difference should be ~0 since
   both come from the same orientations. → `outputs/ipf_real_vs_sim.png`
6. **`06_diagnose_match.py`** — if patterns don't visually agree, tests
   whether a flip/rotation/transpose of the simulated patterns fixes it
   (EMsoft and kikuchipy don't always agree on "which way is up"). → `outputs/diagnose_match.png`

## General visualization (only needs `patterns.h5`, run anytime)

7. **`07_visualize_ebsps.py`** — four diagnostic views of the raw real
   patterns: montage, raw-vs-background-removed, full mosaic, IPF map with
   sample patterns pinned to their locations. → `outputs/ebsp_montage.png`,
   `outputs/ebsp_raw_vs_proc.png`, `outputs/ebsp_mosaic.png`, `outputs/ebsp_with_map.png`
8. **`08_visualize_IPF_colored_map.py`** — single IPF-Z colored map, real or
   simulated depending on which `kp.load(...)` line is active. → `outputs/ipf_z_map_with_key_sim.png`
