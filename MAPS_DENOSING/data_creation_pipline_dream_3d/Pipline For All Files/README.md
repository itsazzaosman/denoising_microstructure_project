# Pipline For All Files

The **full-output** DREAM3D pipeline: generates one synthetic Ni microstructure and
writes it out in *every* format — Euler angle text, IPF PNG, and a DREAM.3D/Xdmf
pair. Use this one when you want the complete record of a map.

The sibling folder [`../Pipline For Pictures Only`](../Pipline%20For%20Pictures%20Only)
runs the identical microstructure generation but stops after the PNG — no angle
file, no `.dream3d`. That's the one wired into
[`11_batch_generation/generate_microstructures_windows.py`](../../11_batch_generation/generate_microstructures_windows.py)
for the 40,000-map batch, since writing three formats per map isn't needed at that
scale.

No EMsoft, no GPU, no indexing anywhere in here — DREAM3D builds the grains and
assigns their orientations directly, so this output *is* ground truth rather than
something recovered from simulated diffraction patterns.

## The pipeline

`Ni_pipline_A_generate_microstructure_with_angles_dream3d_pictures.json`, 11 filters:

| # | Filter | What it does |
|---|---|---|
| 00 | StatsGenerator | Log-normal grain size distribution — `Mu = 2.15`, `Sigma = 0.30` (mean ESD ≈ 9 µm), single-phase Ni, cubic m-3m |
| 01 | Initialize Synthetic Volume | **128 × 128 × 1** voxels at 1 × 1 × 1 µm → a 128 × 128 µm map |
| 02 | Establish Shape Types | Ellipsoid (equiaxed) grain shapes |
| 03 | Pack Primary Phases | Packs grains per the size distribution; creates `FeatureIds` |
| 04 | Find Feature Neighbors | Neighbor/shared-surface lists needed by Match Crystallography |
| 05 | Find Surface Features | Flags grains touching the volume edge |
| 06 | Match Crystallography | Assigns each grain an orientation from a random (untextured) ODF |
| 07 | Export ASCII Data | → `Nickel_128x128_Euler.txt` — per-pixel Euler angles, radians |
| 08 | Generate IPF Colors | IPF-**Z** colors (reference direction `001`) → `IPFColor` |
| 09 | ITK::Image Writer | → `Nickel_128x128.png` — the IPF map as an image |
| 10 | Write DREAM.3D Data File | → `Nickel_128x128.dream3d` + `.xdmf` |

Run it headless with DREAM3D's CLI (v6.5.171 on Windows):

```
PipelineRunner.exe -p Ni_pipline_A_generate_microstructure_with_angles_dream3d_pictures.json
```

The three output paths are hardcoded absolute paths inside the JSON — rewrite them
per run (that's exactly what the batch driver script does) or every run overwrites
the last.

## Example outputs in this folder

One run's worth, kept as a reference of what the pipeline produces:

| File | What it is |
|---|---|
| `Nickel_128x128_Euler.txt` | 16,384 Euler triplets (128 × 128), radians, one header line. This is the format [`convert_dream3d_angles.py`](../../03_ebsd_pattern_simulation/convert_dream3d_angles.py) converts into EMsoft's `eu`/degrees angle files |
| `Nickel_128x128.png` | IPF-Z colored orientation map |
| `Nickel_128x128.dream3d` / `.xdmf` | HDF5 + Xdmf pair carrying `EulerAngles`, `FeatureIds`, `IPFColor`, `Phases`. Open the `.xdmf` in ParaView |

This particular run came out at **51 grains** (`FeatureIds` 1–51; `CellFeatureData`
tuple count 52, the extra being DREAM3D's reserved placeholder feature 0). That
tracks with the 33 grains the same `Mu`/`Sigma` gave on a 100 × 100 map — the
grain count scales with map area.

Note there's no random seed anywhere in the JSON: `Pack Primary Phases` and
`Match Crystallography` re-randomize on every run, which is what makes looping
this pipeline produce a *different* microstructure each time rather than 40,000
copies of the same one.
