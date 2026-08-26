# 04 — Dictionary indexing

**Role:** close the loop on the simulation side — take the synthetic EBSD
patterns produced in [`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md)
and re-index them with EMsoft's GPU dictionary indexer (`EMEBSDDI`), the same
way you'd index a real detector scan. This is where you find out whether
orientations can actually be recovered from the simulated patterns, clean
and noisy, and produces the orientation maps used for visualization.

Note the folder name collision: this directory and
[`04_microstructure_demo`](04_microstructure_demo.md) both happen to start
with `04_` on disk — they're unrelated, independent stages that were just
never renumbered.

**Input:** `03_ebsd_pattern_simulation/Ni_EBSD_sim.h5` (clean, 10,000 ×
480 × 640 patterns) and `Ni_EBSD_sim_NOISY.h5` (noisy counterpart, same
shape) + `02_master_pattern/Ni_master_hires.h5` (needed by dynamic
indexing to build the on-the-fly dictionary)
**Output:** `Ni_CLEAN_indexed.h5`/`.ang` and `Ni_NOISY_indexed.h5`/`.ang` →
converted to `clean.dream3d`/`.xdmf` and `noisy.dream3d`/`.xdmf` for
visualization in ParaView

## How dictionary indexing works here

`indexingmode = 'dynamic'` means EMsoft doesn't precompute and store a giant
dictionary on disk — it generates candidate patterns on the fly from
`masterfile` at `ncubochoric = 100` cubochoric sampling points, then compares
every experimental pattern against that dictionary on the GPU using the
normalized dot product (`similaritymetric = 'ndp'`), keeping the top
`nnk = 50` matches per pixel.

Key parameters in `EMEBSDDI.nml` / `EMEBSDDI_noise.nml`:

- **`exptnumsx`/`exptnumsy` = 640/480, `binning = 8`** — the experimental
  patterns are stored full-resolution (640×480, matching `Ni_EBSD_sim.h5`)
  but binned down by 8× before the dot-product comparison, i.e. compared at
  80×60.
- **`ipf_wd`/`ipf_ht` = 100/100, `stepX`/`stepY` = 1.0** — the 100×100 scan
  grid, 1 μm step, matching the Dream3D-generated microstructure the
  patterns were simulated from.
- **`thetac`, `L`, `delta`, `xpc`, `ypc`** — same detector geometry as
  [`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md)'s
  `EMEBSD_ni.nml`, since the dictionary has to be generated under the same
  geometry the experimental patterns were simulated with.
- **`hipassw`, `nregions`** — pattern preprocessing (high-pass filter,
  adaptive histogram equalization) applied before comparison, standard
  Hough/dictionary-indexing practice.

`EMEBSDDI.template` is a blank, generic reference copy — not read by
anything, same convention as `03`'s `.template` file.

## Two runs

| File | Input patterns | Output |
|---|---|---|
| `EMEBSDDI.nml` | `Ni_EBSD_sim.h5` (clean) | `Ni_CLEAN_indexed.h5` / `.ang` |
| `EMEBSDDI_noise.nml` | `Ni_EBSD_sim_NOISY.h5` (noisy) | `Ni_NOISY_indexed.h5` / `.ang` |

The two `.nml` files are otherwise identical (same detector geometry, same
scan grid, same master pattern) — only `exptfile`, `datafile`, and `angfile`
differ.

## Running it

Unlike `03`, this needs a GPU — `submit_indexing.sh` requests one via SLURM
(`--gres=gpu:1`) and runs `EMEBSDDI`:

```bash
sbatch submit_indexing.sh
```

`submit_indexing.sh` currently points `$NML` at `EMEBSDDI_noise.nml` (the
clean-run line is commented out just above it) — swap which line is
commented to switch which of the two runs it submits. It also sets
`OCL_ICD_VENDORS=/etc/OpenCL/vendors`, a fix for an `Unknown CL error code
: -1001` (`CL_PLATFORM_NOT_FOUND_KHR`) OpenCL platform-discovery error seen
on this cluster, and checks that `~/.config/EMsoft` has already been
configured (paths, license) before running.

## Output contents

`Ni_CLEAN_indexed.h5` / `Ni_NOISY_indexed.h5` (per-scan-point results, under
`/Scan 1/EBSD/Data/`):

- `Phi1`, `Phi`, `Phi2` — indexed Euler angles (flat, 10,000 points)
- `CI` — confidence index, `Fit`, `IQ` — image quality, `ISM` —
  indexing success metric, per point
- `CIMap`, `IQMap`, `ISMap`, `OSM`, `AvDotProductMap`, `KAM` — the same
  quantities (plus orientation similarity and kernel average misorientation)
  already reshaped to the 100×100 scan grid
- `DictionaryEulerAngles` — the full 333,824-orientation cubochoric
  dictionary generated on the fly
- `TopDotProductList` / `TopMatchIndices` — the top-50 matches kept per
  point, before collapsing to a single best orientation

`Ni_CLEAN_indexed.ang` / `Ni_NOISY_indexed.ang` — the same per-point result
in standard TSL `.ang` text format (header + one row per pixel: Euler
angles, x/y position, CI, Fit, phase), the format most downstream EBSD
tools (MTEX, orix, DREAM3D) expect.

## Visualization: `clean.xdmf` / `noisy.xdmf`

`clean.dream3d`/`clean.xdmf` and `noisy.dream3d`/`noisy.xdmf` were built by
importing the corresponding `.ang` file into DREAM3D (its "Import EDAX ANG
Data" reader) and writing it back out as a DREAM3D/Xdmf pair — there's no
script for this in the repo, it's a manual DREAM3D pipeline step. Open the
`.xdmf` file in ParaView to visualize.

**Important — no grain field yet.** The DREAM3D file only carries the raw
per-pixel `CellData` (Confidence Index, EulerAngles, Fit, IPFColor, Image
Quality, Phases, SEM Signal, X/Y Position) copied straight from the `.ang`
file. There's no `FeatureIds` array, so ParaView will show orientation/IPF
color and quality maps but **not** a grain count or grain boundaries —
that requires an additional segmentation step (e.g. DREAM3D's "Segment
Features (Misorientation)" filter, or `calcGrains` in MTEX/orix on the
`.ang` file) that hasn't been run on this output.
