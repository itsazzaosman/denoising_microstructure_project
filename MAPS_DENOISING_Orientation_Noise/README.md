# Orientation-noise denoising

Denoising EBSD orientation maps corrupted by two kinds of noise at once, and
beating the MTEX filter baseline at it.

The noise model has two parts, because real maps have both:

| | what it does | how much |
|---|---|---|
| **scatter** | nudges every pixel by a small random rotation | 0-1 deg, so ~0.5 deg typical |
| **misindexing** | replaces a pixel with a completely random orientation | 5% of pixels, 4x more likely on a grain boundary |

The second is what actually breaks maps: the diffraction pattern was too weak
to index and the software picked the wrong solution. It is salt-and-pepper
noise, but in orientation space rather than colour space.

**The data now on disk is a second, scatter-only experiment.** There are three
noise levels: scatter up to 5, 7 and 9 deg (2.5, 3.5 and 4.5 deg on average),
with no misindexing. The 1 deg + 5% misindexing set (`noisy_mis05`) described
above, and used for the model and baseline sections below, has been removed.
Its results are still in `figures/scores.csv` and `checkpoints/history.json`.

## Pipeline

```
DREAM.3D                                    (outside this repo)
   |  synthetic microstructures -> Euler angles per pixel
   v
datasets/clean_euler/map_XXXXX_clean_euler.txt      GROUND TRUTH, 40,000 maps
   |
   |  src/add_ebsd_noise.py      adds scatter (+ misindexing, off for these sets)
   v
datasets/noise_scatter{5,7,9}/   maps 1-500, one folder per noise level
   |-- ..._noisy.txt                     noisy Euler angles
   |-- ..._noisy.txt.badmask.npy         which pixels were replaced
   `-- ang_files/
       |-- ..._noisy_clean.ang           truth, in MTEX's format
       `-- ..._noisy_noisy.ang           noisy, in MTEX's format   <-- MTEX INPUT
   |
   |  MTEX (MATLAB)              12 classical filters
   v
datasets/mtex_out_scatter{5,7,9}/..._noisy_noisy__<method>.txt     maps 1-10 so far
   |
   |  src/prepare_cache.py       pack clean maps into one .npy  (once)
   |  src/train.py               train the model on maps 501+
   |  src/inference.py           predict maps 1-500 -> ..._unet.txt
   v
src/score_dataset.py             one table, filters and model side by side
```

**Maps 1-500 are the test set** and are never trained on. They are the maps
MTEX was given, with the exact noise MTEX saw, so the comparison is
like-for-like. Training uses maps 501+ and corrupts them fresh every epoch.

## The model

A U-Net with three heads, because the noise has two regimes that want
different treatment:

```
refine    a bounded small rotation applied to the input   -> handles scatter
replace   a full orientation, predicted from scratch      -> handles misindexing
gate      how much to trust the input at this pixel       -> chooses between them

q_out = slerp(refine(q_in), replace(x), gate)
```

Input is 13 channels per pixel: the 9 rotation-matrix entries, a 3-vector
pointing to the edge-preserving local mean, and the local-misfit scalar.

Four design points matter:

- **The gate has a dead zone.** A plain sigmoid never reaches zero, so a
  "closed" gate of even 1e-3 still drags a pixel ~0.06 deg toward the replace
  head, which is larger than the error being chased. Subtracting a dead zone
  makes closed mean exactly closed, so untouched pixels are bit-exact. The
  model therefore **cannot win the ALL column by blurring the BOUNDARY
  column**, which is the usual failure mode of a smoothing filter.

- **The untrained model is a known filter, not noise.** With the local-mean
  term it starts at 0.069 deg (the classical edge-preserving mean); set
  `--mean-radius 0` and it starts as the exact identity instead. Either way
  epoch 0 is an interpretable baseline rather than garbage.

- **Orientations are regressed in the 6D representation** (Zhou et al. 2019).
  Every rotation representation of four or fewer numbers is discontinuous
  somewhere, and the discontinuity shows up as large errors at unpredictable
  places.

- **The refinement is COMPOSED ON TOP of an edge-preserving local mean**, not
  predicted from scratch:

  ```
  q_refine = learned_delta  *  local_mean_delta  *  q_in
  ```

  The correction a denoiser must apply to a pixel is the inverse of that
  pixel's own random noise draw. A head asked to synthesise that from trunk
  features gets a gradient that cancels across pixels, so Adam's `m/sqrt(v)`
  stalls and the head never leaves its initialisation. This was measured, not
  assumed: after a full epoch `head_refine.weight` was still 7.6e-5 while the
  other heads had reached 1e-1, and with refinement as the SOLE objective on a
  fixed batch, 200 steps moved the error only 0.497 -> 0.494 deg.

  Feeding the local mean in as an input channel was **not** sufficient -- the
  refine head reads the trunk output, seven conv blocks downstream and shaped
  by the much larger gate and replace losses, so the signal did not survive.
  Composing it in at the head is what works. It also makes the untrained model
  the local-mean filter (0.069 deg) rather than the identity (0.53 deg).

  Known limitation: the *learned* part of the refinement still trains very
  weakly. Most of the scatter removal comes from the local-mean term, which is
  a classical filter. The learned contribution is concentrated in the gate and
  replace heads, i.e. in the misindexing repair.

The gate is supervised directly from the bad-pixel mask the noise generator
records. That is the one thing this model knows that MTEX's pre-cleaning step
has to guess, and it is where the headroom is.

## Scoring

Four numbers, per map, then aggregated across maps:

| column | meaning |
|---|---|
| `ALL` | median disorientation over every pixel. Flattered by the 95% of pixels that were only lightly perturbed. |
| `CORRUPTED` | median over pixels the generator actually replaced. Did the method repair them? |
| `BOUNDARY` | median over pixels touching a grain boundary. Watch this whenever a method wins on `ALL`. |
| `>10 deg` | share of pixels still grossly wrong. **The headline number.** |

The `(IQR)` column is the spread across maps. A gap between two methods
smaller than their IQR is not a result.

### The baseline (MTEX, map 1)

```
method                  ALL   CORRUPTED   BOUNDARY   >10 deg
noisy (no filter)     0.522      42.329      0.576      4.95
clean-spline          0.026       0.032      0.029      1.48   <- best
clean-median          0.066       0.105      0.105      1.48
spline                0.027      42.297      0.029      4.94
```

Note that **all six `clean-*` methods share exactly 1.48%**. That number is set
by the pre-cleaning step, not by the filter: no choice of filter moves it. It
is ~242 pixels per map that the cleanup could not identify, and it is the gap a
learned detector can attack.

`clean-` means bad pixels were removed *before* filtering; the bare names are
the same filters on the raw noisy map. Filters cannot repair a wildly wrong
pixel, so the bare variants leave `CORRUPTED` at 42 deg.

## Running it

The `make` targets live in the parent folder's `Makefile`
(`/project/community/aiosman/Makefile`), so run them from there. Some of
them still point at old paths; see [Out-of-date paths](#out-of-date-paths).

```bash
make score                # single-map table (fast, for sanity checks)
make score-all            # aggregated over every map MTEX has finished
make plot                 # IPF and error maps -> figures/
make check-orientation    # verify the orientation maths against numpy
```

Training:

```bash
python src/prepare_cache.py --n-train 12000    # once, ~1 min
sbatch src/train_model.sh      --resume auto   # GPU,  checkpoints_gpu/
sbatch src/train_model_cpu.sh  --resume auto   # CPU,  checkpoints/
python src/inference.py --checkpoint checkpoints/best.pth
make score-all
```

Both job scripts checkpoint every epoch and take `--resume auto`, so a
preemption or a wall-clock kill costs at most the epoch in flight.

## What's in this folder

Counts and sizes are as of 2026-10-02. `datasets/` and `checkpoints/` are
gitignored and backed up to GCS every night by `src/sync_to_gcs.py`. Everything
else is tracked in git.

```
MAPS_DENOISING_Orientation_Noise/
|-- README.md                       this file
|-- src/                            all code
|-- datasets/                       maps, noise sets, MTEX outputs, cache  (~27 GB)
|-- checkpoints/                    trained models and training histories
|-- figures/                        plots and per-map score tables
`-- dataset_checkpoints_sync_log/   log of the nightly GCS backup
```

### `src/` -- code

| file | what it has |
|---|---|
| `orientation.py` | Orientation maths in numpy, used by the scorers. Euler-to-quaternion conversion, cubic disorientation, map loading and the grain-boundary mask. Its disorientation uses the classical cubic shortcut, ~67x faster than searching all 24x24 symmetry pairs. |
| `orientation_torch.py` | The same maths in torch, batched and differentiable, which the model is built on. Adds the 6D rotation representation, `disorientation_loss` (bounded gradient at zero error), slerp, and the edge-preserving local mean and misfit used as model inputs. `--self-test` checks it all against `orientation.py`. |
| `add_ebsd_noise.py` | The noise model: scatter plus misindexing, with optional bias toward grain boundaries. Takes one map or a whole folder (`--limit`, `--start`). `--save-mask` writes the `.badmask.npy` files and `--ang` writes the `.ang` files MTEX reads. Created `datasets/noise_scatter*/`. |
| `prepare_cache.py` | Packs the clean maps with id > 500 into `datasets/cache/train_clean.npy` (default: the first 12,000 usable ones). With `--noisy-dir` it also packs a test set (`test_*.npy`); that has not been done for the scatter sets. |
| `dataset.py` | PyTorch datasets. `TrainMaps` reads the cache and adds fresh noise on every access. `TestMaps` reads the packed test cache. `DirMaps` reads any clean/noisy folder pair straight from `.txt`, so it needs no cache. |
| `architecture.py` | The gated U-Net (refine, replace and gate heads) and its loss, `denoise_loss`. |
| `train.py` | Training loop. Validation reports the same four columns as the scorer. Writes `best.pth`, `last.pth` and `history.json` to `--checkpoints-dir`, and redraws `figures/training_curves.png` every epoch. |
| `inference.py` | Runs a checkpoint over the test maps and writes `<stem>__<method-name>.txt` in MTEX's output format, so the scorers treat it like any other method. Pass `--clean-dir` and `--noisy-dir` to read a noise folder directly instead of the test cache. |
| `score_filter_denoising.py` | Scores one map: ALL / CORRUPTED / BOUNDARY / >10 deg for every `<stem>__<method>.txt` in a results folder. |
| `score_dataset.py` | Scores a whole dataset: per-map scores, then the median and IQR across maps. `--metric mean` gives the mean-based numbers; `--only` and `--exclude` pick methods; `--csv` writes the per-map table. |
| `plot_maps.py` | Draws IPF-Z and per-pixel error figures for one map. `--only` picks methods and `--crop` zooms in. |
| `plot_method_comparison.py` | Draws bar charts of methods (all pixels vs boundary pixels), from a `score_dataset.py` CSV or from numbers recorded in the script (`--recorded scatter5`; `--compare scatter5 scatter7` for side by side). |
| `verify_orientation_noise.py` | Checks DREAM.3D's own "Add Orientation Noise" filter: how big the noise is, and whether it is applied per pixel or per grain. |
| `sync_to_gcs.py` | Nightly backup: `gcloud storage rsync` of `datasets/` and `checkpoints/` to `gs://cmu-gpucloud-aiosman/DATASET_CHECKPOINTS`. It never deletes from the bucket, and a lock file stops two runs overlapping. Cron runs it at 23:00 and appends its output to `dataset_checkpoints_sync_log/sync_log.log`. |
| `train_model.sh` | SLURM job: trains on 1 GPU, then runs inference and `score_dataset.py`. Arguments after the script name are passed to `train.py`; `CKPT_DIR` sets the checkpoint folder (default `checkpoints_gpu/`). |
| `train_model_cpu.sh` | SLURM job: CPU version of `train_model.sh` (32 cores, no AMP). Default checkpoint folder is `checkpoints/`; `METHOD` sets the output method name. |
| `__pycache__/` | Python bytecode. Safe to delete. |

### `datasets/` -- data

| folder | what it has | files | size |
|---|---|---|---|
| `clean_euler/` | **Ground truth.** `map_00001_clean_euler.txt` to `map_40000_clean_euler.txt`. Read-only. | 40,000 | 20 GB |
| `noise_scatter5/` | Maps 1-500 with scatter up to 5 deg (2.5 deg average) and no misindexing. | 1,000 + 1,000 in `ang_files/` | 1.7 GB |
| `noise_scatter7/` | Same, scatter up to 7 deg (3.5 deg average). | 1,000 + 1,000 | 1.7 GB |
| `noise_scatter9/` | Same, scatter up to 9 deg (4.5 deg average). | 1,000 + 1,000 | 1.7 GB |
| `mtex_out_scatter5/` | MTEX filter outputs for `noise_scatter5`, maps 1-10 only. | 120 | 56 MB |
| `mtex_out_scatter7/` | Same for `noise_scatter7`. | 120 | 56 MB |
| `mtex_out_scatter9/` | Same for `noise_scatter9`. | 120 | 56 MB |
| `cache/` | Packed training maps for `train.py`. | 2 | 2.2 GB |

Every `.txt` map has the same layout: a header line
`EulerAngles_0 EulerAngles_1 EulerAngles_2`, then 16,384 rows of three Euler
angles, one per pixel of the 128 x 128 grid (x varies fastest). The angles are
Bunge ZXZ in radians.

Each `noise_scatterN/` folder has four files per map:

| file | what it has |
|---|---|
| `map_XXXXX_clean_euler_noisy.txt` | The noisy Euler angles. |
| `map_XXXXX_clean_euler_noisy.txt.badmask.npy` | 128 x 128 bool, True where a pixel was replaced by misindexing. All False in these sets, which have no misindexing. |
| `ang_files/map_XXXXX_clean_euler_noisy_clean.ang` | The clean map in TSL `.ang` format (nickel, cubic, square grid, 1 um step). |
| `ang_files/map_XXXXX_clean_euler_noisy_noisy.ang` | The noisy map in `.ang` format. **This is what MTEX reads.** |

Each `mtex_out_scatterN/` folder has 12 files per map, named
`map_XXXXX_clean_euler_noisy_noisy__<method>.txt`:

| method | what it is |
|---|---|
| `mean`, `median`, `kuwahara`, `spline`, `halfquad`, `infconv` | the MTEX filter applied to the raw noisy map |
| `clean-mean`, `clean-median`, ... (the same six) | the same filter after MTEX's bad-pixel pre-cleaning |

`cache/` has:

| file | what it has |
|---|---|
| `train_clean.npy` | float32, shape (12000, 16384, 3): clean Euler angles for the first 12,000 usable maps with id > 500, one row per map. Made by `prepare_cache.py --n-train 12000`. |
| `train_clean_ids.npy` | int32, shape (12000,): the map id for each row. |

There are no `test_*.npy` files, so `TestMaps` and plain `inference.py` fail.
Use `inference.py --clean-dir ... --noisy-dir ...` instead.

### `checkpoints/` -- trained models

| path | what it has |
|---|---|
| `history.json` | Per-epoch log of the first run, on the old 1 deg + 5% misindexing set (40 epochs; final val >10 deg 0.47%, gate precision/recall 0.98/0.99). Its `.pth` files are no longer here. |
| `checkpoints_scatter5/` | Model trained on 5 deg scatter: `best.pth` (val ALL 0.209 deg vs 2.50 for the noisy input), `last.pth` (epoch 40), `history.json`. |
| `checkpoints_scatter7/` | Same for 7 deg: best val ALL 0.280 deg vs 3.50. |
| `checkpoints_scatter9/` | Same for 9 deg: best val ALL 0.314 deg vs 4.50. |
| `checkpoints_v1_no_localmean/history.json` | 3 epochs of an earlier model without the local-mean term. ALL stayed at 0.53 deg (no better than the input), which is the evidence behind the refinement design note above. |

Each `.pth` (~21 MB) holds `epoch`, `model`, `optimizer`, `scheduler`,
`best_score`, `history` and the `args` it was trained with, so `--resume auto`
can continue from it. Each `history.json` is a list with one entry per epoch:
`epoch`, `train_loss`, `lr`, `minutes`, and `val`, which has:

| key | meaning |
|---|---|
| `all`, `bad`, `bnd`, `gross` | the model's ALL, CORRUPTED, BOUNDARY and >10 deg values |
| `in_all`, `in_gross` | the same, for the noisy input |
| `gate_precision`, `gate_recall` | how well the gate finds the misindexed pixels (0 when there are none) |
| `score` | the model-selection score, `gross% + ALL median` |

### `figures/` -- plots and scores

| file | what it has |
|---|---|
| `scores.csv` | Per-map scores from the first (misindexing) run. Columns are `map, method, all_med, bad_med, bnd_med, gross_pct`. There are 500 maps for `noisy (no filter)` and `unet`, but only map 1 for the 12 MTEX filters. |
| `maps_ipf.png` | IPF-Z colour maps of map 1 for every method, from the first run. |
| `maps_error.png` | Per-pixel error maps for the same map, on a log scale. |
| `training_curves.png` | Loss and validation curves of the latest `train.py` run. Every run overwrites it; it currently shows the 9 deg scatter run. |
| `noise_level_comparison.png` | Bar chart of mean disorientation per method at 5 and 7 deg scatter, for all pixels and for grain-boundary pixels. U-Net is lowest at both levels. |
| `scatter5/scores.csv` | Per-map scores for 5 deg scatter: 10 maps x 8 methods (noisy, the six bare filters, unet). These are per-map **means** (`--metric mean`), although the columns are still named `_med`. |
| `scatter5/method_comparison.png` | The same kind of bar chart, for 5 deg only. |

### `dataset_checkpoints_sync_log/`

| file | what it has |
|---|---|
| `sync_log.log` | The output of every nightly `sync_to_gcs.py` run: timestamped start/finish lines plus gcloud's progress output (~10 MB). It is tracked in git, which is why it always shows as modified. |

### Outside this folder

| path | what it has |
|---|---|
| `../Makefile` | `make` targets for scoring, plotting, training and GCS. |
| `../logs/` | SLURM output. `sc5_*`, `sc7_*` and `sc9_*` are the three scatter training runs. |
| `../.gitignore` | Why `datasets/`, `checkpoints/`, `*.pth` and `*.npy` are not in git. |

## Out-of-date paths

The scatter-only data lives in new folders, but several defaults still point
at the old layout:

- `datasets/noisy_mis05/` and `mtex_out/` no longer exist. `make score`,
  `make plot`, `make score-all`, both `train_model*.sh` scripts and the default
  `--out` of `inference.py` still use them. Pass
  `--noisy-dir datasets/noise_scatterN` and
  `--results datasets/mtex_out_scatterN` explicitly.
- The scatter runs were trained with `CKPT_DIR=.../checkpoints_scatterN` and
  then moved into `checkpoints/`. Point `--checkpoint` at
  `checkpoints/checkpoints_scatterN/best.pth`, not `checkpoints/best.pth`.
- The inference step at the end of all three scatter jobs failed because
  `datasets/cache/test_clean.npy` does not exist (see `../logs/sc*_*.err`).
  Re-run it with `--clean-dir datasets/clean_euler --noisy-dir datasets/noise_scatterN`.

## Gotchas worth knowing

- **Crystal symmetry acts by LEFT multiplication** (`s*q`) under this
  convention. Right multiplication is not a symmetry and silently changes
  disorientations by tens of degrees. `orientation_torch.py --self-test` has a
  guard rail asserting this.
- **Never optimise `disorientation_deg` directly.** `d(arccos)/dx` is infinite
  at zero error, which is exactly where a working model converges, so it NaNs
  the moment it starts succeeding. Use `disorientation_loss`; keep the degree
  form for reporting.
- **float64 cannot resolve better than ~3e-6 deg through arccos**, since
  theta ~ 2*sqrt(2*eps) near zero. Check round trips as `1 - cos(theta/2)`.
- **The charbonnier epsilon sets a precision floor.** The loss turns from
  L1-like to L2-like below `2*sqrt(2*eps)` radians, which for the usual 1e-6 is
  0.16 deg -- above the ~0.03 deg being chased, cutting the gradient there by
  ~77x. The default here is 1e-10 (turnover at 0.0016 deg). It barely changes
  the gradient at 0.5 deg, so it only sharpens the endgame.
- **The word "clean" means two different things** in this repo: `clean_euler`
  and `_clean.ang` are the noise-free truth, while `clean-median` means "bad
  pixels removed, then median filter".
- Two source maps are unusable: `08231` is empty and `34360` is truncated,
  both from an interrupted generation run. (`06770` and `28160` used to be
  missing too, but are now present at full size.) `prepare_cache.py` filters
  out bad maps by size.

## Not in this repo

**The MTEX script that produced `mtex_out_scatter*/`.** There is no `.m` file anywhere
in the tree or in git history, so the middle step of the pipeline cannot
currently be reproduced. It also means it is unverified whether the `clean-`
pre-cleaning used the bad-pixel mask as an oracle; if it did, 1.48% is not an
honest baseline.
