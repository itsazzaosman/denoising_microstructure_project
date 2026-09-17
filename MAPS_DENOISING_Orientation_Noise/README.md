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

## Pipeline

```
DREAM.3D                                    (outside this repo)
   |  synthetic microstructures -> Euler angles per pixel
   v
datasets/clean_euler/map_XXXXX_clean_euler.txt      GROUND TRUTH, 34,359 maps
   |
   |  src/add_ebsd_noise.py      adds scatter + misindexing
   v
datasets/noisy_mis05/            maps 1-500, four files each
   |-- ..._noisy.txt                  noisy Euler angles
   |-- ..._noisy.txt.badmask.npy      which pixels were replaced
   |-- ..._noisy_clean.ang            truth, in MTEX's format
   `-- ..._noisy_noisy.ang            noisy, in MTEX's format   <-- MTEX INPUT
   |
   |  MTEX (MATLAB)              12 classical filters
   v
mtex_out/..._noisy_noisy__<method>.txt
   |
   |  src/prepare_cache.py       pack clean maps into one .npy  (once)
   |  src/train.py               train the model on maps 501+
   |  src/inference.py           predict maps 1-500 -> mtex_out/..._unet.txt
   v
make score-all                   one table, filters and model side by side
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

## Files

| file | what it is |
|---|---|
| `src/orientation.py` | numpy orientation maths shared by the scorers |
| `src/orientation_torch.py` | the differentiable version; `--self-test` checks it against numpy |
| `src/add_ebsd_noise.py` | the noise model, and `.ang` export for MTEX |
| `src/prepare_cache.py` | packs clean maps into one memmappable `.npy` |
| `src/dataset.py` | on-the-fly corruption for training, fixed noise for test |
| `src/architecture.py` | the gated U-Net and its loss |
| `src/train.py` | training loop; reports the same four columns as the scorer |
| `src/inference.py` | writes predictions into `mtex_out/` as `__unet.txt` |
| `src/score_filter_denoising.py` | single-map scoring |
| `src/score_dataset.py` | whole-dataset scoring with across-map spread |
| `src/plot_maps.py` | IPF and per-pixel error figures |

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
- Four source maps are unusable (`06770`, `08231`, `28160`, `34360`) -- short,
  empty, or missing from an interrupted generation run. `prepare_cache.py`
  filters them out by size.

## Not in this repo

**The MTEX script that produced `mtex_out/`.** There is no `.m` file anywhere
in the tree or in git history, so the middle step of the pipeline cannot
currently be reproduced. It also means it is unverified whether the `clean-`
pre-cleaning used the bad-pixel mask as an oracle; if it did, 1.48% is not an
honest baseline.
