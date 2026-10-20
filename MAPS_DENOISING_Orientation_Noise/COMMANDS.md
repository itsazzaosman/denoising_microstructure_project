# Commands: data, training, testing, figures

Every command below was run for this project (last checked 2026-10-14). Run
them **from `/project/community/aiosman`** unless a step says otherwise.

```bash
cd /project/community/aiosman
P=MAPS_DENOISING_Orientation_Noise          # the experiment folder
PY=~/miniconda3/envs/ebsd/bin/python        # Python with PyTorch (model code)
```

Plain `python3` is enough for scoring and plotting; anything that loads the
model or `dataset.py` needs `$PY`. **Training and testing run on a GPU only,
through the Slurm job scripts.**

Steps marked *(done)* have already been run; their outputs are on disk and in
the nightly backup. Repeat them only to rebuild from scratch.

---

## 1. Data

**1.1 Restore the data from the backup** (only if `datasets/` is missing):
```bash
gcloud storage rsync -r gs://cmu-gpucloud-aiosman/DATASET_CHECKPOINTS/datasets $P/datasets
gcloud storage rsync -r gs://cmu-gpucloud-aiosman/DATASET_CHECKPOINTS/checkpoints $P/checkpoints
```

**1.2 Split the maps** into test (1-500), validation (501-1000) and training
(1001-40000) *(done)*. Writes `configs/splits/*.txt` and link folders in
`datasets/splits/`.
```bash
python3 $P/src/make_splits.py
```

**1.3 Pack the training maps** into one file for fast loading *(done, ~6 min)*:
```bash
python3 $P/src/prepare_cache.py --jobs 8
```

**1.4 The validation set**: 500 maps, a random noise level each *(done)*:
```bash
make noise NOISE=val_random   # -> datasets/noise_scatter_misindexing_val/noise_val_random
```

**1.5 The test sets** *(done)*: 28 grid sets (scatter 1-15 deg x misindexing
0-9 %) and the mixed-level set `test_random`. The first command writes the 28
configs; `make noise` builds one set (~1.5 min each).
```bash
python3 $P/src/make_test_grid.py
for S in $(cat $P/configs/test_grid.txt) test_random; do make noise NOISE=$S; done
```

**1.6 Check the noise** that was applied:
```bash
python3 $P/src/summarize_test_grid.py      # input error of every test set -> figures/test_grid/input_noise.csv
$PY $P/src/check_noise.py                  # training + validation noise   -> figures/noise_check/
```

---

## 2. Training (GPU)

**2.1 Commit first**, so every result records a clean code version (no `-dirty`):
```bash
git add $P && git commit -m "describe the change"
```

**2.2 Optional timing test**: one epoch in a throw-away folder (~7 min):
```bash
CONFIG=blind_learned_seed1 METHOD=timing_test CKPT_DIR=$PWD/$P/checkpoints/runs/timing_test \
  sbatch --time=01:00:00 --job-name=timing_test $P/src/train_model.sh --epochs 1
```

**2.3 Train the three runs** *(done: about 5 h 50 min each on one H100)*. Each
config differs only in its seed. Your account runs one job at a time, so they
run one after another.
```bash
for s in 1 2 3; do
  CONFIG=blind_learned_seed$s sbatch --time=07:30:00 --job-name=blind_seed$s \
    $P/src/train_model.sh --resume auto
done
```
A run saves to `$P/checkpoints/runs/blind_learned_seed<s>/`: `best.pth`,
`last.pth`, `history.json`, `run_info.json`, `training_curves.png` (redrawn
every epoch). If a job stops, submit the same command again: `--resume auto`
continues from the last epoch.

**2.4 Watch the jobs:**
```bash
squeue -u $USER                                   # running (R) and waiting (PD) jobs
tail -f logs/orient_unet_<JOBID>.out              # live output (Ctrl+C stops watching only)
grep -E "^epoch|val  loss" logs/orient_unet_<JOBID>.out   # one line per epoch
cat logs/orient_unet_<JOBID>.err                  # errors
```

---

## 3. Testing (GPU)

**3.1 Run each model on all 29 test sets** *(done, ~15 min per run)*. The
predictions go to `$P/datasets/mtex_out_<set>/<map>__blind_learned_seed<s>.txt`.
```bash
for s in 1 2 3; do
  RUN=blind_learned_seed$s sbatch --time=01:00:00 --job-name=eval_seed$s $P/src/eval_model.sh
done
tail -f logs/orient_eval_<JOBID>.out
```

**3.2 Score every test set** from the saved predictions (CPU). One
table per set, `$P/figures/<set>/scores_mean.csv`: mean per-pixel error per
map, for every method present (noisy input, models, MTEX filters).
```bash
make score-grid
```

**3.3 One results table over all sets and runs** (mean of the three runs +-
their standard deviation): `figures/results/blind_learned_results.csv`, and
the error-vs-scatter figure `blind_learned_results.png/.pdf`.
```bash
python3 $P/src/summarize_results.py --runs blind_learned_seed1 blind_learned_seed2 blind_learned_seed3
```

**3.4 One test set, printed on screen** (e.g. to look at MTEX against the model):
```bash
make score-all NOISE=scatter5_mis6
```

---

## 4. MTEX baseline (Windows MATLAB, first results)

MTEX runs on the Windows computer, on a small package of test maps: the same
maps the network was scored on, so the two can be compared map by map.

**4.0 Run MATLAB SCRIPT** 
```bash
cd 'C:\Users\STUDENT\OneDrive\Documenten\MATLAB\mtex-7.1.0'
startup_mtex
cd 'C:\Users\STUDENT\OneDrive\Documenten\MATLAB\mtex-7.1.0\mtex_package\mtex_package\run_mtex_baseline.m'
which run_mtex_baseline -all
run_mtex_baseline
```

**4.1 Build the package** *(done)*: maps 1-10 of all 29 test sets, the MATLAB
script and a README, zipped (`$P/datasets/mtex_package.zip`, 71 MB):
```bash
python3 $P/src/make_mtex_package.py
```

**4.2 On Windows**: download the zip (in VS Code: right-click it > Download),
unzip, start MATLAB + `startup_mtex`, open `mtex_package/run_mtex_baseline.m`
and press Run. It runs the six filters (mean, median, Kuwahara, spline, half-quadratic,
infimal convolution), each with and without pre-cleaning, on map 1
of every set; set `maps = 1:10` at the top for all ten. Finished outputs are
skipped, so it can be stopped and started again.

**4.3 Bring the results back**: upload the folder `mtex_package/mtex_out` to
`$P/datasets/mtex_package/mtex_out`, then:
```bash
bash $P/src/import_mtex_results.sh       # into datasets/mtex_out_<set>/; old Sept outputs -> datasets/mtex_out_archive/
python3 $P/src/compare_with_mtex.py      # network vs MTEX on the same maps -> figures/mtex_comparison/
```
`compare_with_mtex.py` prints one row per set and a paired t-test across the
sets, and draws error vs scatter for the network and each filter.

*(Later, for all 500 maps: MATLAB R2026a and MTEX 7.1.0 are also installed on
the cluster in `/project/community/aiosman/matlab/`, not yet activated; then
`sbatch $P/src/run_mtex_grid.sh` runs every map.)*

---

## 5. Figures

**5.1 Training curves:**
```bash
python3 $P/src/plot_training.py $P/checkpoints/runs/blind_learned_seed1            # one run
python3 $P/src/plot_training.py $P/checkpoints/runs/blind_learned_seed{1,2,3} \
    --out $P/figures/training/seeds_comparison.png                                  # three runs together
```

**5.2 Paper figures** (into `$P/figures/paper/`):
```bash
python3 $P/src/plot_noise_examples.py --map 1      # the noise model: noise_examples.pdf
python3 $P/src/make_results_figures.py --runs blind_learned_seed1 blind_learned_seed2 blind_learned_seed3
                                                   # training_loss.pdf, denoised_examples.pdf
cp $P/figures/results/blind_learned_results.pdf $P/figures/paper/results_vs_scatter.pdf
```

**5.3 Look at any test map**: truth, input, denoised, errors; `--circle`
marks the pixels still more than 10 deg wrong:
```bash
python3 $P/src/plot_error_maps.py --run blind_learned_seed1 --map 1 \
    --sets scatter15 scatter15_mis9 --circle      # -> $P/figures/error_maps/
```

---

## 6. Paper

The text is in `$P/paper/` (`main.tex` includes `data_and_noise.tex` and
`results.tex`); the figures are read from `$P/figures/paper/`.
```bash
conda install -c conda-forge tectonic            # once
cd $P/paper && tectonic main.tex                 # -> main.pdf
```

---

## Old per-level models (before the single blind model)

The scatter-only models `checkpoints/checkpoints_scatter{5,7,9}` (one per
noise level, configs `configs/train/scatter{5,7,9}.yaml`) can still be run:
```bash
for s in 5 7 9; do make infer NOISE=scatter$s; done
```


# Gitignore

## How to ignore files in this repo


## Pattern cheatsheet

| Pattern | Meaning |
|---|---|
| `datasets/` | the whole directory (preferred) |
| `logs/*` | contents of `logs/`, directory itself still visited |
| `*.h5` | that extension anywhere in the repo |
| `**/euler_maps/` | that directory name at any depth |
| `/build` | only `build` at the repo root |
| `!keep.txt` | exception — un-ignore this file. Put it *after* the broad rule, and it will **not** work if the parent was ignored as `dir/` (use `dir/*` + `!dir/keep.txt`) |