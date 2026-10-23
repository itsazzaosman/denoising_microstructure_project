# AGENT.md

Instructions for AI coding agents working in this repository.

The active project is **`MAPS_DENOISING_Orientation_Noise/`**: a neural network
that cleans noisy EBSD crystal-orientation maps (128 × 128 pixels, cubic
symmetry), compared against the MTEX filters for a paper. The other top-level
folders are older or separate work; leave them alone unless asked.

| Folder | What it is |
|---|---|
| `MAPS_DENOISING_Orientation_Noise/` | The active project: data, model, evaluation, paper |
| `Makefile`, `make/*.mk` | All everyday commands (run from the repo root) |
| `Dataset_creation/` | EBSD simulation pipeline (how the synthetic data was made) |
| `MAPS_DENOSING_Gaussian_Noise/` | An earlier project: Gaussian noise on RGB images |
| `logs/` | Slurm job logs (gitignored) |

Current status and the remaining work are in
`MAPS_DENOISING_Orientation_Noise/RESEARCH_PLAN.md` (sections 3 and 5). Parts of
its `README.md` are out of date (see Known traps), so trust the plan and the
code over the README.

---

## Working with the user

- **Explain plainly.** The user has limited materials-science background. Give
  the main answer first, in simple words, explain any technical term briefly,
  and keep replies short. Use tables only for real data, not to structure an
  explanation. (Also in `.claude/CLAUDE.md`.)
- **GPU only for training and model inference.** Never train on a CPU and never
  submit jobs yourself. Prepare the code, then give the user the `make` command
  to run (e.g. `make train-gpu CONFIG=<name>`, `make eval-val`). A tiny CPU check
  of code (`--limit 2`, output to a scratch folder) is fine. Say it is a test,
  not a result, and delete its output.
- **Do not commit or push unless asked.**
- **When asked to "just explain" or "don't execute", run nothing.**
- **Ask before deleting or overwriting data.** `datasets/` and `MODEL_PIPELINE/checkpoints/` are
  not in git; a deleted file cannot be recovered from the repo.
- For `.gitignore` questions, point to the "Gitignore" section of
  `MAPS_DENOISING_Orientation_Noise/COMMANDS.md`.

---

## Project layout

```
MAPS_DENOISING_Orientation_Noise/
  MODEL_PIPELINE/   the network: src/, configs/train/, checkpoints/, figures/
  MTEX_PIPELINE/    the MTEX baseline: src/ (MATLAB, packaging, import), figures/
  SHARED/           used by both: src/ (test sets, orientation maths, paths,
                    scoring, model-vs-MTEX comparison), configs/ (noise/,
                    splits/, test_grid.txt), figures/ (score tables, slides,
                    paper figures), dataset_checkpoints_sync_log/
  datasets/         all data (gitignored)
  paper/            the LaTeX paper (reads SHARED/figures/paper/)
  README.md, COMMANDS.md, RESEARCH_PLAN.md
```

Put new code in the folder it belongs to: model-only in `MODEL_PIPELINE/src`,
MTEX-only in `MTEX_PIPELINE/src`, anything both use (or that compares them) in
`SHARED/src`.

---

## The pipeline, with the file for each step

All paths below are relative to `MAPS_DENOISING_Orientation_Noise/`.

| Step | File | Output |
|---|---|---|
| Clean maps (ground truth, from DREAM.3D) | downloaded | `datasets/clean_euler_zip_unzip/clean_euler/` (40,000 maps) |
| Split train / val / test | `SHARED/src/make_splits.py` | `SHARED/configs/splits/*.txt`: test 1–500, val 501–1000, train 1001–40000 |
| Pack the training maps | `MODEL_PIPELINE/src/prepare_cache.py` | `datasets/cache/split_train_clean.npy` (38,998 maps) |
| Make fixed noisy test and validation sets | `SHARED/src/add_ebsd_noise.py` + `SHARED/configs/noise/*.yaml` | `datasets/noise_scatter_misindexing{,_val}/` |
| Train (noise added on the fly, every epoch) | `MODEL_PIPELINE/src/train.py`, `MODEL_PIPELINE/src/dataset.py`, `MODEL_PIPELINE/src/architecture.py`, run by `MODEL_PIPELINE/src/train_model.sh` | `MODEL_PIPELINE/checkpoints/runs/<RUN>/` |
| Predict a test set | `MODEL_PIPELINE/src/inference.py`, run by `MODEL_PIPELINE/src/eval_model.sh` | `datasets/results_test_set_model/<set>/` |
| MTEX filters (MATLAB) | `MTEX_PIPELINE/src/run_mtex_grid.m` / `.sh` | `datasets/results_test_set_mtex/<set>/` |
| Score | `SHARED/src/score_dataset.py`, `SHARED/src/compare_with_mtex.py` | `SHARED/figures/<set>/scores_mean.csv`, `SHARED/figures/mtex_comparison/` |
| Tables, figures, statistics | `SHARED/src/make_slide_tables.py`, `SHARED/src/make_slide_figures.py`, `SHARED/src/plot_nonsignificant.py` | `SHARED/figures/slides/`, `SHARED/figures/paper/` |
| Paper (LaTeX) | `paper/main.tex` | `cd paper && tectonic main.tex` |

The model is described in the docstring at the top of `MODEL_PIPELINE/src/architecture.py`.

---

## Where data lives

- **`SHARED/src/paths.py` is the single source of folder paths for Python.** The
  Makefile repeats `CLEAN_DIR`, `TEST_SETS` and `VAL_SETS`. The shell scripts
  (`train_model*.sh`, `eval_model.sh`, `import_mtex_results.sh`) and
  `run_mtex_grid.m` hard-code paths. When moving data, update all of them; the
  `paths.py` docstring lists the places.
- `datasets/` and `MODEL_PIPELINE/checkpoints/` are gitignored and large. They are backed up
  nightly to `gs://cmu-gpucloud-aiosman` by `SHARED/src/sync_to_gcs.py`.
- **The clean maps are read-only ground truth.**

```
datasets/
  clean_euler_zip_unzip/clean_euler/           ground truth, map_XXXXX_clean_euler.txt
  noise_scatter_misindexing/noise_<set>/       the 29 test sets (maps 1-500)
  noise_scatter_misindexing_val/noise_<set>/   the validation set, val_random (maps 501-1000)
  results_test_set_model/<set>/                network predictions on each test set
  results_test_set_mtex/<set>/                 MTEX filter outputs on each test set
  results_val_set_model/<set>/                 network predictions on the validation set
  splits/<split>/clean_euler/                  symlinks to the clean maps of each split
  cache/                                       packed training maps
```

---

## Naming rules

- **Noise sets:** `SHARED/configs/noise/<name>.yaml` builds the folder `noise_<name>/`.
  A name starting with `val_` is a validation set and goes to
  `noise_scatter_misindexing_val/`; every other name is a test set. The
  Makefile, `paths.noise_dir()` and `eval_model.sh` all apply this rule.
- **Test grid:** `scatter<S>` (no misindexing) or `scatter<S>_mis<M>`, with
  S ∈ {1, 3, 5, 7, 9, 12, 15} degrees and M ∈ {3, 6, 9} %. That is 28 sets,
  listed in `SHARED/configs/test_grid.txt`, plus the mixed set `test_random`. They
  are written by `SHARED/src/make_test_grid.py`.
- **Predictions:** `<stem>__<method>.txt` inside the set's results folder, so
  several methods sit side by side and are scored together.
- **Training runs:** `MODEL_PIPELINE/configs/train/<RUN>.yaml` trains into
  `MODEL_PIPELINE/checkpoints/runs/<RUN>/`. The current model is `blind_learned_seed{1,2,3}`:
  three runs identical except for the seed.
- **"clean" means two things:** `clean_euler` and `_clean.ang` are the
  noise-free truth, while `clean-median` (and the other `clean-*` filters)
  means "small grains removed, then the filter applied".

---

## Running things

Run `make` from the repo root. Settings live in the root `Makefile`; targets
live in `make/cloud.mk`, `make/data.mk`, `make/train.mk` and `make/eval.mk`.
Override a setting for one command with `NAME=value`. Use `make -n <target>`
to print the commands without running them.

```bash
make noise NOISE=scatter5_mis3                           # build one noisy set from its config
make preview MAP=00001 SCATTER="1 5 9" MISINDEX="0 9 15" # picture of noise levels, no data written
make check-orientation                                   # self-test of the orientation maths
make train-gpu CONFIG=blind_learned_seed1                # submit training (user runs this)
make watch                                               # the user's Slurm queue
make eval-val                                            # score RUNS on the validation set (GPU jobs)
RUN=<run> sbatch MAPS_DENOISING_Orientation_Noise/MODEL_PIPELINE/src/eval_model.sh   # all 29 test sets
make score-grid                                          # re-score every test set (CPU, ~20 min)
make upload / make download                              # sync datasets with the GCS bucket
```

- **Two Python environments.** Use plain `python` (numpy, scipy, matplotlib,
  yaml) for data, scoring and figures. Use `~/miniconda3/envs/ebsd/bin/python`
  for anything that imports torch.
- **Slurm.** `train_model.sh` runs on the `preempt` partition and resumes with
  `--resume auto` from `last.pth` after an eviction. `eval_model.sh` runs on
  `general`. Logs go to `/project/community/aiosman/logs/`.

---

## Experiment rules (do not break these)

1. **Never edit a config once its run has started.** Copy it under a new name
   instead, so each config still describes the results next to it.
2. **Every error number is a mean** (each map's mean, then the mean across
   maps), decided 2026-10-13. The one exception is the saved validation history
   of `blind_learned_seed1-3`, trained 2026-10-12, which holds medians; their
   `best.pth` was chosen on the median-based score. Never mix the two in one
   comparison.
3. **Test and validation sets are fixed files.** Never regenerate them with
   different settings. Seeds: the grid uses 0 + map index (shared by every cell,
   so the cells form a paired design), validation uses 1000–1499 (config
   `seed: 500` with `start: 500`), and `test_random` uses 2000 + index. Training
   noise is unseeded and new every epoch.
4. **Never train on maps 1–1000.** They are the test and validation splits.
5. **MTEX comparison conventions:** maps 1–10 of each set, the six `clean-*`
   filters, MTEX given the true scatter level for its grain threshold. Tests
   compare against the closest MTEX filter per measure: paired t-test over the
   maps, Holm-corrected over the 29 sets (`make_slide_tables.test`).
6. **Commit before a run that will go in the paper.** Every result records the
   git commit that produced it (`-dirty` if `src/` or `configs/` had
   uncommitted changes); see `SHARED/configs/README.md`.
7. **Measure orientation error as the disorientation angle** (cubic symmetry,
   `orientation.disorientation_deg`), never by subtracting Euler angles.

---

## Code conventions

- **Every script opens with a docstring:** what it does, what it reads and
  writes, and a `Usage` block. Options use argparse. Folders come from
  `paths.py`. Copy this shape for new scripts.
- **Imports across folders:** every script sets `ROOT = Path(__file__).resolve().parents[2]`
  (the project folder). Scripts in `MODEL_PIPELINE/src` and `MTEX_PIPELINE/src` that
  import shared modules first add `SHARED/src` to `sys.path` (copy the two lines
  from `train.py`). Shared scripts never import model or MTEX modules.
- **Match the surrounding code.** Comments explain *why*, not *what*.
- **Orientation maths:** quaternions are `(w, x, y, z)`; Euler angles are Bunge
  ZXZ in radians. Crystal symmetry acts by **left** multiplication (`s * q`).
  Train with `disorientation_loss`, report with `disorientation_deg`.
- **Figures:** matplotlib with the `Agg` backend; colours from
  `compare_with_mtex.COLOURS` (ours = `#2a78d6`); save both `.png` and `.pdf`.
  Slide figures go in `SHARED/figures/slides/`, paper figures in `SHARED/figures/paper/`.
- **Reuse the existing loaders and tests** (`make_slide_tables.load` / `test`,
  `make_slide_figures` helpers) rather than recomputing results a second way.

---

## Checking your work

- `make check-orientation` after touching `MODEL_PIPELINE/src/orientation_torch.py`.
- `make -n <target>` for Makefile changes; `bash -n <script>` for job scripts.
- Small runs with `--limit` into a scratch folder before anything long.
- When moving or rebuilding data, compare the new files byte for byte with the
  old ones before switching any code to them.

---

## Known traps

- **`make cache` passes `--n-train 12000`.** Running it would overwrite the
  38,998-map `split_train_clean.npy` with only 12,000 maps.
- **Makefile:** settings must stay above the `include` line. Don't put a
  comment on the same line as an assignment: the spaces before `#` become part
  of the value.
- **Two source maps are unusable:** `08231` (empty) and `34360` (truncated).
  `prepare_cache.py` filters them out.
- **`README.md` is partly stale.** Its intro describes an older scatter-only
  experiment, and "Not in this repo" says the MTEX script is missing, but
  `MTEX_PIPELINE/src/run_mtex_grid.m` exists. `RESEARCH_PLAN.md` section 5.8 lists the docs
  that still need updating.
- **`datasets/splits/val/noise_random/` is an old copy** of the validation set,
  byte-identical to `noise_scatter_misindexing_val/noise_val_random/`. Nothing
  uses it any more.
- **`SHARED/dataset_checkpoints_sync_log/sync_log.log` is tracked in git**, so it
  always shows as modified. Leave it out of commits unless asked.
- **The home folder hit its disk quota on 2026-10-14**, which can make conda or
  pip writes fail.
