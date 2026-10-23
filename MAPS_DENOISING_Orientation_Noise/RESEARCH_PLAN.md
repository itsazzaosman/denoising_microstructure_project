# Research plan: learned denoising of EBSD orientation maps

Written 2026-10-11, after reading the 8 papers in the `papers_ebsd_ml` Drive
folder. Rewritten 2026-10-14 as **what is done** (Section 4) and **what
remains** (Section 5), checked against the code, data and paper on that day.
Tags: **MUST**, **SHOULD**, **NICE**.

---

## 1. The story of the paper (one paragraph)

Orientation maps from EBSD contain two kinds of error: small random tilts on
every pixel (*scatter*) and pixels that are completely wrong (*misindexing*).
Today people clean maps with hand-designed filters (MTEX: Hielscher et al.
2019; weighted TV flow: Atindama et al. 2023). Those methods were tested on
about 10 synthetic maps, with one error number, and never checked against a
trustworthy reference on real data. Deep-learning work in EBSD has focused on
the *diffraction patterns* (Andrews 2023, Mani Krishna 2023, Prajapat 2026),
which only helps labs that saved the raw patterns. Most labs did not. **We
propose a learned denoiser that works on the orientation map itself, so it
can be applied to any existing map. We test it on a large, statistically
powered benchmark covering both noise types, with fairly tuned baselines,
and validate it on real repeated scans of the same nickel sample.**

---

## 2. What the papers did and where they fall short

| Paper | What it does | Main weakness | What we do instead |
|---|---|---|---|
| Atindama 2023 (closest to us) | Weighted TV flow on **maps**, auto stopping | 10 maps per noise level; mean error only; boundaries judged by eye; real data judged by eye only; works on Euler angles (needs wraparound hacks); flattens grains (their own stated limit) | 100–500 maps, boundary metrics, real data with a reference, symmetry-aware maths, gradient test |
| Hielscher 2019 (MTEX filters, cited by Atindama) | The classical filters we compare against | Each filter needs a parameter tuned to the noise level | Tune every filter on separate maps before testing |
| Shen 2019 | CNN reads patterns → orientation | Big errors near the edge of the "fundamental zone" (a representation jump); real "truth" taken from Hough indexing; train/test pixels likely from the same grains | 6D representation + symmetry-aware loss (already done); test maps are different microstructures |
| Ding 2020 | CNN indexing with a disorientation loss | Not robust to noise it never saw in training; on real data, accepted indexing methods disagree by **1.1–1.5° on average**, so the "truth" is uncertain | Train on a range of noise; measure the reference's own uncertainty on real data |
| Andrews 2023 | Denoising autoencoder on **patterns** | No ground truth: judged by CI, IQ, fit (self-consistency scores, not accuracy); one sample | Judge by orientation error against an independent reference |
| Mani Krishna 2023 | cGAN on **patterns**, paired good/bad scans of the same area | Fails at grain boundaries (too few boundary examples in training); test set 500 patterns; few numbers on orientation accuracy | Boundary metric + boundary-biased misindexing; borrow their repeat-scan idea for real-data validation |
| Prajapat 2026 | Diffusion model on **patterns** | One 50×50 map; trained on 10% of pixels and tested on the other 90% of **the same map** (leakage); judged by CI/PQ/SSIM, not orientation error; data not public | Separate test microstructures; public benchmark |
| Kaufmann 2021 | Few-shot classification of patterns | (Not a denoiser.) Good practice: 5 repeated trainings with std, a test on a completely new sample | Copy the practice: 3 seeds, unseen-sample test |
| Martinez Ostormujof 2022 | U-Net on raw quaternions (phase segmentation) | Found that quaternions must be made continuous across a grain; the network took "shortcuts". Good practice: 5 seeds, learning curves, a test on a new heat treatment | Symmetry check (Step 4), robustness tests (Step 7) |

**The gap we fill:** a map-level learned method, evaluated with enough maps
for real statistics, against fairly tuned baselines, with boundary-specific
metrics, and validated on real data against an independent reference. None of
the 8 papers does all of these.

---

## 3. Where we are (2026-10-14)

| Part | Status |
|---|---|
| Data, splits, 28 test sets + mixed set | **Done** |
| Blind model, 3 seeds, scored on all 29 test sets | **Done** |
| MTEX filters (6 `clean-*`, maps 1–10, 29 sets) | **Done** (2026-10-14) |
| Model vs MTEX comparison | **Done**: in the paper (figure, table, example maps, appendix) |
| Local-mean-only baseline and ablations | Not started |
| Statistics | Per-set paired t-test, Holm, 95% CI **done**; boundary F1 not started |
| Robustness tests | Not started |
| Real data (ni_gain) | Not started |
| Paper | Abstract, Methods (data, noise, MTEX filters) and a Results draft written. Introduction, Related work, Network, Training, Discussion and Conclusion still empty. |

### The problems found on 2026-10-11, and their status

| # | Problem | Status |
|---|---|---|
| 1 | Data and checkpoints not on disk | Fixed: restored from GCS |
| 2 | MTEX not tuned, model retrained per level | **Open.** The model is now one blind model, but the MTEX filters are still not tuned (see Remaining). |
| 3 | MTEX on 10 maps only | Kept on purpose: 10 maps per set, now for all 29 sets (decided 2026-10-14) |
| 4 | MTEX grain threshold too small; variant-B loop commented out | Fixed: threshold max(10°, 2θmax); script rewritten |
| 5 | Local mean aligned neighbours by sign only | Fixed: aligned over all 24 cubic variants (2026-10-11) |
| 6 | No misindexing test set | Fixed: misindexing 3, 6 and 9% in the grid |
| 7 | Noise levels far above real EBSD | Fixed: 1° and 3° levels added |
| 8 | No "local mean only" baseline | **Open** (A1 below) |
| 9 | One seed, not converged | Fixed: 3 seeds of 60 epochs. Best epochs 58, 50, 58, so two runs were still improving near the end (see Remaining). |

---

## 4. Done

### Pipeline and data
- [x] Data and checkpoints restored from GCS; the nightly sync works again.
- [x] One file for all data locations: `SHARED/src/paths.py` (2026-10-14).
- [x] Results split into two folders (2026-10-14):
  `datasets/results_test_set_model/<set>/` (network) and
  `datasets/results_test_set_mtex/<set>/` (MTEX). The September 2026 MTEX
  outputs are kept apart in `datasets/results_test_set_mtex_old/`.
- [x] One config file per experiment in `configs/`; the git commit is saved
  next to every results file (`*.provenance.json`).
- [x] All commands to train, test and plot in `COMMANDS.md`.
- [x] Splits: test maps 1–500, validation 501–1000, training 1001–40000
  (38,998 maps; 2 broken maps left out).

### Test sets (Step 2, 2026-10-12)
- [x] 28 sets: scatter max 1, 3, 5, 7, 9, 12, 15° × misindexing 0, 3, 6,
  9%, 500 maps each, boundary bias 4. One shared seed (0), so two sets
  differ only in the amount of noise.
- [x] Mixed set `test_random`: every map has its own random noise level.
- [x] Input noise of every set measured: `SHARED/figures/test_grid/input_noise.csv`.
- [x] `.ang` copies of every set for MTEX.

### Model (Step 4)
- [x] Neighbours aligned over all 24 cubic symmetry variants (2026-10-11).
- [x] Blind model: never told the noise level; trained on scatter 0–15° and
  misindexing 0–10%, drawn afresh for every map.
- [x] 3 seeds, 60 epochs each, about 5.8 h per run on one H100; best
  checkpoints at epochs 58, 50 and 58.
- [x] Scored on all 29 test sets (500 maps each). It removes 86–93% of the
  mean error in every condition; mixed set 5.69° → 0.59°; spread across
  seeds at most 0.03°. Gate (misindexing detector) on validation: recall
  98.8%, precision 99.9%.
- [x] Speed: 500 maps in 21–23 s on one H100, about 0.04 s per map.

### Metrics (Step 6, part)
- [x] Four columns, all means (decided 2026-10-13): ALL, CORRUPTED,
  BOUNDARY, >10°.
- [x] Fraction of noise removed (1 − output error / input error).
- [x] The map is the unit; paired t-test (used for the 10-map September
  MTEX comparison in `results.tex`).

### MTEX baseline (Step 3, part)
- [x] `run_mtex_baseline.m` rewritten: runs from the package folder with
  no paths to edit, all filters in one list, restartable, logs the time of
  every output.
- [x] Grain threshold max(10°, 2θmax) per set (per map for the mixed set).
- [x] Edge bug fixed (`'fill', 'extrapolate'`): no empty pixels at the map
  edge.
- [x] Package with the 290 `.ang` files: `datasets/mtex_package.zip`.
- [x] Decided (2026-10-14): the 6 `clean-*` filters only, maps 1–10 of each
  of the 29 sets. The plain filters are skipped: they leave misindexed
  pixels untouched (MTEX does not filter grains under 5 pixels), and with
  no misindexing they give the same result as `clean-*` (checked on map 1).
- [x] Run on the Windows PC; 1,740 outputs, no empty pixels; imported to
  `datasets/results_test_set_mtex/` (2026-10-14).

### Paper (Step 9, part)
- [x] Abstract (draft, `paper/abstract.tex`).
- [x] Methods: data, noise model, test conditions (`paper/data_and_noise.tex`);
  baseline filters (`paper/mtex_baselines.tex`).
- [x] Results draft (`paper/results.tex`) and figures: noise examples,
  training loss, results vs scatter, denoised examples.

---

## 5. Remaining

### 5.1 MTEX comparison — done 2026-10-14

- [x] Imported (`bash MTEX_PIPELINE/src/import_mtex_results.sh datasets/mtex_out`); old upload deleted.
- [x] `python3 SHARED/src/compare_with_mtex.py`: scores, per-set paired t-tests with
  Holm correction and 95% CIs, figures (`SHARED/figures/mtex_comparison/`,
  `SHARED/figures/paper/mtex_comparison.pdf`).
- [x] `python3 SHARED/src/make_mtex_tables.py`: main table and appendix tables
  (`paper/table_mtex_main.tex`, `paper/table_mtex_full.tex`).
- [x] `python3 SHARED/src/plot_mtex_examples.py`: example maps
  (`SHARED/figures/paper/mtex_examples.pdf`).
- [x] Results section, abstract and timings updated.
- Result: the network is lower than the best filter in all 29 sets, significantly
  in 28 (not in scatter3, where the spline filter is close).
- [ ] Still open: `[processor]` and `[how these values were chosen]` in
  `paper/mtex_baselines.tex`.

### 5.2 A fair comparison (MUST)

- [ ] **Local-mean-only baseline (A1, answers H3).** Our edge-preserving
  local mean with no network, on all 500 maps of every set. Without it a
  reviewer can say the gain does not come from learning.
- [ ] **MTEX tuning: decide.** The filters use near-default settings. Either
  tune them (tuning maps 501–520 at the grid settings, grid-search each
  filter's strength per condition, then freeze) or say plainly in the paper
  that they are not tuned. H1 as written ("the best *tuned* filter") needs
  the tuning. Note that MTEX is given the true noise level through the grain
  threshold, and the network is not.
- [ ] **Checkpoint choice.** The 3 runs picked their best checkpoint with a
  score built from **medians** (the code switched to means on 2026-10-13),
  and two runs peaked at epoch 58 of 60. Either state both in the Methods,
  or retrain longer with the mean score (GPU only, `train_model.sh`).
- [ ] Say in the paper that `clean-*` does not use the true bad-pixel mask
  (it only deletes grains under 3 pixels). Partly done in
  `mtex_baselines.tex`.

### 5.3 Statistics (Step 6, MUST)

- [ ] Per map: model error − best MTEX filter error (same map, same noise).
- [ ] Paired t-test on the 10 maps per set, bootstrap 95% interval of the
  mean difference (resample maps 10,000 times), Holm correction across the
  29 sets.
- [ ] Claim a win only if the interval excludes zero **and** the gap is
  bigger than the spread across seeds.
- [ ] **Boundary F1:** find boundaries (neighbour disorientation > 5°, check
  that 10° gives the same ranking) in the denoised map and compare with the
  true boundaries.

**Done when:** one CSV per set plus a summary table with intervals and
corrected p-values.

### 5.4 Ablations (Step 5: A1 MUST, the rest SHOULD)

An *ablation* removes one part and measures the drop. All training on the GPU.

- **A1** local mean only, no network (5.2).
- **A2** network without the local mean (`--mean-radius 0`).
- **A3** no local-misfit input (`--no-misfit`).
- **A4** per-level models vs the blind model. The old per-level U-Nets
  exist for scatter 5/7/9 only (`unet`, scatter only).
- **A5** sign-only vs symmetry-aware alignment.

One seed is acceptable for A2–A5 if the paper says so.

### 5.5 Robustness tests (Step 7, SHOULD)

Training and test noise come from the same generator, so good scores may
partly mean "learned this generator's habits". Each test changes one thing
the model never saw.

- [ ] **Symmetry relabelling:** store each test pixel (a) in the fundamental
  zone, as vendor software does, and (b) as a random symmetry variant.
  Scores must not change. Note: training does **not** yet include a
  random-variant augmentation (only the neighbour alignment was done), so
  run this test before claiming the model handles real vendor files.
- [ ] **Noise outside the training range.** A reviewer will ask what happens
  when the noise is stronger than anything in training (scatter 0–15°,
  misindexing 0–10%). No retraining is needed.
  - Test sets: `scatter18`, `scatter20` (more scatter), `scatter5_mis15`,
    `scatter5_mis20` (more misindexing), `scatter20_mis15` (both).
  - Each needs a copy of `SHARED/configs/noise/scatter15_mis9.yaml` with new
    `scatter` / `misindex` values (keep `seed: 0`), then
    `make noise NOISE=<set>`.
  - One GPU job per seed (`eval_model.sh` takes one RUN):
    `RUN=blind_learned_seed1 SETS="scatter18 scatter20 scatter5_mis15 scatter5_mis20 scatter20_mis15" sbatch MODEL_PIPELINE/src/eval_model.sh`,
    then seed2 and seed3. About 22 s per set per seed.
  - Write one honest sentence in the Discussion: "the error grows gradually
    beyond the training range" or "it breaks down above X".
- [ ] **Different noise shape:** same average error, but a bell-shaped tilt
  angle (von Mises–Fisher, or the de la Vallée Poussin noise Atindama used)
  instead of uniform 0–max.
- [ ] **Grains with internal orientation gradients:** apply a smooth rotation
  that grows across the map (e.g. 0 → 5° over 128 px) to the clean maps, then
  add noise. Check whether the model flattens the gradient (Atindama's stated
  weakness).
- [ ] **Different microstructure (NICE):** a few DREAM.3D maps with smaller
  grains (more boundaries).

### 5.6 Real data (Step 8, MUST for a strong paper)

Biggest gap in all 8 papers: map-level papers judged real data by eye,
pattern-level papers by CI/IQ (self-consistency, not accuracy).

**Data:** *ni_gain* (Ånes et al.), 10 scans of the **same area** of
recrystallised nickel at camera gain 0–24 dB. Map 149 × 200, CC BY 4.0,
doi:10.5281/zenodo.7497682, via
`kikuchipy.data.ni_gain(number=k, allow_download=True)`.

- [ ] Install in the `ebsd` env: `pip install kikuchipy pyebsdindex orix`
  (still not installed on 2026-10-14).
- [ ] Hough-index all 10 scans (PyEBSDIndex): 10 real noisy maps.
- [ ] Reference: dictionary indexing + refinement of the lowest-gain scan.
- [ ] Reference floor: disagreement between two good indexings of the 0 dB
  scan. Smaller differences cannot be claimed.
- [ ] Check drift between scans; register the maps if they shifted.
- [ ] Denoise each map with the blind model (no retuning) and the MTEX
  filters (frozen settings). Pad the map to a multiple of 8.
- [ ] Plot error vs reference and >10° fraction against gain, for raw /
  MTEX / ours.

**Done when:** one figure and one table.

### 5.7 Writing (Step 9)

- [ ] Title, Introduction, Related work (use the table in Section 2).
- [ ] Methods: network (learned neighbour weights, symmetry alignment, gate),
  training (loss, optimiser, 60 epochs, 3 seeds, checkpoint choice),
  statistics.
- [ ] Hypotheses H1–H5 (Section 6) are not in the paper yet.
- [ ] Discussion and limitations: synthetic training, cubic only, 2D maps,
  uniform-angle noise, grains of one orientation, the real reference has its
  own error. Conclusion.
- [ ] Reproducibility: release the code, the MTEX script, seeds, test sets
  and checkpoints.

### 5.8 Housekeeping

- [ ] `COMMANDS.md`, `README.md` and `SHARED/configs/README.md` still use the old
  paths (`mtex_out_<set>`, `datasets/clean_euler`, `datasets/noise_<set>`).
- [ ] Test the figure scripts after the path changes: `plot_error_maps`,
  `make_results_figures`, `plot_noise_examples`, `make_mtex_package`,
  `check_noise`, `summarize_test_grid`.
- [ ] Home folder over its disk quota ("Disk quota exceeded" on 2026-10-14);
  free some space.
- [ ] The rclone OneDrive remote is empty; re-create it with `rclone config`.
- [ ] Optional: activate MATLAB on the cluster (CMU licence) and add the
  Statistics and Machine Learning Toolbox, to run MTEX here
  (`MTEX_PIPELINE/src/run_mtex_grid.sh`).

---

## 6. Hypotheses (written 2026-10-11, before the results)

| | Claim | Can be tested after |
|---|---|---|
| H1 | Lower ALL and BOUNDARY error than the best (tuned) MTEX filter at every scatter level | MTEX run (5.1); tuning decision (5.2) |
| H2 | Lower >10° fraction than the best `clean-*` MTEX filter | MTEX run (5.1) |
| H3 | Better than the local mean alone | A1 (5.2) |
| H4 | On real scans, closer to the reference than the raw map and MTEX | Real data (5.6) |
| H5 | The ranking survives the robustness tests | 5.5 |

If one fails, report it. "Learning helps mainly with misindexing and
boundaries" is still a publishable, honest result.

---

## 7. Decisions that changed the original plan

- 2026-10-12: test grid of 28 sets (7 scatter × 4 misindexing) instead of 6.
- 2026-10-12: training on the GPU only.
- Training range widened from scatter 0–9° to 0–15°, to cover the grid.
- Fixed 60 epochs per run instead of "stop when validation stops improving".
- 2026-10-13: every error number is a mean, not a median.
- 2026-10-14: MTEX on maps 1–10 of each set (not 100–500), 6 `clean-*`
  filters only.
- 2026-10-14: results in two folders, model and MTEX.

---

## 8. Order of the remaining work

1. Local-mean baseline (A1) and boundary F1 (5.2, 5.3).
2. Decide on MTEX tuning and on the checkpoint question (5.2).
3. Write the empty paper sections (5.7).
4. Real data (5.6).
5. Robustness tests (5.5), then ablations A2–A5 (5.4).

---

## 9. References

- Atindama E, Lef P, Doğan G, Athavale P (2023) Restoration of noisy orientation maps from EBSD imaging. *Integr Mater Manuf Innov* 12:251–266. doi:10.1007/s40192-023-00304-8
- Hielscher R, Silbermann CB, Schmidl E, Ihlemann J (2019) Denoising of crystal orientation maps. *J Appl Crystallogr* 52:984–996. doi:10.1107/S1600576719009075
- Shen Y-F, Pokharel R, Nizolek TJ, Kumar A, Lookman T (2019) CNN-based real-time orientation indexing of measured EBSD patterns. *Acta Mater* 170:118–131. doi:10.1016/j.actamat.2019.03.026
- Ding Z, Pascal E, De Graef M (2020) Indexing of EBSD patterns using a convolutional neural network. *Acta Mater* 199:370–382. doi:10.1016/j.actamat.2020.08.046
- Andrews CE, Strantza M, Calta NP, Matthews MJ, Taheri ML (2023) A denoising autoencoder for improved Kikuchi pattern quality and indexing in EBSD. *Ultramicroscopy* 253:113810. doi:10.1016/j.ultramic.2023.113810
- Mani Krishna KV, Madhavan R, Pantawane MV, Banerjee R, Dahotre NB (2023) ML-based de-noising of electron back scatter patterns. *Ultramicroscopy* 247:113703. doi:10.1016/j.ultramic.2023.113703
- Prajapat M, Alankar A (2026) Generative diffusion modeling protocols for improving Kikuchi pattern indexing in EBSD. *Modelling Simul Mater Sci Eng* 34:065016. doi:10.1088/1361-651X/ae956a
- Kaufmann K, Lane H, Liu X, Vecchio KS (2021) Efficient few-shot machine learning for classification of EBSD patterns. *Sci Rep* 11:8172. doi:10.1038/s41598-021-87557-5
- Martinez Ostormujof T et al. (2022) Deep learning for automated phase segmentation in EBSD maps. *Mater Charact* 184:111638. doi:10.1016/j.matchar.2021.111638
- Zhou Y, Barnes C, Lu J, Yang J, Li H (2019) On the continuity of rotation representations in neural networks. *CVPR*.
- Brewer L, Michael J (2010) Risks of "cleaning" EBSD data. *Microsc Today* 18(2):10–15. (cited by Atindama; useful for the "do not invent data" discussion)
- Wright SI, Nowell MM, de Kloe R, Chan L (2014) Orientation precision of EBSD measurements near grain boundaries. *Microsc Microanal* 20:852–863. (why boundary pixels are hardest)
- Ånes HW et al. Electron backscatter patterns from nickel acquired with varying camera gain. Zenodo, doi:10.5281/zenodo.7497682 (CC BY 4.0)
