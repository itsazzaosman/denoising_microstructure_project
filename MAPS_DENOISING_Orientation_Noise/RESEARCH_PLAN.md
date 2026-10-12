# Research plan: learned denoising of EBSD orientation maps

Written 2026-10-11, after reading the 8 papers in the `papers_ebsd_ml` Drive
folder and checking the current code. Each step says **why** (which weakness
in the literature or in our pipeline it fixes), **what to do**, and **done
when**. Steps are tagged **MUST**, **SHOULD** or **NICE**.

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

## 3. Problems in our current pipeline (fix before final runs)

These were found by reading the code and the paper limitations side by side.

1. **Data are not on disk.** `datasets/` and all `.pth` files are missing
   locally (they are safe in `gs://cmu-gpucloud-aiosman/DATASET_CHECKPOINTS/`).
   The nightly backup has logged an error since 10 Oct because the folder is gone.
2. **Unfair baselines.** MTEX ran with default parameters, the same at every
   noise level, while our model was retrained for each level. Atindama's whole
   paper is about this tuning problem, so a reviewer will spot it at once.
3. **MTEX ran on 10 maps only** (maps 1–10). That is the same weakness as
   Atindama. Too few for statistics.
4. **MTEX grain threshold is too small for high scatter.** `calcGrains(...,
   10*degree)`: at 9° scatter two neighbouring pixels of one grain can differ
   by up to ~18°, so grains get split. Also, the "variant B" `for` line is
   commented out in the current `src/run_mtex_baseline.m`, so only the last
   filter runs in that block.
5. **Symmetry bug risk in the local mean (tested today).** A cubic
   orientation can be written 24 equivalent ways (*symmetry variants*). Real
   vendor files often store neighbouring pixels of one grain in different
   variants. Our local mean aligns neighbours by sign only, so it drops them.
   Test on a toy map at 5° scatter: **0.31° error → 1.34°** when each pixel
   is stored as a random variant (worst case; same physical data). Our
   synthetic data never contain this, so the problem is invisible in our
   scores but will show up on real data.
6. **The model's main feature is untested right now.** The gate/replace heads
   exist for misindexing, but the current data are scatter-only (gate
   precision/recall = 0). The misindexing test set was deleted.
7. **Noise levels are far above real EBSD.** Our 5/7/9° sets have 2.5–4.5°
   mean error. Atindama used 0.56–1.45°. Ding shows real methods differ by
   ~1–1.5°. Our earlier 1° set (0.5° mean) was realistic. High levels can stay
   as a stress test, but they cannot be the main result.
8. **Most of the scatter removal comes from the classical local mean**
   (README, and the 11% ratio found earlier). Without a "local mean only"
   baseline, a reviewer can say the gain is not from learning.
9. **One seed, not converged.** One training run per model; the 9° model was
   still improving at epoch 40. The `--mean-tol` used is unknown until the
   `.pth` files are restored (it is saved inside them).

---

## 4. The plan, step by step

### Step 0 — Restore and freeze (MUST, do first)

- [x] Restore data and checkpoints:
  `gcloud storage rsync -r gs://cmu-gpucloud-aiosman/DATASET_CHECKPOINTS/datasets datasets`
  (same for `checkpoints`). This also stops the nightly sync errors.
- [x] Read the saved `args` in each `best.pth` and write down `mean_tol`,
  `mean_radius`, `max_refine_deg`, `epochs`, `seed`.
- [x] Fix the out-of-date paths listed in the README (Makefile defaults,
  `inference.py --out`).
- [x] Make a `configs/` file per experiment (noise settings, seed, model
  flags) and record the git commit hash in every results file.

**Done when:** `make score-all` runs end to end on restored data.

### Step 1 — Write the claims down before running anything (MUST)

Write these into the paper draft now, with empty result tables. Fixing the
claims in advance stops you choosing, after the fact, the comparison that
happens to look good.

- **H1 (scatter):** the model has lower ALL and BOUNDARY error than the best
  *tuned* MTEX filter at every scatter level.
- **H2 (misindexing):** the model has a lower >10° fraction than the best
  `clean-*` MTEX method.
- **H3 (learning matters):** the model beats the local-mean filter alone.
- **H4 (real data):** on real noisy scans, the denoised map is closer to the
  reference than the raw map and the tuned MTEX output.
- **H5 (robustness):** the ranking survives the robustness tests in Step 7.

If one fails, report it. "Learning helps mainly with misindexing and
boundaries" is still a publishable, honest result. From the scatter-only
numbers, expect this.

### Step 2 — Build the test sets (MUST) — done 2026-10-12

**Why:** problems 6 and 7. We need realistic levels and both noise types.

Full grid = 7 scatter levels × 4 misindexing levels = **28 test sets**, maps
1–500 each, boundary bias 4 (decided 2026-10-12, replacing the 6-set grid):

- scatter max 1, 3, 5, 7, 9, 12, 15° (1° = good scan; 15° = grains only just
  visible, the worst level that can still be denoised)
- misindexing 0, 3, 6, 9% (0% keeps the scatter-only results, including the
  existing scatter5/7/9 sets)

- [x] Configs written by `src/make_test_grid.py` (`configs/noise/`,
  list in `configs/test_grid.txt`); sets generated in `datasets/noise_<name>/`.
- [x] **One shared seed (0)** for every set: a paired, factorial design where
  two cells differ only in the amount of noise. State this in the Methods.
- [x] Input noise of every set measured: `figures/test_grid/input_noise.csv`.
- [x] Training and validation range widened to scatter 0–15° to cover the grid.
- [ ] Make a **tuning set** for the baselines: maps 501–520 at the grid
  settings. These are never used for testing.

### Step 3 — Fair baselines (MUST)

**Why:** problems 2, 3, 4 and 8, and the core weakness of Atindama.

- [ ] Fix `run_mtex_baseline.m`: restore the variant-B `for` loop; make the
  input/output folders and `nMaps` arguments rather than hand-edited lines.
- [ ] Set the grain threshold per condition to above the largest
  within-grain difference (about 2× the max scatter; e.g. 5° for 1°
  scatter, 20° for 9°), and report the values. Apply the same rule to our
  `--mean-tol` (its help text says ~1.6× max scatter).
- [ ] **Tune each filter on the tuning maps (501–520), separately per
  condition.** Grid-search the main strength setting of each filter (for
  example `numNeighbours` for mean/median/Kuwahara, `alpha` for spline and
  half-quadratic, the two weights of infimal convolution; check exact names
  in the MTEX 7.1 docs). Pick the value with the lowest mean error, then
  freeze it.
- [ ] Run the tuned filters on test maps **1–100** for all 6 conditions
  (all 500 if time allows). Decide the number now and do not change it later.
- [ ] Add the **local-mean-only** baseline: our edge-preserving local mean
  with no network (the "epoch 0" model). Run it on all 500 maps.
- [ ] Note in the paper that `clean-*` removes grains under 3 pixels and does
  **not** use the true bad-pixel mask (confirmed in the script). This
  answers the README's open question: the 1.48% baseline is honest.

**Done when:** a results table where every method had the same chance to be
tuned.

### Step 4 — Train the final model properly (MUST)

**Why:** problems 5, 6 and 9; Ding (train on the noise you will meet);
Kaufmann and Martinez (repeat trainings).

- [x] **Fix the symmetry handling first.** (Done 2026-10-11 inside the learned kernel, `aligned_neighbours`.) In `local_mean_quat`, align each
  neighbour to the centre pixel over all 24 cubic symmetry variants (the
  existing `align_to()` in `orientation_torch.py` does this), *then* apply the
  tolerance test. Also add a training augmentation that stores each pixel as
  a random symmetry variant. Re-run the toy test above. It must give the
  same error with and without random variants.
- [x] (configs written: `configs/train/blind_learned_seed1-3.yaml`; training not started) **Train one "blind" model** that never knows the noise level: per
  training map, draw scatter max uniformly from 0–9° and misindexing from
  0–10%. (`TrainMaps` currently uses fixed values. This needs a small change
  where `corrupt(...)` is called.) This is the main model. It is more useful
  in practice, and fairer against filters tuned per level.
- [ ] Keep the per-level models as an ablation ("does knowing the noise level
  help?").
- [ ] Train until validation stops improving for ~10 epochs (max ~100), with
  the same rule for every run.
- [ ] **3 seeds** for the blind model. Report mean ± std. A difference smaller
  than the seed-to-seed spread is not a result.

**Done when:** 3 converged blind-model checkpoints that pass the symmetry test.

### Step 5 — Ablations: what makes it work (MUST for A1, SHOULD for the rest)

An *ablation* means removing one part and measuring the drop.

- **A1** local mean only, no network (from Step 3). Answers H3. **MUST.**
- **A2** network without the local mean (`--mean-radius 0`), trained to
  convergence. The old v1 run had only 3 epochs.
- **A3** no local-misfit input channel (`--no-misfit`).
- **A4** per-level models vs the blind model.
- **A5** sign-only vs symmetry-aware local mean (the fix in Step 4).

One seed is acceptable for A2–A5 if you say so in the paper.

### Step 6 — Metrics and statistics (MUST)

**Why:** Atindama used one number, 10 maps and a t-test, with no correction
for many comparisons.

Report per condition:
- Current four: ALL, CORRUPTED, BOUNDARY (medians), and >10° (%).
- **Mean** disorientation too, so numbers can be set beside Atindama and
  Hielscher (they use the mean).
- **Fraction of noise removed** = 1 − output error / input error. This
  makes levels comparable and answers the "same percentage" question
  directly.
- **Boundary F1** (new): detect boundaries (neighbour disorientation > 5°)
  in the denoised map and compare with the true boundaries. F1 combines
  "found the true ones" and "did not invent fake ones". Check that 10° gives
  the same ranking. This replaces Atindama's by-eye boundary figures with a
  number.

Statistics:
- The unit is the **map**, not the pixel. Pixels in one map are not
  independent, so pixel-level tests overstate certainty.
- For each map, compute model error − best-baseline error (a *paired*
  difference: same map, same noise).
- Test with the **Wilcoxon signed-rank test** (safe for skewed errors), and
  give a **bootstrap 95% confidence interval** of the median difference
  (resample maps 10,000 times).
- Correct for the many comparisons with **Holm's method**.
- Claim a win only if the interval excludes zero **and** the gap is bigger
  than the seed spread.

**Done when:** `score_dataset.py` (or a small `stats.py`) writes one CSV per
condition plus a summary table with intervals and corrected p-values.

### Step 7 — Robustness tests (SHOULD)

**Why:** training and test noise come from the same generator, so good scores
may partly mean "learned this generator's habits". Each test below changes
one thing the model never saw.

- [ ] **Symmetry relabelling:** store each test pixel (a) in the fundamental
  zone, as vendor software does, and (b) as a random variant. Scores must not
  change. MTEX will pass this; our current model will not (Step 4 fixes it).
- [ ] **Different noise shape:** same average error, but the tilt angle
  follows a bell-shaped distribution (e.g. von Mises–Fisher, or the de la
  Vallée Poussin noise Atindama used, which MTEX can sample) instead of
  uniform 0–max.
- [ ] **Grains with internal orientation gradients:** DREAM.3D grains have
  one orientation each, but real deformed grains bend slowly. Apply a smooth
  sample rotation that grows across the map (e.g. 0 → 5° over 128 px) to
  the clean maps, then add noise. Check whether the model flattens the
  gradient. This is Atindama's stated weakness and a risk for every
  method trained on DREAM.3D.
- [ ] **Different microstructure (NICE):** a few DREAM.3D maps with smaller
  grains (more boundaries), like Martinez's unseen heat treatment.
- [ ] **Cross-level matrix:** evaluate every model at every noise level.

### Step 8 — Real-data validation (MUST for a strong paper)

**Why:** this is the biggest gap in all 8 papers. Map-level papers judged
real data by eye. Pattern-level papers used CI/IQ, which measure
self-consistency, not accuracy.

**Data:** the public *ni_gain* dataset (Ånes et al.), 10 scans of the **same
area** of recrystallised **nickel** (cubic, like our synthetic data), at
camera gain 0–24 dB (more gain = noisier patterns). Map 149 × 200, patterns
60 × 60, CC BY 4.0, Zenodo DOI 10.5281/zenodo.7497682. It loads through
`kikuchipy.data.ni_gain(number=k, allow_download=True)`.

- [ ] Install in the `ebsd` env: `pip install kikuchipy pyebsdindex orix`
  (none are installed yet).
- [ ] Index all 10 scans with Hough indexing (PyEBSDIndex) → 10 realistic
  noisy maps, with real scatter *and* real misindexing.
- [ ] Build the **reference**: index the lowest-gain scan with dictionary
  indexing + refinement (kikuchipy). This is the most accurate map we can make.
- [ ] Measure the **reference floor**: the disagreement between two good
  indexings (e.g. Hough vs dictionary on the 0 dB scan). Differences
  smaller than this cannot be claimed. Ding found ~1° between methods.
- [ ] Check **drift** between scans (misorientation map between two low-gain
  scans), as Mani Krishna and Shen did. Register the maps if they shifted.
- [ ] Denoise each noisy map with the blind model (synthetic training only,
  no retuning on this data) and the tuned MTEX filters (parameters frozen
  from Step 3). The U-Net is fully convolutional; pad the map to a multiple
  of 8.
- [ ] Plot error vs reference, and >10° fraction, against gain for raw /
  MTEX / ours.

**Done when:** one figure and one table showing how far each method moves
real noisy maps toward the reference.

### Step 9 — Write it up (start today)

These sections do not depend on final numbers, so write them now:

- **Introduction + Related work:** use the table in Section 2. Make the
  "patterns vs maps" split explicit: our method needs only the map.
- **Methods:** data (DREAM.3D, 40,000 maps, test 1–500 vs train 501+, no
  overlap, unlike Prajapat), noise model, architecture (6D representation
  per Zhou 2019, symmetry-aware loss, gated heads), baselines and tuning
  protocol, metrics, statistics. Write the Step 1 hypotheses here.
- **Results:** create the empty tables and figure captions now. One table per
  hypothesis.
- **Limitations (be upfront):** synthetic training, cubic only, 2D maps,
  uniform-angle noise as the main model, the real reference has its own
  error, piecewise-constant training grains.
- **Reproducibility:** release code, the frozen MTEX script, seeds, test sets
  and checkpoints. Atindama and Prajapat did not release data, so a public
  benchmark is a contribution in itself.

---

## 5. Order of work

1. Step 0 (restore), then Step 1 (claims), and start Step 9 writing. **Today.**
2. Step 2 (test sets) and the Step 4 symmetry fix. Both are small code jobs.
3. In parallel: Step 3 MTEX tuning + runs (MATLAB machine) and Step 4
   blind-model training with 3 seeds (GPU).
4. Step 5 A1 + Step 6 statistics → first full results table.
5. Step 8 real data.
6. Step 7 robustness, then the remaining ablations.

---

## 6. References

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
