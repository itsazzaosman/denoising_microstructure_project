# 09 — Evaluation

**Role:** does the denoiser from
[`08_synthetic_training_data`](08_synthetic_training_data.md) actually work
— on data it never trained on, and by the metric that actually matters
(indexing), not just image fidelity?

**Input:** `08_synthetic_training_data/denoiser_30.keras` (the model) +
[`05_noising_training_data/training_pairs.h5`](05_noising_training_data.md)
(the held-out test set — 4,125 real-orientation-derived patterns this model
never saw during training)
**Output:** two stages of results, both compared in the
[Results](#results) section below

Split into two scripts across two environments, deliberately: TensorFlow and
`pyebsdindex`/`kikuchipy`/`orix` don't need to coexist, so Stage 1 (needs
TensorFlow, runs in `~/.dl`) saves its full pattern arrays to disk, and
Stage 2 (needs pyebsdindex, runs in `~/.ebsd` or the main `venv` — both
turned out to have everything needed) reads those back without ever
importing TensorFlow.

Both scripts accept `--tag`/`--holdout-file` so the exact same analysis can
be re-run at a different noise level without overwriting prior results —
that's how the calibrated vs. harsh comparison below exists side by side.

## Stage 1 — pattern quality (`01_pattern_quality_holdout.py`)

Loads the model, loads the holdout set, verifies preprocessing matches
(`float32`, `[0,1]`, correct shape — asserted, not assumed), predicts, and
computes two fidelity metrics against the clean target:

- **PSNR** (peak signal-to-noise ratio, dB) — pixel-wise fidelity; every
  +3 dB roughly halves the mean squared error.
- **NCC** (normalized cross-correlation) — structural agreement independent
  of brightness/contrast; 1.0 is identical.

This answers "did the model generalize to orientations it never trained
on," but it's necessarily a proxy question — good PSNR/NCC against a
*simulated* clean pattern doesn't by itself prove the denoised pattern is
more useful for the real downstream task. That's what Stage 2 tests
directly.

## Stage 2 — indexing accuracy (`02_indexing_accuracy.py`)

Hough-indexes three pattern sets — noisy (**baseline**), denoised
(**result**), and clean (**upper bound**) — against the same detector
geometry and nickel phase, then scores each against the known ground truth
by symmetry-aware disorientation.

**The Hough indexing pipeline**, per pattern, independently (no pattern ever
uses information from its neighbors):

1. **Radon transform** — converts the image so straight Kikuchi band edges
   become single bright peaks in (ρ,θ) space, which is far more
   noise-tolerant than trying to find lines directly in the image.
2. **Convolution** — a peak-sharpening filter enhances genuine band peaks
   over background.
3. **Peak ID** — local maxima become candidate bands. This is where noise
   costs real time: at the calibrated noise level, this step took ~1.8s for
   noisy patterns vs. ~0.85s for denoised/clean, because noise generates far
   more spurious candidate peaks to sort through.
4. **Band labeling** — candidate peaks convert back into real-space band
   positions and get ranked by strength.
5. **Triplet voting** — the strongest ~8-10 bands are checked in every
   possible triplet against the phase's precomputed table of theoretical
   interplanar angles (derived from the space group + lattice parameter in
   `00_crystal_structure`). The orientation with the most/strongest votes
   wins, along with a `fit` (band-position residual — lower is better) and
   `cm` (confidence margin over the runner-up — the "CI").

**Symmetry-aware disorientation:** nickel's cubic symmetry means 48
different-looking orientation descriptions can represent the same physical
orientation. `orix.quaternion.Orientation.angle_with()`, with the m-3m
point group attached to both the ground truth and the indexed result, finds
the smallest angle among all 48 equivalent comparisons — the physically
meaningful "disorientation," not a raw quaternion angle that would overstate
error.

**Indexing failures, once the noise got harsh enough to cause them:** at the
calibrated noise level every point indexed successfully. At the harsher
level tested below, Hough genuinely gives up on a small fraction of points
(phase `"not_indexed"`). Those are excluded from median/mean disorientation
(can't compute an angle with no solution) but explicitly counted against
`fraction < 1°`/`fraction < 2°`, and reported as their own metric,
`fraction_indexed` — an outright indexing failure is a real outcome, not
something to silently drop from the average.

## Dictionary indexing: considered, not run

Hough was prioritized as the first (and so far only) indexing method tested,
based on a prediction worth recording: Dictionary Indexing (DI) —
cross-correlating the *whole* pattern against a dense simulated
orientation dictionary — is generally *more* noise-robust than Hough, not
less, since whole-image correlation is an even stronger noise-averaging
operation than the Radon transform already is. Given Hough already sits
near-ceiling at the calibrated noise level (see below), DI would likely show
the same or a smaller gap there, for the same reason. Where DI could show
something genuinely *different* is the opposite direction: it's sensitive to
whole-pattern contrast/background style, not just band geometry, so any
subtle full-image artifact the denoiser introduces (the mild blur visible in
Stage 1's example figures) could hurt DI's cross-correlation even where it
doesn't move band positions enough to bother Hough. Not run here — the
harsher-noise Hough test below was judged the higher-priority next step,
since it tests the paper's actual claim more directly and far more cheaply.

---

## Results

Two noise levels, same model (`denoiser_30.keras`, trained only at SNR
1.27), same 4,125-pattern holdout set, same ground truth. **The harsh level
is genuinely out-of-distribution for this model** — it never saw noise this
severe during training — which makes any improvement found there a stronger
result than if the model had been trained specifically for it.

### Stage 1 — pattern quality

| | Calibrated (SNR 1.27) | Harsh (SNR ≈ 0.50) |
|---|---|---|
| PSNR, noisy input | 16.01 dB | 11.55 dB |
| PSNR, denoised | 19.72 dB | 15.72 dB |
| **PSNR gain** | **+3.71 dB** | **+4.17 dB** |
| NCC, noisy input | 0.785 | 0.443 |
| NCC, denoised | 0.833 | 0.557 |
| **NCC gain** | **+0.048** | **+0.114** |

The denoiser's raw fidelity improvement is *larger* in absolute terms at the
harsh level (more than double the NCC gain) — it has more noise to remove,
and removes proportionally more of it, despite never having trained on
noise this severe.

### Stage 2 — indexing accuracy

| | Calibrated (SNR 1.27) | | | Harsh (SNR ≈ 0.50) | | |
|---|---|---|---|---|---|---|
| | baseline | denoised | clean | baseline | denoised | clean |
| Fraction indexed at all | 100%* | 100%* | 100%* | 99.81% | 99.98% | 100% |
| Median disorientation | 0.517° | 0.494° | 0.488° | 0.580° | 0.558° | 0.488° |
| **Mean disorientation** | 0.538° | 0.502° | 0.521° | **1.374°** | **0.659°** | 0.521° |
| Fraction < 1° | 98.88% | 99.98% | 99.10% | 94.86% | 97.92% | 99.10% |
| Fraction < 2° | 100% | 100% | 100% | 98.04% | 99.78% | 100% |
| Hough fit (mean, lower better) | 0.535 | 0.349 | 0.460 | 0.668 | 0.630 | 0.460 |
| Hough CI (mean, higher better) | 0.718 | 0.734 | 0.733 | 0.547 | 0.631 | 0.733 |

*No indexing failures were observed at the calibrated level (the field
wasn't tracked in that run, since the failure mode hadn't shown up yet).

### What this actually shows

**At the calibrated noise level — the real, measured noise of the actual
detector — indexing is already close to its ceiling before denoising even
enters the picture.** Baseline is already 98.9% accurate under 1°. There is
real, measurable improvement from denoising (median disorientation error
down ~4.5% relative, fraction-under-1° up to essentially 100%), and the
result does sit between baseline and upper bound on every metric — but the
gap is small because there wasn't much room left to close. This is why the
IPF-orientation maps for all three pattern sets looked visually identical
earlier: color differences below ~0.5° simply aren't visible in that kind
of rendering, regardless of whether the underlying numbers moved.

**Push the noise to roughly half the SNR (~0.50, still Gaussian, following
Andrews et al.'s own harsher preset), and a real, substantial gap opens
up** — and it shows up most clearly in the *mean*, not the median. Baseline's
mean disorientation (1.374°) is more than double its median (0.580°) — a
classic signature of a heavy tail: most points are still indexed
reasonably well, but a meaningful subset are badly wrong or fail outright.
Denoising cuts that mean roughly in half (0.659°), pulling it much closer to
the clean upper bound (0.521°), while barely moving the median at all. In
other words: **denoising's main benefit at harsh noise isn't making already-good
answers slightly better — it's rescuing the worst-case points**, exactly the
tail that traditional per-pixel fidelity metrics like PSNR/NCC don't
directly measure, but that determines whether a real EBSD map has visible
holes in it. `fraction_indexed` tells the same story more starkly: denoising
closes 89% of the gap between baseline's outright-failure rate (0.19%) and
clean's (0%).

**One anomaly worth reporting rather than smoothing over:** in the
calibrated run, "clean" scored *worse* than "denoised" on fraction-under-1°
(99.1% vs. 99.98%) and mean Hough fit (0.460 vs. 0.349) — an upper bound
losing to the thing it's supposed to bound. Plotting per-point disorientation
as a heatmap (rather than IPF color, which can't show sub-degree
differences) revealed why: clean's higher-error points cluster spatially in
specific grains rather than scattering randomly the way noise-driven errors
do — the signature of **pseudosymmetry**, a known EBSD failure mode where
certain crystal orientations (typically near a high-symmetry zone axis)
produce band geometry that's genuinely ambiguous between two distinct
orientations, independent of noise. The most likely explanation: the
denoiser's smoothing (an MSE-trained network's well-known tendency to
average away fine detail) happens to erase exactly the subtle secondary-band
detail that creates that ambiguity in the sharp "clean" pattern — an
incidental side effect of denoising, not a real violation of the upper
bound, and not something that shows up at the harsh noise level where
noise-driven errors dominate everything else.

### Bottom line

**Does denoising improve indexing?** Yes, confirmed at both noise levels,
and the model does this on a noise level it was never trained for. **By how
much** depends entirely on how hard the indexing problem already is: at the
real, measured noise level of this specific detector, the improvement is
real but small because indexing was already easy; the paper's claimed
effect becomes clearly visible once the noise is pushed harder than what
this detector/sample combination actually produces. The practical
implication: whether denoising is "worth it" for indexing accuracy on a
given real dataset should be checked against *that dataset's* actual noise
level, not assumed from a generically noisy-looking pattern.
