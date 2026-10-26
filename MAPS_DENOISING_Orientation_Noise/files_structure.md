# MODEL_PIPELINE: methods notes for the paper

Working notes on the learned denoiser in `MODEL_PIPELINE/`: the data, the noise
model, the network, the loss, training, validation and evaluation, with the
file that implements each step and the decisions behind them. They describe the
**current setup** (training configs `blind_both_seed{1,2,3}`, 2026-10-26). The
earlier model (`blind_learned`) and its results are summarised in
[Section 13](#13-earlier-model-blind_learned-and-its-results).

Equations use `$…$` and `$$…$$`; open the Markdown preview in VS Code
(Ctrl+Shift+V) to see them rendered.

**Contents**

1. [Data](#1-data)
2. [Orientation representation](#2-orientation-representation)
3. [Noise model](#3-noise-model-added-on-the-fly)
4. [Data augmentation (training only)](#4-data-augmentation-training-only)
5. [Training and validation datasets](#5-training-and-validation-datasets)
6. [Network](#6-network)
7. [Loss](#7-loss)
8. [Training procedure](#8-training-procedure)
9. [Evaluation metrics and testing](#9-evaluation-metrics-and-testing)
10. [Code: files and how they connect](#10-code-files-and-how-they-connect)
11. [The training config file](#11-the-training-config-file)
12. [Decision log](#12-decision-log)
13. [Earlier model (blind_learned) and its results](#13-earlier-model-blind_learned-and-its-results)

---

## 1. Data

### 1.1 Synthetic orientation maps

- **Source:** 40,000 synthetic polycrystalline microstructures generated with
  DREAM.3D, about 52 grains per map.
- **Size:** each map is $128 \times 128 = 16{,}384$ pixels.
- **Content:** each pixel holds one crystal orientation, as three Bunge
  (ZXZ) Euler angles $(\varphi_1, \Phi, \varphi_2)$ in radians.
- **Crystal symmetry:** cubic, point group $m\bar{3}m$ (nickel, lattice
  parameter 3.524 Å).
- **Files:** one `.txt` per map, `datasets/clean_euler_zip_unzip/clean_euler/map_XXXXX_clean_euler.txt`.
- **Unusable maps:** 2 of the 40,000 (`08231` is empty, `34360` is truncated),
  so **39,998 maps are usable**. They are left out of every split.

### 1.2 Train / validation / test split

Script: `SHARED/src/make_splits.py`. The split lists are tracked in git:
`SHARED/configs/splits/{train,val,test}.txt`.

| Split | Map IDs | Maps | Used for |
|---|---|---|---|
| Test | 1–500 | 500 | Final evaluation and the MTEX comparison |
| Validation | 501–1000 | 500 | Model selection during training |
| Training | 1001–40000 | 38,998 | Training (the 2 unusable maps are in this range) |

Contiguous ID ranges are safe because every map was generated with the same
DREAM.3D settings (about 52 grains per map in every ID range).

### 1.3 Clean caches (`.txt` → `.npy`)

Script: `MODEL_PIPELINE/src/prepare_cache.py`.

1. Reads the map IDs from `train.txt`, `val.txt` and `test.txt`.
2. Loads the matching clean Euler-angle `.txt` files.
3. Checks every map (shape $16{,}384 \times 3$, no NaN or infinity) and stores
   the values as float32.
4. Packs each split into one memory-mappable `.npy` file, so training reads one
   file instead of tens of thousands of text files.
5. Saves the map IDs separately, in the same row order, to keep the link to the
   original maps.
6. Adds **no noise** and does **not** convert to quaternions.

| Split | File | Shape (maps × pixels × angles) | IDs file |
|---|---|---|---|
| Training | `datasets/cache/split_train_clean.npy` | $(38998, 16384, 3)$ | `split_train_clean_ids.npy` |
| Validation | `datasets/cache/split_val_clean.npy` | $(500, 16384, 3)$ | `split_val_clean_ids.npy` |
| Testing | `datasets/cache/split_test_clean.npy` | $(500, 16384, 3)$ | `split_test_clean_ids.npy` |

Verified on 2026-10-25: the shapes above, the ID files identical to the split
lists, and 5 maps per split identical to their `.txt` files.

Run once, or again whenever the clean data or the split change:
`python MAPS_DENOISING_Orientation_Noise/MODEL_PIPELINE/src/prepare_cache.py`.
It refuses to replace an existing cache unless `--overwrite` is given.

---

## 2. Orientation representation

### 2.1 Euler angles to quaternions

Each orientation is converted to a unit quaternion $q = (q_w, q_x, q_y, q_z)$
when a map is loaded (in memory; the cache keeps the Euler angles). With

$$
\sigma = \tfrac{1}{2}(\varphi_1 + \varphi_2), \qquad
\delta = \tfrac{1}{2}(\varphi_1 - \varphi_2),
$$

$$
q = \Big(\cos\tfrac{\Phi}{2}\cos\sigma,\; -\sin\tfrac{\Phi}{2}\cos\delta,\;
-\sin\tfrac{\Phi}{2}\sin\delta,\; -\cos\tfrac{\Phi}{2}\sin\sigma\Big),
$$

and the sign is chosen so that $q_w \ge 0$ ($q$ and $-q$ are the same
rotation). A map becomes a $128 \times 128 \times 4$ array.

### 2.2 Crystal symmetry and the disorientation angle

A cubic crystal looks the same after any of the 24 rotations $s_k$ of its
symmetry group, so $q$ and $s_k \otimes q$ describe the same orientation.
Symmetry acts by **left** multiplication in this convention.

The error between two orientations is the **disorientation angle**: the
smallest rotation angle between them over all symmetry-equivalent versions. With
the misorientation $m = q_2 \otimes q_1^{-1}$ and its absolute components sorted
as $a \ge b \ge c \ge d$, the cubic case has the closed form

$$
\cos\frac{\theta}{2} = \max\Big(a,\; \frac{a+b}{\sqrt{2}},\; \frac{a+b+c+d}{2}\Big),
\qquad \theta = 2\arccos(\cdot).
$$

$\theta$ ranges from $0^\circ$ to $62.8^\circ$ for cubic symmetry. All errors in
these notes are disorientation angles in degrees, never differences of Euler
angles.

---

## 3. Noise model (added on the fly)

Script: `MODEL_PIPELINE/src/dataset.py` (`corrupt()` and `_CleanMaps.__getitem__`).
Two kinds of EBSD noise are applied together, because real maps contain both.

### 3.1 Random noise level per map

Every time a map is accessed, the loader draws its own noise level:

$$
\boxed{\theta_{\max} \sim \mathcal{U}(1^\circ, 15^\circ)}
\qquad
\boxed{p_{\mathrm{mis}} \sim \mathcal{U}(0.01, 0.20)}
$$

- $\theta_{\max}$ is the maximum scatter angle, $p_{\mathrm{mis}}$ the fraction
  of misindexed pixels.
- Both are drawn continuously and independently, not from a fixed list of
  levels.
- The minimums are above zero, so **every map carries both noise types**:
  there is never scatter without misindexing, or misindexing without scatter.
- The upper bound of 20% misindexing makes the task harder than the earlier
  model's 10% (Section 13).

### 3.2 Orientation scatter

Every pixel $i$ is tilted by its own small random rotation. The axis is a
normalised Gaussian vector and the angle is uniform up to $\theta_{\max}$:

$$
\mathbf{n}_i = \frac{\mathbf{v}_i}{\lVert\mathbf{v}_i\rVert},\quad
\mathbf{v}_i \sim \mathcal{N}(\mathbf{0}, I_3),
\qquad
\alpha_i \sim \mathcal{U}(0, \theta_{\max}),
$$

$$
r_i = \Big(\cos\frac{\alpha_i}{2},\; \mathbf{n}_i \sin\frac{\alpha_i}{2}\Big),
\qquad
\boxed{q_i^{\mathrm{scatter}} = r_i \otimes q_i^{\mathrm{clean}}}
$$

where $\otimes$ is quaternion multiplication. The pixel keeps its position; only
its orientation changes. The mean tilt is $\theta_{\max}/2$: for example, with
$\theta_{\max} = 10^\circ$ each pixel is tilted between $0^\circ$ and $10^\circ$,
$5^\circ$ on average.

### 3.3 Misindexing

After scatter, a fraction $p_{\mathrm{mis}}$ of the pixels is replaced by a
completely random orientation:

$$
N_{\mathrm{mis}} = \operatorname{round}(p_{\mathrm{mis}} \, H W), \qquad H = W = 128,
$$

$$
\boxed{
q_i^{\mathrm{noisy}} =
\begin{cases}
q_i^{\mathrm{random}}, & i \in \mathcal{M},\\[2pt]
q_i^{\mathrm{scatter}}, & i \notin \mathcal{M},
\end{cases}}
$$

where $\mathcal{M}$ is the set of misindexed pixels and $q^{\mathrm{random}}$ is
uniform on $SO(3)$ (Shoemake's method: with $u_1, u_2, u_3 \sim \mathcal{U}(0,1)$,
$q = (\sqrt{u_1}\cos 2\pi u_3,\, \sqrt{1-u_1}\sin 2\pi u_2,\, \sqrt{1-u_1}\cos 2\pi u_2,\, \sqrt{u_1}\sin 2\pi u_3)$).

**Boundary bias.** Misindexing happens most at grain boundaries, where the
diffraction pattern mixes two grains. Boundary pixels get four times the
selection weight of interior pixels, and $\mathcal{M}$ is drawn without
replacement with probability proportional to the weights:

$$
w_i = \begin{cases} 4, & i \text{ is a boundary pixel},\\ 1, & i \text{ is an interior pixel}. \end{cases}
$$

A pixel is a boundary pixel if any of its 4 neighbours has a different clean
quaternion (a numerical equality test on the clean map, allowing for $q$ and
$-q$). This is exact for these piecewise-constant synthetic maps; it is not a
symmetry-aware grain reconstruction.

**Size of the errors.** A misindexed pixel's error is the disorientation between
a random orientation and the truth. For cubic symmetry this follows the
Mackenzie distribution: mean $\approx 40.7^\circ$, maximum $62.8^\circ$,
independent of $\theta_{\max}$ and $p_{\mathrm{mis}}$ (a simulation with 200,000
random pairs gives 40.76°; the test sets measure 40.6–40.9°).

---

## 4. Data augmentation (training only)

`TrainMaps` applies two augmentations, both physically exact.

**A. Global orientation rotation.** One random unit quaternion $g$ (uniform on
$SO(3)$) rotates every clean orientation of the map, before the noise is added:

$$
\boxed{q_i' = q_i \otimes g}
$$

The rotation acts on the **right** because symmetry acts on the left, so the two
commute. It leaves every misorientation between pixels unchanged,
$q_i' \otimes q_j'^{-1} = q_i \otimes q_j^{-1}$, so the denoising task is the
same; it only stops the network from learning the particular orientation
distribution DREAM.3D produced.

**B. Spatial augmentation** of the pixel grid:

| Transformation | Probability |
|---|---|
| Horizontal flip | 50% |
| Vertical flip | 50% |
| Rotation by $0^\circ$, $90^\circ$, $180^\circ$ or $270^\circ$ | one of the four, at random |

These are applied identically to the clean map, the noisy map and the
misindexing mask. They rearrange pixel positions (grain shapes) without changing
any orientation value.

---

## 5. Training and validation datasets

Script: `MODEL_PIPELINE/src/dataset.py`. Training and validation use **the same
process** (`_CleanMaps`) on different caches:

1. Read the clean Euler map from the split's cache.
2. Convert it to quaternions (Section 2.1).
3. Training only: augment (Section 4).
4. Draw a noise level and add the noise (Section 3).
5. Return the noisy map, the clean map and the misindexing mask.

| | `TrainMaps` | `ValMaps` |
|---|---|---|
| Cache | `split_train_clean.npy` (38,998 maps) | `split_val_clean.npy` (500 maps) |
| Augmentation | yes | no |
| Noise | a **fresh** draw on every access | **one fixed draw per map**, seeded by its map ID and reused every epoch |

**Why validation noise is frozen.** Each validation map gets a random level
from the same ranges as training, but the same realisation every epoch
(generated on the fly from a seed equal to the map ID, which is equivalent to
generating it once and storing it). A change in the validation score then means
the model changed, not the noise. This matters: in the earlier model's last
epochs the validation score improved by only about 0.001 per epoch, while fresh
noise every epoch would move it by roughly ten times that by chance, making the
choice of `best.pth` mostly luck.

**Outputs of each sample** (PyTorch tensors):

| Output | Shape | Type |
|---|---|---|
| Noisy quaternions | $128 \times 128 \times 4$ | float32 |
| Clean quaternions | $128 \times 128 \times 4$ | float32 |
| Misindexing mask | $128 \times 128$ | bool |

The datasets only prepare inputs, targets and masks; they do no optimisation.

---

## 6. Network

Scripts: `MODEL_PIPELINE/src/architecture.py` (`GatedOrientationUNet`) and
`MODEL_PIPELINE/src/orientation_torch.py` (orientation maths in PyTorch).
**1.83 million parameters.**

The design follows the two noise types. Scatter needs a **small correction**;
misindexing needs a **completely new orientation**. The network therefore has two
output paths, refine and replace, and a per-pixel **gate** that chooses between
them.

### 6.1 Inputs (91 channels per pixel, fixed computations)

| Input | Channels | Definition |
|---|---|---|
| Orientation | 9 | Rotation matrix $R(q_i)$ of the symmetry-reduced quaternion (the equivalent with the largest $\lvert q_w \rvert$) |
| Neighbour angles | 81 | $\min(\theta_{ik}/30^\circ,\,1)$ for every pixel $k$ of the $9 \times 9$ window around $i$ |
| Misfit | 1 | $\operatorname{median}_{j \in \mathcal{N}_8(i)} \theta_{ij} \,/\, 62.8^\circ$ over the 8 touching neighbours |

Before measuring $\theta_{ik}$, each neighbour is rewritten in the symmetry
variant closest to the centre pixel ($\tilde{q}_k$), so two pixels of the same
grain are never mistaken as different just because they are stored differently.
The misfit is a near-ready misindexing detector: a misindexed pixel disagrees with
all 8 neighbours (median about 40°), while a boundary pixel agrees with most of
them (small median).

### 6.2 U-Net trunk

Each block is two $3 \times 3$ convolutions, each followed by GroupNorm (8 groups)
and SiLU. Down-sampling is $2 \times 2$ max-pooling; up-sampling is bilinear
($\times 2$) followed by a $1 \times 1$ convolution, then concatenation with the
skip connection.

| Level | Map size | Channels (encoder → decoder) |
|---|---|---|
| 0 | $128 \times 128$ | 91 → 32 … 32 |
| 1 | $64 \times 64$ | 64 … 64 |
| 2 | $32 \times 32$ | 128 … 128 |
| 3 (bottom) | $16 \times 16$ | 256 |

At the bottom, one cell covers an $8 \times 8$ patch of the original map, so the
trunk sees a wide area; the skip connections restore pixel-level detail. The
output is 32 features per pixel, read by five $1 \times 1$ convolution heads:
$\tau$ (1), $\delta$ (81), refine (3), replace (6) and gate (1).

### 6.3 Learned kernel: which neighbours to average

The refine path starts from a weighted average of the 81 aligned neighbours.
The weights come from the neighbour angles, a per-pixel strictness $\tau_i$ and
per-neighbour corrections $\delta_{ik}$, both predicted by the network:

$$
\tau_i = \max\!\big(\operatorname{softplus}(h^{\tau}_i),\, 0.01\big)\ \text{[deg]},
\qquad
w_{ik} = \operatorname{softmax}_k\!\Big(-\frac{\theta_{ik}}{\tau_i} + \delta_{ik}\Big),
$$

$$
q_i^{\mathrm{base}} = \frac{\sum_k w_{ik}\,\tilde{q}_k}{\big\lVert \sum_k w_{ik}\,\tilde{q}_k \big\rVert}.
$$

Neighbours outside the map get weight 0. Because the weight falls off
exponentially with the angle, same-grain neighbours (small $\theta$) dominate
and other grains or misindexed pixels (tens of degrees) get almost nothing, so
grain boundaries stay sharp. Example with $\tau = 5^\circ$: a $3^\circ$ neighbour
weighs about 1,600 times more than a $40^\circ$ neighbour.

### 6.4 Refine, replace and gate

**Refine** (for scatter): a small extra rotation on top of the average,

$$
\mathbf{v}_i = \theta_r \tanh(h^{v}_i),\quad \theta_r = 25^\circ \text{ (each of the 3 components)},
\qquad
q_i^{\mathrm{ref}} = \frac{\exp(\mathbf{v}_i) \otimes q_i^{\mathrm{base}}}{\lVert\cdot\rVert},
$$

where $\exp(\mathbf{v}) = \big(\cos\frac{\lVert\mathbf{v}\rVert}{2},\, \frac{\mathbf{v}}{\lVert\mathbf{v}\rVert}\sin\frac{\lVert\mathbf{v}\rVert}{2}\big)$
turns the rotation vector into a quaternion. The total angle is at most
$25^\circ\sqrt{3} \approx 43^\circ$.

**Replace** (for misindexing): a brand-new orientation from 6 numbers
$(\mathbf{a}_1, \mathbf{a}_2)$, the continuous 6D representation of Zhou et al.
(2019), made into a rotation matrix by Gram–Schmidt:

$$
\mathbf{b}_1 = \frac{\mathbf{a}_1}{\lVert\mathbf{a}_1\rVert},\quad
\mathbf{b}_2 = \frac{\mathbf{a}_2 - (\mathbf{b}_1\!\cdot\!\mathbf{a}_2)\,\mathbf{b}_1}{\lVert\cdot\rVert},\quad
\mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2,
\qquad
R = [\mathbf{b}_1\ \mathbf{b}_2\ \mathbf{b}_3] \;\to\; q_i^{\mathrm{rep}}.
$$

**Gate** (chooses per pixel): a sigmoid with a dead zone, so a closed gate is
exactly closed,

$$
g_i = \frac{\max\big(0,\ \sigma(z_i) - 0.05\big)}{1 - 0.05} \in [0, 1].
$$

**Output:** spherical interpolation between the two paths (after aligning
$q^{\mathrm{rep}}$ to the closest symmetry variant of $q^{\mathrm{ref}}$),

$$
\boxed{q_i^{\mathrm{out}} = \operatorname{slerp}\big(q_i^{\mathrm{ref}},\, q_i^{\mathrm{rep}};\, g_i\big)}
$$

$g_i \approx 0$ keeps the refined pixel (scatter); $g_i \approx 1$ takes the
replacement (misindexing).

### 6.5 Initialisation

The untrained network is already a sensible filter:

- refine head weights 0, so $\mathbf{v} = 0$ (no extra rotation);
- gate bias $-4$, so $\sigma(-4) = 0.018 < 0.05$ and $g = 0$ (nothing replaced);
- replace bias $(1,0,0,0,1,0)$, the identity rotation;
- kernel heads give $\delta = 0$ and $\tau = 5^\circ$.

At epoch 0 the network is therefore an edge-preserving average filter with
$\tau = 5^\circ$; everything after that is learned.

---

## 7. Loss

Script: `MODEL_PIPELINE/src/architecture.py` (`denoise_loss`).

**Per-pixel error.** A Charbonnier function of the disorientation, with a
bounded gradient at zero error:

$$
\ell(q, q^{\star}) = \sqrt{1 - \cos\frac{\theta}{2} + \varepsilon} - \sqrt{\varepsilon},
\qquad \varepsilon = 10^{-10},
$$

with $\theta$ the disorientation between prediction $q$ and truth $q^{\star}$.
For $\theta$ above a tiny threshold, $\ell \approx \theta / (2\sqrt{2})$: it
behaves like an L1 loss on the angle (edge-preserving, and it keeps pushing on
small errors). The angle itself is never optimised directly, because the
gradient of $\arccos$ is infinite at zero error.

**Pixel weights.** Misindexed pixels ($b_i = 1$) and grain-boundary pixels
($e_i = 1$, from the clean map: disorientation above $0.1^\circ$ to any 4-neighbour)
count more:

$$
\omega_i = 1 + 5\,b_i + 2\,e_i \qquad (1,\ 3,\ 6 \text{ or } 8).
$$

**The four terms:**

$$
\mathcal{L}_{\mathrm{main}} = \frac{\sum_i \omega_i\, \ell(q_i^{\mathrm{out}}, q_i^{\star})}{\sum_i \omega_i},
\qquad
\mathcal{L}_{\mathrm{gate}} = \operatorname{BCE}\big(z_i,\ b_i\big),
$$

$$
\mathcal{L}_{\mathrm{refine}} = \operatorname*{mean}_{i:\,b_i = 0} \ell(q_i^{\mathrm{ref}}, q_i^{\star}),
\qquad
\mathcal{L}_{\mathrm{replace}} = \operatorname*{mean}_{i:\,b_i = 1} \ell(q_i^{\mathrm{rep}}, q_i^{\star}),
$$

$$
\boxed{\mathcal{L} = \mathcal{L}_{\mathrm{main}} + 1\cdot\mathcal{L}_{\mathrm{gate}} + 5\cdot\mathcal{L}_{\mathrm{refine}} + 0.5\cdot\mathcal{L}_{\mathrm{replace}}}
$$

The gate is supervised directly by the true misindexing mask, which the noise
generator records. The refine and replace terms grade each path on its own
pixels; they are needed because the gate starts closed, so without them the
replace path would receive almost no gradient and barely learn.

---

## 8. Training procedure

Scripts: `MODEL_PIPELINE/src/train.py` (the loop) and
`MODEL_PIPELINE/src/train_model.sh` (the GPU job; it **only trains** and saves
checkpoints).

| Setting | Value |
|---|---|
| Optimiser | AdamW, learning rate $\eta_0 = 2\times10^{-4}$, weight decay 0 |
| Learning-rate schedule | cosine, per epoch, down to $\eta_{\min} = 0.01\,\eta_0$ |
| Epochs | 60 |
| Batch size | 16 maps; $\lfloor 38998/16 \rfloor = 2437$ steps per epoch |
| Gradient clipping | norm 1.0 |
| Precision | bf16 mixed precision for the network; orientation geometry in float32 |
| Data loading | 8 worker processes |
| Seed | fixes the initial weights and the shuffling order (training noise stays unseeded) |

$$
\eta_e = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\Big(1 + \cos\frac{\pi e}{E}\Big),
\qquad E = 60.
$$

**After every epoch:** validate on the 500 validation maps (frozen noise) and
record ALL, BOUNDARY, CORRUPTED, $>10^\circ$ and the gate's precision and recall
(Section 9.1). The model-selection score is

$$
S = \big(\%\ \text{of pixels} > 10^\circ\big) + \overline{\theta}_{\mathrm{ALL}}\ [\deg],
$$

and the checkpoint with the lowest $S$ is saved as `best.pth`. `last.pth` is
saved every epoch, so an interrupted job resumes with `--resume auto`. Outputs go
to `MODEL_PIPELINE/checkpoints/runs/<config name>/`: `best.pth`, `last.pth`,
`history.json` (all per-epoch numbers), `run_info.json` (code version and
settings), `training_curves.png`.

**Runs:** three seeds (`blind_both_seed1/2/3`), identical except for the seed, to
measure run-to-run spread. The earlier model took about 5.8 minutes per epoch,
about 5.8 hours per run, on the cluster GPU.

---

## 9. Evaluation metrics and testing

### 9.1 Metrics

For each map, with $\theta_i$ the disorientation between the cleaned and the
true orientation of pixel $i$:

| Metric | Definition | What it shows |
|---|---|---|
| ALL | $\frac{1}{N}\sum_i \theta_i$ | Overall error (orientation mean absolute error) |
| BOUNDARY | mean of $\theta_i$ over grain-boundary pixels | Whether edges stay sharp |
| CORRUPTED | mean of $\theta_i$ over misindexed pixels | How well misindexed pixels are repaired |
| $>10^\circ$ | $100 \cdot \frac{1}{N}\sum_i \mathbf{1}[\theta_i > 10^\circ]$ | Share of badly wrong pixels (they create false grains) |

Every number is a **mean**: first over the pixels of each map, then over the
maps (decided 2026-10-13).

### 9.2 Testing (separate job, added after training)

Testing is a separate GPU job (`MODEL_PIPELINE/src/eval_model.sh`, which runs
`inference.py` on `best.pth`). It still needs updating to the new dataset code
before use.

- **Test sets** (predefined, saved in `datasets/noise_scatter_misindexing/`,
  maps 1–500):
  - a $7 \times 4$ grid: $\theta_{\max} \in \{1, 3, 5, 7, 9, 12, 15\}^\circ$ and
    $p_{\mathrm{mis}} \in \{0, 3, 6, 9\}\%$, boundary bias 4. Every set uses the
    same seed per map, so the sets differ only in the amount of noise (a paired
    design);
  - `test_random`: each map with its own random level
    ($\theta_{\max} \sim \mathcal{U}(0, 15^\circ)$, $p_{\mathrm{mis}} \sim \mathcal{U}(0, 10\%)$).
- **Baseline:** six MTEX filters (mean, median, Kuwahara, spline,
  half-quadratic, infimal convolution), each after pre-cleaning (grains smaller
  than 3 pixels deleted, then smoothed and filled), on maps 1–10 of every test
  set. MTEX gets the grain threshold $\max(10^\circ, 2\,\theta_{\max})$, which
  uses the true scatter level.
- **Statistics:** for each set and metric, a paired t-test over the maps against
  the closest MTEX filter on that metric, with a 95% confidence interval;
  Holm correction over the 29 sets.
- **Note on the new training range:** the model now never sees
  $p_{\mathrm{mis}} = 0$, so the 7 grid sets without misindexing are outside its
  training range.

---

## 10. Code: files and how they connect

| Stage | File (`MODEL_PIPELINE/src/`) | Role |
|---|---|---|
| Before training | `prepare_cache.py` | Clean caches, Section 1.3 |
| | `check_noise.py` | Measures the realised noise levels of training and validation maps |
| Network | `architecture.py` | Network (Section 6) and loss (Section 7) |
| | `orientation_torch.py` | Orientation maths in PyTorch: symmetry, disorientation, neighbour alignment, misfit, slerp, 6D |
| Training | `dataset.py` | `TrainMaps` and `ValMaps`, Sections 3–5 |
| | `train.py` | Training loop, validation, checkpoints, Section 8 |
| | `train_model.sh` | GPU job: trains and saves checkpoints only |
| | `plot_training.py` | Training curves, redrawn every epoch |
| After training | `inference.py` | Runs `best.pth` on a noisy set and saves the cleaned maps |
| | `eval_model.sh` | GPU job: testing (Section 9.2) |
| | `summarize_results.py`, `make_results_figures.py`, `plot_error_maps.py` | Results tables and figures |

Shared code used from `SHARED/src/`: `orientation.py` (numpy orientation maths),
`paths.py` (folders), `provenance.py` (config reading, run records),
`score_dataset.py` (scoring).

```
configs/train/<name>.yaml → train_model.sh → train.py (dataset.py + architecture.py)
     → checkpoints/runs/<name>/best.pth
     → (separate job) eval_model.sh → inference.py → cleaned maps
     → SHARED/src/score_dataset.py → scores
```

**Commands** (from `/project/community/aiosman`):

```bash
make train-gpu CONFIG=blind_both_seed1     # also seed2, seed3
make watch                                 # job queue
```

---

## 11. The training config file

Files: `MODEL_PIPELINE/configs/train/blind_both_seed{1,2,3}.yaml`.

### 11.1 How it is read

1. **You start it:** `make train-gpu CONFIG=blind_both_seed1`.
2. **`train_model.sh`** finds the file in `MODEL_PIPELINE/configs/train/`, uses
   the file name as the run name (checkpoints go to
   `MODEL_PIPELINE/checkpoints/runs/blind_both_seed1/`) and passes the file to
   `train.py` as `--config`.
3. **`train.py`** reads the `train:` section (through `parse_with_config` in
   `SHARED/src/provenance.py`):
   - each key is one of `train.py`'s options (`scatter_min_deg` is `--scatter-min-deg`);
   - the values become the defaults; a flag on the command line overrides them;
   - **a misspelled or unknown key stops the run with an error**, so a typo cannot
     silently fall back to a default;
   - the whole file is saved inside `run_info.json` and `best.pth`, so every model
     records the settings that made it.

### 11.2 Settings

| Setting(s) | Value | Goes to | What it does |
|---|---|---|---|
| `description` | — | — | A label for humans only |
| `scatter_min_deg`, `scatter_max_deg` | 1, 15 | `dataset.py` | $\theta_{\max} \sim \mathcal{U}(1^\circ, 15^\circ)$ |
| `misindex_min`, `misindex_max` | 0.01, 0.20 | `dataset.py` | $p_{\mathrm{mis}} \sim \mathcal{U}(0.01, 0.20)$ |
| `boundary_bias` | 4 | `dataset.py` | Boundary pixels 4× more likely to be misindexed |
| `seed` | 1, 2, 3 | `train.py` | Initial weights and shuffling order |
| `kernel`, `mean_radius` | learned, 4 | `architecture.py` | Learned averaging over a $9\times9$ window |
| `base_channels` | 32 | `architecture.py` | Trunk widths 32/64/128/256 |
| `max_refine_deg` | 25 | `architecture.py` | $\theta_r$ in Section 6.4 |
| `no_misfit` | false | `architecture.py` | Keeps the misfit input |
| `epochs`, `batch_size` | 60, 16 | `train.py` | Section 8 |
| `lr`, `weight_decay` | $2\times10^{-4}$, 0 | `train.py` | AdamW, cosine schedule |
| `loss`, `loss_eps` | charbonnier, $10^{-10}$ | `architecture.py` | Section 7 |
| `w_bad`, `w_bnd` | 5, 2 | `architecture.py` | Pixel weights $\omega_i$ |
| `w_gate`, `w_refine`, `w_replace` | 1, 5, 0.5 | `architecture.py` | Weights of the helper loss terms |

The job script sets `num_workers`, `checkpoints_dir` and `figures_dir`; those do
not belong in the config. Configs are never edited after their run starts: a new
experiment gets a new file.

---

## 12. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-12 | Training on GPU only | Project rule: training jobs run on GPU nodes, never on CPU |
| 2026-10-12 | Test grid of 28 sets ($7 \times 4$) plus a mixed set | See how each noise type and their combination affect every method |
| 2026-10-13 | Every error statistic is a mean, not a median | The median hides the few badly wrong pixels that matter most |
| 2026-10-14 | MTEX on maps 1–10 of each set, pre-cleaned filters only | See `RESEARCH_PLAN.md`, section 7 |
| 2026-10-23 | Project split into `MODEL_PIPELINE/`, `MTEX_PIPELINE/`, `SHARED/` | Clear separation of the model, the baseline and shared code |
| 2026-10-25 | `prepare_cache.py` rewritten: clean caches for train, val and test only | One simple job: `.txt` → `.npy`, no noise |
| 2026-10-25 | Noise: $\theta_{\max} \in [1^\circ, 15^\circ]$ and $p_{\mathrm{mis}} \in [1\%, 20\%]$, both always on | No scatter-only or misindexing-only maps, and a harder upper bound |
| 2026-10-25 | Validation = the same process as training on the validation cache, without augmentation | Assess the model under the same conditions it is trained on |
| 2026-10-26 | Validation noise frozen per map (seed = map ID) | A moving validation score must reflect the model, not the noise draw (Section 5) |
| 2026-10-26 | Training job only trains; testing is a separate job | Clean separation; the old end-of-training test on `scatter5` was a leftover |
| 2026-10-26 | New configs `blind_both_seed1/2/3` | The new noise design needs new runs; old configs stay as the record of old runs |

---

## 13. Earlier model (`blind_learned`) and its results

The same network and loss, trained before the changes above:

- **Noise:** $\theta_{\max} \sim \mathcal{U}(0, 15^\circ)$ and $p_{\mathrm{mis}} \sim \mathcal{U}(0, 10\%)$,
  drawn independently (so near scatter-only or misindexing-only maps occurred).
- **Validation:** a saved noisy set (maps 501–1000), median-based validation
  statistics (runs trained 2026-10-12), so `best.pth` was chosen on a
  median-based score.
- **Runs:** `blind_learned_seed1/2/3`.

**Results against the closest MTEX filter** (test maps 1–10; ours is the mean of
the 3 runs; all numbers are means):

| Test set ($\theta_{\max}$, $p_{\mathrm{mis}}$) | Metric | Noisy input | Best MTEX | Ours | Ours better by |
|---|---|---|---|---|---|
| 1°, 0% | ALL (°) | 0.50 | 0.081 (median) | 0.068 | 1.2× |
| 1°, 9% | ALL (°) | 4.11 | 0.628 (spline) | 0.268 | 2.3× |
| 15°, 0% | ALL (°) | 7.51 | 3.485 (median) | 0.919 | 3.8× |
| 15°, 9% | ALL (°) | 10.49 | 3.992 (median) | 1.211 | 3.3× |
| 1°, 3% | BOUNDARY (°) | 3.73 | 1.292 (spline) | 0.392 | 3.3× |
| 15°, 0% | BOUNDARY (°) | 7.44 | 7.141 (inf. conv.) | 1.342 | 5.3× |
| 1°, 3% | CORRUPTED (°) | 40.9 | 6.37 (spline) | 2.00 | 3.2× |
| 15°, 9% | CORRUPTED (°) | 40.6 | 7.94 (median) | 3.12 | 2.5× |
| 1°, 9% | $>10^\circ$ (%) | 8.94 | 1.343 (mean) | 0.361 | 3.7× |
| 15°, 0% | $>10^\circ$ (%) | 33.4 | 7.657 (median) | 0.056 | 137× |

**Significance:** significantly better in **99 of 109** comparisons (ALL 28/29,
BOUNDARY 26/29, CORRUPTED 22/22, $>10^\circ$ 23/29).

- *No significant difference:* ALL at 3°/0% and BOUNDARY at 1°, 3°, 5°/0% (the
  spline filter varies strongly from map to map and wins on some maps);
  $>10^\circ$ at 1°, 3°, 5°, 7°/0% (all methods near zero) and on the mixed set
  (ours lower on all 10 maps, $p = 0.07$ after Holm correction).
- *Significantly worse:* $>10^\circ$ at 9°/0% (+0.6 badly wrong pixels per map,
  out of 16,384).
- All of these are in the easiest conditions (low scatter, no misindexing).

**Limitations to state in the paper:** synthetic data only (the model is trained
on the same noise model it is tested on); MTEX received the true scatter level
for its grain threshold; MTEX was run on 10 maps per set.

Figures: `SHARED/figures/slides/curves_{all,boundary,corrupted,gross}.png`,
the same with significance marks in `curves_*_tests.png`, and the paper figures
in `SHARED/figures/paper/`.
