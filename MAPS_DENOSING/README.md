# EBSD Map Denoising

A U-Net that takes a Gaussian-noised EBSD orientation map and reconstructs the
clean one.

- **Input:** `datasets/ni_g_noisy_maps/` — clean maps with `np.random.normal(0, 25)` added
- **Target:** `datasets/ni_clean_maps/` — the ground truth
- **Pairing:** matching filenames, 40,000 pairs, 128×128 RGB

## Folder layout

```
MAPS_DENOSING/
├── datasets/
│   ├── ni_clean_maps/          ground truth (40,000 PNGs)
│   └── ni_g_noisy_maps/        noisy inputs  (40,000 PNGs)
│
├── checkpoints/                ← MODEL WEIGHTS
│   ├── best.pth                highest validation PSNR (use this for inference)
│   ├── last.pth                most recent epoch (use this to resume)
│   └── history.json            per-epoch metrics, feeds the plots
│
├── visualize/                  ← ALL PICTURES
│   ├── curves/
│   │   ├── loss_curve.png      training vs validation loss
│   │   ├── psnr_curve.png      validation PSNR vs the do-nothing baseline
│   │   └── ssim_curve.png      validation SSIM vs the do-nothing baseline
│   └── comparisons/
│       └── Map_XXXXX_comparison.png    clean | noisy | denoised, side by side
│
└── src/
    ├── paths.py                every directory, defined once — start here
    ├── architecture.py         the U-Net (and the older VAE, no longer used)
    ├── dataset.py              paired loading, train/val split, augmentation
    ├── metrics.py              PSNR and SSIM
    ├── visualize.py            all plotting
    ├── train.py                training loop
    ├── inference.py            run a checkpoint, write comparison figures
    └── train_model.sh          SLURM submission script
```

`checkpoints/` and `visualize/` are created automatically on the first run.

## Running it

Train (on the cluster):

```bash
sbatch src/train_model.sh                 # fresh run
sbatch src/train_model.sh --resume auto   # continue after a 12h wall-clock kill
```

Look at the results:

```bash
cd src
python inference.py                       # 4 validation maps, one figure each
python inference.py --num 6 --grid        # 6 maps stacked into a single figure
python inference.py --maps Map_00001.png  # a specific map
python visualize.py                       # redraw the curves from history.json
```

Score a large sample without writing figures:

```bash
python inference.py --num 500 --no-figures
```

## Reading the numbers

Both metrics are **higher is better**, and both are reported against the score
of doing nothing at all (feeding the noisy input straight through).

| Metric | Range | Meaning |
|---|---|---|
| PSNR | dB, higher better | per-pixel accuracy. +6 dB ≈ half the pixel error |
| SSIM | 0 → 1 | structural similarity — catches blurred grain boundaries that PSNR misses |

**Baseline: the noisy inputs score 15.27 dB / SSIM 0.38 against the clean maps.**
Anything the model produces has to beat that to be worth anything.

Watch both together. A model that learns to blur will raise PSNR while SSIM
stalls — and blurred grain boundaries are the one failure mode that matters
for orientation maps.

## Notes on the data

Two properties worth knowing, both measured from the files:

- **The stored noise is not zero-mean.** A third of clean pixels sit at exactly
  255 (IPF colouring saturates a channel inside each grain), so clipping to
  [0, 255] could only push those down. Measured over the whole image the
  difference is mean −20.9, std 38.3; restricted to the unclipped mid-range it
  is mean −2.5, std 28.5, which is the `N(0, 25)` you would expect.
- **The palette is continuous**, ~49 grains per map but no shared color set
  across the dataset. This is a regression problem, not segmentation.
