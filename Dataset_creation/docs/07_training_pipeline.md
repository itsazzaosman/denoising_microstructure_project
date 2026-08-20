# 07 — Training pipeline (first attempt)

**Role:** train a denoising autoencoder on the real-orientation-derived
training pairs — the first attempt, later found to be bottlenecked by
dataset size rather than training time. That finding is what motivated
[`08_synthetic_training_data`](08_synthetic_training_data.md).

**Input:** [`05_noising_training_data/training_pairs.h5`](05_noising_training_data.md)
(4,125 patterns, SNR 1.27)
**Output:** `denoiser.keras` — superseded by `08`'s `denoiser_30.keras` as the
model actually evaluated in [`09_evaluation`](09_evaluation.md), but kept
here as the baseline that motivated the change

Needs TensorFlow — a separate environment (`~/.dl`, Python ≤3.13) from the
main repo `venv` (Python 3.14, which has no TensorFlow wheels yet).

## Architecture

Replicates Andrews et al. (2023), *Ultramicroscopy* 253, 113810 — a small
convolutional encoder-decoder:

```
Input (60,60,1)
  -> Conv2D(64, 1x1) -> Conv2D(128, 3x3) -> MaxPool 2x2
  -> Conv2D(128, 5x5) -> MaxPool 2x2
  -> Conv2DTranspose(128, 3x3, stride 2) -> Conv2DTranspose(64, 1x1, stride 2)
  -> Conv2D(1, 3x3, sigmoid)
```

Trained with Adam + MSE loss, batch size 64 — all matching the paper's
stated hyperparameters. The paper's own training loop isn't published
(referenced in their README but absent from their repository), so the loop
here is reconstructed from the paper's description.

## Two runs, same small dataset

| Run | Epochs | Gradient steps | Final train loss | Final val loss |
|---|---|---|---|---|
| First pass | 10 | ~410 | 0.0148 | 0.0144 |
| Second pass | 50 (`EARLY_STOP=True`, never triggered) | ~2,750 | 0.00940 | 0.00954 |

(Paper's own reported loss: 0.0022, on their data — not directly comparable,
but useful as an order-of-magnitude reference.)

The second run's loss curve is smooth, monotonic, and near-identical between
train and validation throughout — no overfitting — but visibly flattening
by epoch 50 (per-epoch improvement shrank from ~0.0065 early on to ~0.00005
near the end). `EarlyStopping`/`ReduceLROnPlateau` never fired despite that
visible flattening, because Keras's default `min_delta=0` counts *any*
decrease, however microscopic, as "still improving" — worth setting
`min_delta` explicitly (e.g. `1e-4`) if these callbacks are meant to
actually do something on a future long run.

## The actual conclusion: this is a data problem, not a training-time problem

`training_pairs.h5` has only 3,506 *training* patterns after the val split
— and they all come from one real scan whose 4,125 points reduce to only
~3,490 *distinct* orientations (neighboring points inside the same grain
repeat the same orientation). By epoch 50, the model had seen each unique
pattern roughly 50 times. The flattening curve is what a small, fixed
dataset looks like once it's been fully "wrung out" — more epochs on the
same 3,506 patterns yields shrinking returns, not more capability.

That's the reasoning that led directly to
[`08_synthetic_training_data`](08_synthetic_training_data.md): instead of
training longer on the same small, repetitive dataset, generate a much
larger and more diverse one.

## Outputs

- `training_loss.png` — the loss curve described above
- `denoise_examples.png` — noisy/denoised/clean example grid; visibly
  sharper at 50 epochs than at 10, but still slightly blurred relative to
  the clean target (typical of an MSE-trained, still slightly data-limited
  denoiser)
- `training_history.json` — full per-epoch loss/val_loss/learning_rate
