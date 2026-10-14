#!/bin/bash
#SBATCH --job-name=ebsd_orient_unet
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/project/community/aiosman/logs/orient_unet_%j.out
#SBATCH --error=/project/community/aiosman/logs/orient_unet_%j.err

# Trains the gated orientation U-Net (train.py) on maps with id > 500,
# corrupting them on the fly. Maps 1-500 are the MTEX baseline set and are
# never trained on.
#
# Cluster notes (same as the Gaussian project's script):
#  - partition/qos are "general"/"general_qos"; 12:00:00 is that partition's max
#  - no `module load`; call the conda env's interpreter directly, because
#    `conda activate` needs shell-hook init that a batch job does not have
#  - cpus-per-task=8 matches --num-workers 8
#
# train.py writes checkpoints/last.pth every epoch, so if the wall clock kills
# the job, resubmit to carry on:
#     sbatch train_model.sh --resume auto
#
# The same script is submitted to both `general` (12h, dedicated) and `preempt`
# (2 days, can be evicted). --dependency=singleton keeps only one of them
# running at a time, and --requeue plus `--resume auto` means an eviction costs
# at most the epoch in flight -- checkpoints/last.pth is written every epoch.

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
SCRIPT_DIR="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/src"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

mkdir -p "$PROJECT_DIR/logs"
[[ -x "$PYTHON" ]] || { echo "ERROR: no python at $PYTHON" >&2; exit 1; }

echo "Job started: $(date) on $(hostname)"
nvidia-smi || { echo "ERROR: no GPU in this allocation; training runs on a GPU only" >&2; exit 1; }
# CONFIG names an experiment: configs/train/<CONFIG>.yaml (or a path to a YAML
# file) holding every train.py setting. It also names the checkpoint folder,
# checkpoints/runs/<CONFIG>, and its `test_set:` picks the maps scored at the
# end. Resubmitting the same CONFIG with --resume auto continues that run.
#     CONFIG=scatter7 sbatch train_model.sh --resume auto
# Without CONFIG the old hard-coded settings below are used.
if [[ -n "${CONFIG:-}" ]]; then
    CFG_DIR="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/configs/train"
    if [[ -f "$CONFIG" ]]; then CONFIG="$(realpath "$CONFIG")"
    elif [[ -f "$CFG_DIR/$CONFIG.yaml" ]]; then CONFIG="$CFG_DIR/$CONFIG.yaml"
    else echo "ERROR: no config '$CONFIG' (also looked in $CFG_DIR)" >&2; exit 1; fi
    TRAIN_ARGS=(--config "$CONFIG")
    METHOD="${METHOD:-$(basename "$CONFIG" .yaml)}"   # name of the model in the score table
    CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/checkpoints/runs/$(basename "$CONFIG" .yaml)}"
    # Training curves go in the run's own folder, so runs in parallel do not
    # overwrite each other's figures/training_curves.png.
    TRAIN_ARGS+=(--figures-dir "$CKPT_DIR")
    NOISE="${NOISE:-$("$PYTHON" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["test_set"])' "$CONFIG")}"
else
    TRAIN_ARGS=(--epochs 40 --batch-size 16 --base-channels 32)
fi
CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/checkpoints_gpu}"
# NOISE picks the test set scored at the end (datasets/noise_scatter_misindexing/noise_$NOISE). It must
# match the training noise passed to train.py, e.g.
#     NOISE=scatter7 sbatch train_model.sh --noise-mode scatter --scatter-deg 7
NOISE="${NOISE:-scatter5}"
DATA="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/datasets"
FIGURES="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/figures/$NOISE"
echo "checkpoints -> $CKPT_DIR   test set -> noise_$NOISE"

cd "$SCRIPT_DIR"

"$PYTHON" train.py "${TRAIN_ARGS[@]}" --num-workers 8 --checkpoints-dir "$CKPT_DIR" "$@"

echo "Training finished: $(date)"

# Write the held-out predictions and score them against every MTEX filter.
"$PYTHON" inference.py --method-name "${METHOD:-unet-gpu}" --checkpoint "$CKPT_DIR/best.pth" \
    --clean-dir "$DATA/clean_euler_zip_unzip/clean_euler" --noisy-dir "$DATA/noise_scatter_misindexing/noise_$NOISE"

mkdir -p "$FIGURES"
"$PYTHON" score_dataset.py \
    --clean-dir "$DATA/clean_euler_zip_unzip/clean_euler" \
    --noisy-dir "$DATA/noise_scatter_misindexing/noise_$NOISE" \
    --results   "$DATA/results_test_set_model/$NOISE" "$DATA/results_test_set_mtex/$NOISE" \
    --shape 128 128 --jobs 8 --all-maps \
    --csv "$FIGURES/scores_mean.csv"

echo "Job finished: $(date)"
