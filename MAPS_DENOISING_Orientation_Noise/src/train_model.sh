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
nvidia-smi || true
CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/checkpoints_gpu}"
echo "checkpoints -> $CKPT_DIR"

cd "$SCRIPT_DIR"

"$PYTHON" train.py --epochs 40 --batch-size 16 --base-channels 32 --num-workers 8 --checkpoints-dir "$CKPT_DIR" "$@"

echo "Training finished: $(date)"

# Write the held-out predictions and score them against every MTEX filter.
"$PYTHON" inference.py --method-name unet-gpu --checkpoint "$CKPT_DIR/best.pth"

"$PYTHON" score_dataset.py \
    --clean-dir "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/datasets/clean_euler" \
    --noisy-dir "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/datasets/noisy_mis05" \
    --results   "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/mtex_out/" \
    --shape 128 128 --jobs 8 --all-maps \
    --csv "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/figures/scores.csv"

echo "Job finished: $(date)"
