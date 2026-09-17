#!/bin/bash
#SBATCH --job-name=ebsd_orient_unet
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/project/community/aiosman/logs/orient_unet_cpu_%j.out
#SBATCH --error=/project/community/aiosman/logs/orient_unet_cpu_%j.err

# CPU-only twin of train_model.sh.
#
# Every GPU on this cluster was allocated with ~470 jobs queued, and slurm put
# the GPU job's estimated start 40 hours out -- while 330 CPU cores sat idle.
# At base 32 a training step is ~2 s on 8 cores, so a 32-core allocation gets
# through an epoch of 11,760 maps in roughly 10 minutes, which is fast enough
# to be worth doing rather than waiting.
#
# It shares --job-name and the checkpoint directory with the GPU jobs, and
# `--dependency=singleton` keeps only one of them running at a time. The
# architecture is identical (base 32), so whichever gets scheduled resumes from
# the other's checkpoint: if a GPU frees up overnight, the GPU job picks up
# exactly where this one stopped.

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
SCRIPT_DIR="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/src"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

mkdir -p "$PROJECT_DIR/logs"
[[ -x "$PYTHON" ]] || { echo "ERROR: no python at $PYTHON" >&2; exit 1; }

echo "Job started: $(date) on $(hostname)  [CPU, ${SLURM_CPUS_PER_TASK:-?} cores]"
CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/checkpoints}"
# METHOD names the output files, so parallel variants (different
# --mean-radius, say) can all land in mtex_out/ and be compared in one table.
echo "checkpoints -> $CKPT_DIR"

cd "$SCRIPT_DIR"

"$PYTHON" train.py --epochs 40 --batch-size 16 --base-channels 32 \
    --num-workers 8 --no-amp --checkpoints-dir "$CKPT_DIR" "$@"

echo "Training finished: $(date)"

"$PYTHON" inference.py --method-name "${METHOD:-unet}" \
    --checkpoint "$CKPT_DIR/best.pth"

"$PYTHON" score_dataset.py \
    --clean-dir "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/datasets/clean_euler" \
    --noisy-dir "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/datasets/noisy_mis05" \
    --results   "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/mtex_out/" \
    --shape 128 128 --jobs 8 --all-maps \
    --csv "$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/figures/scores.csv"

echo "Job finished: $(date)"
