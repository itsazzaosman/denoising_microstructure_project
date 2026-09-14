#!/bin/bash
#SBATCH --job-name=ebsd_unet_train
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/community/aiosman/logs/unet_train_%j.out
#SBATCH --error=/project/community/aiosman/logs/unet_train_%j.err

# Trains the EBSD denoising U-Net (train.py) on the noisy/clean map pairs in
# MAPS_DENOSING/datasets/{ni_clean_maps,ni_g_noisy_maps}.
#
# Cluster notes:
#  - partition/qos match this cluster's actual setup ("general"/"general_qos");
#    there is no "gpu" partition here. 12:00:00 is the general partition's max.
#  - no `module load` - this cluster doesn't provide python/cuda modules, so we
#    call the conda env's interpreter directly. `conda activate` is avoided on
#    purpose: it needs shell-hook init that isn't present in a batch job.
#  - the "ebsd" env has torch 2.12.1+cu130, torchvision and PIL.
#  - cpus-per-task=8 matches train.py's --num-workers 8.
#
# Outputs (all resolved by src/paths.py, relative to MAPS_DENOSING/):
#   checkpoints/best.pth, last.pth, history.json
#   visualize/curves/       redrawn every epoch, safe to look at mid-run
#   visualize/comparisons/  written when training finishes
#
# train.py checkpoints every epoch to checkpoints/last.pth, so if the 12h wall
# clock kills the job, resubmit and it picks up where it stopped:
#     sbatch train_model.sh --resume auto
# Any other train.py flag can be passed the same way.

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
SCRIPT_DIR="$PROJECT_DIR/MAPS_DENOSING/src"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

mkdir -p "$PROJECT_DIR/logs"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python not found at $PYTHON" >&2
    exit 1
fi

if [[ ! -d "$SCRIPT_DIR" ]]; then
    echo "ERROR: script dir not found at $SCRIPT_DIR" >&2
    exit 1
fi

echo "Job started: $(date) on $(hostname)"
# `|| true` so a diagnostic hiccup cannot kill the job under `set -e`.
nvidia-smi || true

# cd so `import dataset` / `import architecture` resolve.
cd "$SCRIPT_DIR"

"$PYTHON" train.py --epochs 50 --batch-size 64 --num-workers 8 "$@"

echo "Job finished: $(date)"
