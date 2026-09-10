#!/bin/bash
#SBATCH --job-name=ebsd_vae_train
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/community/aiosman/logs/vae_train_%j.out
#SBATCH --error=/project/community/aiosman/logs/vae_train_%j.err

# Trains the EBSD denoising VAE (train.py) on the noisy/clean map pairs in
# /project/community/aiosman/datasets/ni_vae/.
#
# Cluster notes:
#  - partition/qos match this cluster's actual setup ("general"/"general_qos");
#    there is no "gpu" partition here. 12:00:00 is the general partition's max.
#  - no `module load` - this cluster doesn't provide python/cuda modules, so we
#    call the conda env's interpreter directly. `conda activate` is avoided on
#    purpose: it needs shell-hook init that isn't present in a batch job.
#  - the "diffusion" env has torch 2.12.1+cu130, torchvision and PIL.
#  - cpus-per-task=8 leaves headroom for train.py's DataLoader(num_workers=4).

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
SCRIPT_DIR="$PROJECT_DIR/Dataset_creation/13_training_inference"
PYTHON="$HOME/miniconda3/envs/diffusion/bin/python"

mkdir -p "$PROJECT_DIR/logs"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python not found at $PYTHON" >&2
    exit 1
fi

echo "Job started: $(date) on $(hostname)"
nvidia-smi

# cd so `import dataset` / `import architecture` resolve and the saved
# ebsd_vae_weights.pth lands next to the code.
cd "$SCRIPT_DIR"

"$PYTHON" train.py

echo "Job finished: $(date)"
