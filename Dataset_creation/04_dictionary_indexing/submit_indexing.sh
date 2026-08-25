#!/bin/bash
#SBATCH --job-name=emsoft_indexing
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/community/aiosman/logs/indexing_%j.out
#SBATCH --error=/project/community/aiosman/logs/indexing_%j.err

# Runs EMsoft's Dictionary Indexing (EMEBSDDI) on the H100 GPU
# to generate an orientation map from simulated diffraction patterns.

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
DATASET_DIR="$PROJECT_DIR/Dataset_creation"
EMSOFT_BIN="$PROJECT_DIR/emsoft_install/src/EMsoftBuild/Release/Bin"

# Pointing to the EMEBSDDI template we just configured
NML="$DATASET_DIR/04_dictionary_indexing/EMEBSDDI.nml"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

mkdir -p "$PROJECT_DIR/logs"

export PATH="$EMSOFT_BIN:$PATH"

# Fix for "Unknown CL error code : -1001" (CL_PLATFORM_NOT_FOUND_KHR)
export OCL_ICD_VENDORS=/etc/OpenCL/vendors

if [[ -z "$(ls -A "$HOME/.config/EMsoft" 2>/dev/null)" ]]; then
    echo "ERROR: EMsoft has not been configured yet (nothing in $HOME/.config/EMsoft)." >&2
    exit 1
fi

echo "Starting Dictionary Indexing on the GPU..."
EMEBSDDI "$NML"

echo "Job finished: $(date)"
# squeue -j 136063
# scancel 136443
# cat /project/community/aiosman/logs/mc_136063.out
# cat /project/community/aiosman/logs/mc_136063.err
# tail -f /project/community/aiosman/logs/mc_136446.out