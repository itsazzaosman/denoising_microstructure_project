#!/bin/bash
#SBATCH --job-name=ebsd_denoiser_synthetic
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/project/community/aiosman/emsoft_ebsd/logs/train_%j.out
#SBATCH --error=/project/community/aiosman/emsoft_ebsd/logs/train_%j.err

# EDIT ME: where you rsync'd EMsoftData_work to on the cluster, and the
# python interpreter inside the venv you created there and installed
# tensorflow/h5py/matplotlib/numpy into. That venv must be Python <=3.13 —
# TensorFlow has no wheels for 3.14 yet, which is why this is a separate
# venv from the main repo one (same reason ~/.dl exists locally: Python
# 3.13.15 + tensorflow 2.21.0).
PROJECT_DIR=/project/community/aiosman/emsoft_ebsd
VENV_PYTHON=$PROJECT_DIR/venv/bin/python

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

mkdir -p "$PROJECT_DIR/logs"

export PYTHONNOUSERSITE=1

PYVER=$("$VENV_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "Using $VENV_PYTHON (Python $PYVER)"
if [[ "$PYVER" == "3.14" ]]; then
    echo "ERROR: $VENV_PYTHON is Python $PYVER. TensorFlow has no wheel for 3.14 yet." >&2
    echo "        Rebuild this venv with Python <=3.13 before resubmitting." >&2
    exit 1
fi

"$VENV_PYTHON" "$PROJECT_DIR/08_synthetic_training_data/train_autoencoder_synthetic.py"

echo "Job finished: $(date)"