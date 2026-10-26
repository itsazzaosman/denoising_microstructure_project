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


set -euo pipefail

PROJECT_DIR=/project/community/aiosman
SCRIPT_DIR="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/MODEL_PIPELINE/src"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

mkdir -p "$PROJECT_DIR/logs"
[[ -x "$PYTHON" ]] || { echo "ERROR: no python at $PYTHON" >&2; exit 1; }

echo "Job started: $(date) on $(hostname)"
nvidia-smi || { echo "ERROR: no GPU in this allocation; training runs on a GPU only" >&2; exit 1; }
[[ -n "${CONFIG:-}" ]] || { echo "ERROR: set CONFIG=<training config name>" >&2; exit 1; }
CFG_DIR="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise/MODEL_PIPELINE/configs/train"
if [[ -f "$CONFIG" ]]; then CONFIG="$(realpath "$CONFIG")"
elif [[ -f "$CFG_DIR/$CONFIG.yaml" ]]; then CONFIG="$CFG_DIR/$CONFIG.yaml"
else echo "ERROR: no config '$CONFIG' (also looked in $CFG_DIR)" >&2; exit 1; fi
echo "config -> $CONFIG"

cd "$SCRIPT_DIR"

"$PYTHON" train.py --config "$CONFIG" "$@"

echo "Training finished: $(date)"
