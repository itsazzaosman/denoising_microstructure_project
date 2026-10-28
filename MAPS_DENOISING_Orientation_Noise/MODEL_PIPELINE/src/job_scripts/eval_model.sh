#!/bin/bash
#SBATCH --job-name=ebsd_orient_eval
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/project/community/aiosman/logs/orient_eval_%j.out
#SBATCH --error=/project/community/aiosman/logs/orient_eval_%j.err

# INFERENCE ONLY: runs a trained model over the 28 fixed-level test sets in
# SHARED/configs/test_grid.txt (maps 1-500 each) and saves its denoised maps.
# Scoring is a separate step on CPU, `make score-grid`, run whenever you need it.
#
#     CONFIG=blind_both_seed1 sbatch eval_model.sh
#
# CONFIG is the training config (MODEL_PIPELINE/configs/train/<CONFIG>.yaml). Its
# model is best.pth in the config's checkpoints_dir, e.g. MODEL_PIPELINE/checkpoints/seed1,
# and the denoised maps go to a results folder with the same name:
#     datasets/results_test_set_model/seed1/<set>/<stem>__<CONFIG>.txt
# SETS runs fewer sets, e.g. SETS="scatter5 scatter5_mis3".

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
EXP="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

[[ -n "${CONFIG:-}" ]] || { echo "ERROR: set CONFIG=<training config name>" >&2; exit 1; }
CFG_DIR="$EXP/MODEL_PIPELINE/configs/train"
if [[ -f "$CONFIG" ]]; then CONFIG="$(realpath "$CONFIG")"
elif [[ -f "$CFG_DIR/$CONFIG.yaml" ]]; then CONFIG="$CFG_DIR/$CONFIG.yaml"
else echo "ERROR: no config '$CONFIG' (also looked in $CFG_DIR)" >&2; exit 1; fi
RUN="$(basename "$CONFIG" .yaml)"             # the model's name in the score tables
# the config's checkpoints_dir (relative to $EXP unless absolute)
CKPT_DIR="$("$PYTHON" -c 'import sys, yaml, pathlib
d = pathlib.Path(yaml.safe_load(open(sys.argv[1]))["train"].get("checkpoints_dir", "MODEL_PIPELINE/checkpoints"))
print(d if d.is_absolute() else pathlib.Path(sys.argv[2]) / d)' "$CONFIG" "$EXP")"
CKPT="$CKPT_DIR/best.pth"
OUT_ROOT="$EXP/datasets/results_test_set_model/$(basename "$CKPT_DIR")"
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
nvidia-smi || { echo "ERROR: no GPU in this allocation" >&2; exit 1; }
SETS="${SETS:-$(cat "$EXP/SHARED/configs/test_grid.txt")}"

echo "Job started: $(date) on $(hostname)   run $RUN"
echo "model   : $CKPT"
echo "results : $OUT_ROOT/<set>/"
cd "$EXP/MODEL_PIPELINE/src"

for SET in $SETS; do
    echo; echo "===== $SET ====="
    "$PYTHON" inference.py --checkpoint "$CKPT" --method-name "$RUN" --out "$OUT_ROOT/$SET" \
        --clean-dir "$EXP/datasets/clean_euler_zip_unzip/clean_euler" \
        --noisy-dir "$EXP/datasets/noise_scatter_misindexing/noise_$SET"
done

echo; echo "Job finished: $(date)"
echo "score it (CPU):  make score-grid SEED=$(basename "$CKPT_DIR")"
