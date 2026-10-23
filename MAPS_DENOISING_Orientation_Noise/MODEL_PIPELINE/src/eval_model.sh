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

# Runs a trained model over every test set in SHARED/configs/test_grid.txt, plus the
# mixed-level set test_random (one random level per map), and scores it
# against the other models and the MTEX filters of each set. One GPU job, so
# nothing is predicted on a CPU.
#
#     RUN=blind_learned_seed1 sbatch eval_model.sh
#
# RUN is a training experiment: its model is MODEL_PIPELINE/checkpoints/runs/<RUN>/best.pth
# and its predictions are written as
# datasets/results_test_set_model/<set>/<stem>__<RUN>.txt, so several runs sit
# side by side in one score table, next to datasets/results_test_set_mtex/<set>/. SETS overrides the
# list, e.g. SETS="scatter5 scatter5_mis6".
#
# A val_ set is the validation set instead: read from
# datasets/noise_scatter_misindexing_val/, predictions written to
# datasets/results_val_set_model/<set>/, scored with no MTEX results (none exist):
#     RUN=blind_learned_seed1 SETS=val_random sbatch eval_model.sh

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
EXP="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

[[ -n "${RUN:-}" ]] || { echo "ERROR: set RUN=<training experiment name>" >&2; exit 1; }
CKPT="$EXP/MODEL_PIPELINE/checkpoints/runs/$RUN/best.pth"
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
nvidia-smi || { echo "ERROR: no GPU in this allocation" >&2; exit 1; }
SETS="${SETS:-$(cat "$EXP/SHARED/configs/test_grid.txt") test_random}"

echo "Job started: $(date) on $(hostname)   run $RUN"
cd "$EXP/MODEL_PIPELINE/src"

for SET in $SETS; do
    echo; echo "===== $SET ====="
    if [[ "$SET" == val_* ]]; then
        NOISY="$EXP/datasets/noise_scatter_misindexing_val/noise_$SET"
        RESULTS=("$EXP/datasets/results_val_set_model/$SET")
    else
        NOISY="$EXP/datasets/noise_scatter_misindexing/noise_$SET"
        RESULTS=("$EXP/datasets/results_test_set_model/$SET" "$EXP/datasets/results_test_set_mtex/$SET")
    fi
    "$PYTHON" inference.py --checkpoint "$CKPT" --method-name "$RUN" --out "${RESULTS[0]}" \
        --clean-dir "$EXP/datasets/clean_euler_zip_unzip/clean_euler" --noisy-dir "$NOISY"
    mkdir -p "$EXP/SHARED/figures/$SET"
    # --all-maps: MTEX has run on only some maps of some sets; score every map
    # and let the per-method n column show how many each method covers.
    "$PYTHON" "$EXP/SHARED/src/score_dataset.py" \
        --clean-dir "$EXP/datasets/clean_euler_zip_unzip/clean_euler" \
        --noisy-dir "$NOISY" \
        --results   "${RESULTS[@]}" \
        --shape 128 128 --jobs 8 --all-maps \
        --csv "$EXP/SHARED/figures/$SET/scores_mean.csv"
done

echo "Job finished: $(date)"
