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

# Runs a trained model over every test set in configs/test_grid.txt, plus the
# mixed-level set test_random (one random level per map), and scores
# it against everything else already in each set's results folder (MTEX
# filters, other runs). One GPU job, so nothing is predicted on a CPU.
#
#     RUN=blind_learned_seed1 sbatch eval_model.sh
#
# RUN is a training experiment: its model is checkpoints/runs/<RUN>/best.pth
# and its predictions are written as datasets/mtex_out_<set>/<stem>__<RUN>.txt,
# so several runs can sit side by side in one score table. SETS overrides the
# list, e.g. SETS="scatter5 scatter5_mis6".

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
EXP="$PROJECT_DIR/MAPS_DENOISING_Orientation_Noise"
PYTHON="$HOME/miniconda3/envs/ebsd/bin/python"

[[ -n "${RUN:-}" ]] || { echo "ERROR: set RUN=<training experiment name>" >&2; exit 1; }
CKPT="$EXP/checkpoints/runs/$RUN/best.pth"
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
nvidia-smi || { echo "ERROR: no GPU in this allocation" >&2; exit 1; }
SETS="${SETS:-$(cat "$EXP/configs/test_grid.txt") test_random}"

echo "Job started: $(date) on $(hostname)   run $RUN"
cd "$EXP/src"

for SET in $SETS; do
    echo; echo "===== $SET ====="
    "$PYTHON" inference.py --checkpoint "$CKPT" --method-name "$RUN" \
        --clean-dir "$EXP/datasets/clean_euler" --noisy-dir "$EXP/datasets/noise_$SET"
    mkdir -p "$EXP/figures/$SET"
    # --all-maps: MTEX has run on only some maps of some sets; score every map
    # and let the per-method n column show how many each method covers.
    "$PYTHON" score_dataset.py \
        --clean-dir "$EXP/datasets/clean_euler" \
        --noisy-dir "$EXP/datasets/noise_$SET" \
        --results   "$EXP/datasets/mtex_out_$SET/" \
        --shape 128 128 --jobs 8 --all-maps \
        --csv "$EXP/figures/$SET/scores_mean.csv"
done

echo "Job finished: $(date)"
