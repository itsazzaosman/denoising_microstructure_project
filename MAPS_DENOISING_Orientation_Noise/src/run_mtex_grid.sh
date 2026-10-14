#!/bin/bash
#SBATCH --job-name=mtex_grid
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/project/community/aiosman/logs/mtex_grid_%j.out
#SBATCH --error=/project/community/aiosman/logs/mtex_grid_%j.err

# Runs the MTEX filters (run_mtex_grid.m) on every test set: the 28 sets of
# configs/test_grid.txt plus test_random, maps 1-500 each. CPU only.
#
#     sbatch run_mtex_grid.sh
#     SETS="scatter5 scatter5_mis6" MAPS="1:20" sbatch run_mtex_grid.sh   # a test
#
# WORKERS MATLAB processes run side by side, one test set each, because the
# general QOS runs only one job per user at a time. Outputs that exist are
# skipped, so after the 12 h limit simply submit again to continue.
# Each set's MATLAB output goes to datasets/results_test_set_mtex/<set>/matlab.log.

set -euo pipefail

MATLAB=/project/community/aiosman/matlab/R2026a/bin/matlab
MTEX=/project/community/aiosman/matlab/mtex-7.1.0
EXP=/project/community/aiosman/MAPS_DENOISING_Orientation_Noise
SETS="${SETS:-$(cat "$EXP/configs/test_grid.txt") test_random}"
MAPS="${MAPS:-1:500}"
WORKERS="${WORKERS:-8}"

[[ -x "$MATLAB" ]] || { echo "ERROR: no MATLAB at $MATLAB" >&2; exit 1; }
mkdir -p /project/community/aiosman/logs
echo "Job started: $(date) on $(hostname)   $WORKERS workers, maps $MAPS"
echo "sets: $SETS"

run_set() {
    local set="$1"
    mkdir -p "$EXP/datasets/results_test_set_mtex/$set"
    "$MATLAB" -batch "addpath('$MTEX'); startup_mtex; addpath('$EXP/src'); run_mtex_grid('$set', 'maps', $MAPS)" \
        >> "$EXP/datasets/results_test_set_mtex/$set/matlab.log" 2>&1 \
        && echo "$(date +%T) done   $set" || echo "$(date +%T) FAILED $set (see datasets/results_test_set_mtex/$set/matlab.log)"
}
export -f run_set
export MATLAB MTEX EXP MAPS

echo $SETS | tr ' ' '\n' | xargs -P "$WORKERS" -I{} bash -c 'run_set {}'

echo "Job finished: $(date)"
