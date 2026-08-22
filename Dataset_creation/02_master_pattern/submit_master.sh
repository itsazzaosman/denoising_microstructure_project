#!/bin/bash
#SBATCH --job-name=emsoft_master_pattern
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/community/aiosman/logs/master_%j.out
#SBATCH --error=/project/community/aiosman/logs/master_%j.err

# Runs EMsoft's master pattern simulation (EMEBSDmaster) on
# 02_master_pattern/EMEBSDmaster_hires.nml to produce Ni_master_hires.h5.
#
# CPU/OpenMP only - no GPU needed (no platid/devid in this nml, just
# nthreads = 16, matched by --cpus-per-task above).
#
# Prerequisites (same as submit_jobs.sh):
#   - EMsoftinit must already have been run once interactively, with
#     EMdatapathname -> /project/community/aiosman/Dataset_creation and
#     EMXtalFolderpathname -> .../Dataset_creation/00_crystal_structure
#   - 01_monte_carlo/Ni_MC.h5 must already exist (this nml's
#     copyfromenergyfile copies its MC statistics rather than recomputing
#     them, via h5copypath - already pointed at our built SDK's h5copy)

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
DATASET_DIR="$PROJECT_DIR/Dataset_creation"
EMSOFT_BIN="$PROJECT_DIR/emsoft_install/src/EMsoftBuild/Release/Bin"
NML="$DATASET_DIR/02_master_pattern/EMEBSDmaster_hires.nml"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs available: $(nproc)"

mkdir -p "$PROJECT_DIR/logs"

export PATH="$EMSOFT_BIN:$PATH"

# Fix for OpenBLAS flooding stderr with "Detect OpenMP Loop and this
# application may hang" (once per BLAS call from inside EMEBSDmaster's
# own OpenMP threads - hundreds of thousands of lines otherwise): force
# OpenBLAS single-threaded since the outer nthreads=16 parallelism
# already comes from EMEBSDmaster itself.
export OPENBLAS_NUM_THREADS=1

if [[ -z "$(ls -A "$HOME/.config/EMsoft" 2>/dev/null)" ]]; then
    echo "ERROR: EMsoft has not been configured yet (nothing in $HOME/.config/EMsoft)." >&2
    echo "        Run 'EMsoftinit' interactively once before submitting this job." >&2
    exit 1
fi

if [[ ! -f "$DATASET_DIR/01_monte_carlo/Ni_MC.h5" ]]; then
    echo "ERROR: $DATASET_DIR/01_monte_carlo/Ni_MC.h5 not found." >&2
    echo "        Run submit_jobs.sh (the Monte Carlo step) first." >&2
    exit 1
fi

EMEBSDmaster "$NML"

echo "Job finished: $(date)"

# squeue -j 136063
# scancel 136443
# cat /project/community/aiosman/logs/master_136063.out
# cat /project/community/aiosman/logs/master_136063.err
# tail -f /project/community/aiosman/logs/master_136446.out
