#!/bin/bash
#SBATCH --job-name=emsoft_monte_carlo
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/project/community/aiosman/logs/mc_%j.out
#SBATCH --error=/project/community/aiosman/logs/mc_%j.err

# Runs EMsoft's Monte Carlo backscatter simulation (EMMCOpenCL) on
# 01_monte_carlo/MCNi.nml to produce Ni_MC.h5.
#
# NOTE: the nml's `dataname = '01_monte_carlo/Ni_MC.h5'` is relative to
# EMsoftinit's configured EMdatapathname, NOT to this script's $PWD.
# EMsoftinit must be run interactively ONCE, before ever submitting
# this job, to set:
#   EMdatapathname      -> /project/community/aiosman/Dataset_creation
#   EMXtalFolderpathname -> /project/community/aiosman/Dataset_creation/00_crystal_structure
#     (Ni.xtal, referenced by the nml's xtalname, already lives there)
# See /project/community/aiosman/emsoft_install/scripts/05_configure_environment.sh

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
DATASET_DIR="$PROJECT_DIR/Dataset_creation"
EMSOFT_BIN="$PROJECT_DIR/emsoft_install/src/EMsoftBuild/Release/Bin"
NML="$DATASET_DIR/01_monte_carlo/MCNi.nml"

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

mkdir -p "$PROJECT_DIR/logs"

export PATH="$EMSOFT_BIN:$PATH"

if [[ -z "$(ls -A "$HOME/.config/EMsoft" 2>/dev/null)" ]]; then
    echo "ERROR: EMsoft has not been configured yet (nothing in $HOME/.config/EMsoft)." >&2
    echo "        Run 'EMsoftinit' interactively once before submitting this job -" >&2
    echo "        it will prompt for EMdatapathname/EMXtalFolderpathname (see comment above)." >&2
    exit 1
fi

EMMCOpenCL "$NML"

echo "Job finished: $(date)"
