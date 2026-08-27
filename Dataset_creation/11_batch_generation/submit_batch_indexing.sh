#!/bin/bash
#SBATCH --job-name=emsoft_batch
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/community/aiosman/logs/batch_%j.out
#SBATCH --error=/project/community/aiosman/logs/batch_%j.err

# Runs EMEBSD + EMEBSDDI for maps 1-10, both clean and noisy variants
# (20 runs total), sequentially in ONE job - not a SLURM array.
#
# Why not an array: this account's QOS caps it to 1 running job at a time
# (MaxJobsPerUser=1) and 5 submitted jobs at a time (MaxSubmitJobsPerUser=5),
# so an array job buys no real parallelism here and a large array can be
# rejected outright at submission. A single sequential job sidesteps both
# limits - and at this reduced scale (~8 min/map x 20 = ~2.7 hours), it
# comfortably fits inside one job well under the partition's 12-hour cap.

set -euo pipefail

PROJECT_DIR=/project/community/aiosman
DATASET_DIR="$PROJECT_DIR/Dataset_creation"
EMSOFT_BIN="$PROJECT_DIR/emsoft_install/src/EMsoftBuild/Release/Bin"

export PATH="$EMSOFT_BIN:$PATH"
# Fix for "Unknown CL error code : -1001" (CL_PLATFORM_NOT_FOUND_KHR)
export OCL_ICD_VENDORS=/etc/OpenCL/vendors

mkdir -p "$PROJECT_DIR/logs"

if [[ -z "$(ls -A "$HOME/.config/EMsoft" 2>/dev/null)" ]]; then
    echo "ERROR: EMsoft has not been configured yet (nothing in $HOME/.config/EMsoft)." >&2
    exit 1
fi

echo "Job started: $(date) on $(hostname)"

failed=0
for i in $(seq 1 10); do
    for variant in clean noisy; do
        echo "--- map $i ($variant) ---"
        if ! python3 "$DATASET_DIR/11_batch_generation/run_one_map.py" "$i" "$variant"; then
            echo "*** map $i ($variant) FAILED - continuing with the rest ***"
            failed=$((failed + 1))
        fi
    done
done

echo "Job finished: $(date) - $failed failure(s) out of 20 runs"
exit $((failed > 0 ? 1 : 0))
