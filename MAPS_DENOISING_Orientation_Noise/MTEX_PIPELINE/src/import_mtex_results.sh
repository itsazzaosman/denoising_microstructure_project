#!/bin/bash
# Bring MTEX outputs from the package back into the MTEX results folders.
#
#     bash MAPS_DENOISING_Orientation_Noise/MTEX_PIPELINE/src/import_mtex_results.sh [mtex_out folder]
#
# Copies <folder>/<set>/*__<filter>.txt (and mtex_log.csv) into
# datasets/results_test_set_mtex/<set>/. The default folder is
# datasets/mtex_package/mtex_out, where the folder copied back from the MTEX
# computer goes. The network's predictions are in
# datasets/results_test_set_model/<set>/; the September 2026 MTEX outputs (other
# settings, maps 1-10 of scatter5/7/9 only) are kept apart in
# datasets/results_test_set_mtex_old/.

set -euo pipefail

EXP=/project/community/aiosman/MAPS_DENOISING_Orientation_Noise
SRC="${1:-$EXP/datasets/mtex_package/mtex_out}"

[[ -d "$SRC" ]] || { echo "ERROR: no folder $SRC" >&2; exit 1; }

for dir in "$SRC"/*/; do
    set=$(basename "$dir")
    [[ -d "$EXP/datasets/noise_scatter_misindexing/noise_$set" ]] || { echo "skip $set: no test set of that name"; continue; }
    dest="$EXP/datasets/results_test_set_mtex/$set"
    mkdir -p "$dest"
    n=$(find "$dir" -maxdepth 1 -name "*__*.txt" | wc -l)
    cp "$dir"/*__*.txt "$dest/"
    [[ -f "$dir/mtex_log.csv" ]] && cp "$dir/mtex_log.csv" "$dest/mtex_log.csv"
    echo "$set: imported $n MTEX outputs"
done
echo "done. Next: python3 $EXP/SHARED/src/compare_with_mtex.py   and   make score-grid"
