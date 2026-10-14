#!/bin/bash
# Bring MTEX outputs from the package back into the results folders.
#
#     bash MAPS_DENOISING_Orientation_Noise/src/import_mtex_results.sh [mtex_out folder]
#
# Copies <folder>/<set>/*__<filter>.txt into datasets/mtex_out_<set>/, next to
# the network's predictions. The default folder is datasets/mtex_package/mtex_out
# (where to put the folder copied back from the MTEX computer).
#
# MTEX outputs from the earlier runs (Sept 2026: other settings, maps 1-10 of
# scatter5/7/9 only, and the scatter9 folder held copies of the scatter7 files)
# are MOVED, not deleted, to datasets/mtex_out_archive/<set>/ the first time,
# so old and new results never mix in one table.

set -euo pipefail

EXP=/project/community/aiosman/MAPS_DENOISING_Orientation_Noise
SRC="${1:-$EXP/datasets/mtex_package/mtex_out}"
OLD_FILTERS="mean median kuwahara spline halfquad infconv"

[[ -d "$SRC" ]] || { echo "ERROR: no folder $SRC" >&2; exit 1; }

for dir in "$SRC"/*/; do
    set=$(basename "$dir")
    dest="$EXP/datasets/mtex_out_$set"
    [[ -d "$EXP/datasets/noise_$set" ]] || { echo "skip $set: no test set of that name"; continue; }
    mkdir -p "$dest"

    archive="$EXP/datasets/mtex_out_archive/$set"
    if [[ ! -d "$archive" ]]; then
        moved=0
        for f in $OLD_FILTERS; do
            for variant in "$f" "clean-$f"; do
                for old in "$dest"/*__"$variant".txt; do
                    [[ -e "$old" ]] || continue
                    mkdir -p "$archive"
                    mv "$old" "$archive/"
                    moved=$((moved + 1))
                done
            done
        done
        [[ $moved -gt 0 ]] && echo "$set: moved $moved old MTEX outputs to $archive"
    fi

    n=$(find "$dir" -maxdepth 1 -name "*__*.txt" | wc -l)
    cp "$dir"/*__*.txt "$dest/"
    [[ -f "$dir/mtex_log.csv" ]] && cp "$dir/mtex_log.csv" "$dest/mtex_log.csv"
    echo "$set: imported $n MTEX outputs"
done
echo "done. Next: python3 $EXP/src/compare_with_mtex.py"
