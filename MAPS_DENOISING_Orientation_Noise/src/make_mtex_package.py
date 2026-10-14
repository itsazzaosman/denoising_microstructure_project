#!/usr/bin/env python3
"""
Build a small, self-contained folder of test maps for MTEX on another computer.

    datasets/mtex_package/
        run_mtex_baseline.m    the MATLAB script (copied from src/)
        README.txt             how to run it and bring the results back
        sets.txt               the test sets, one per line
        ang/<set>/             the noisy .ang files of the chosen maps
                               (+ levels.csv for test_random)
    datasets/mtex_package.zip  the same, zipped for download

Maps 1-10 of each set are packed by default; run_mtex_baseline.m uses only
map 1 unless you change its `maps` setting. The maps are the same test maps
the network was scored on, so the two can be compared map by map.

Usage
-----
    python make_mtex_package.py                 # 28 grid sets + test_random, maps 1-10
    python make_mtex_package.py --maps 1 2 3    # other maps
"""

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

README = """\
MTEX package: test maps for the MTEX baseline
=============================================

1. Copy this whole folder to the computer with MATLAB + MTEX 7.1.
2. Start MATLAB, run startup_mtex.
3. Open run_mtex_baseline.m from THIS folder and press Run.
   It runs spline, median, clean-spline and clean-median on map 1 of every
   set in sets.txt. For more maps, change `maps = 1` to `maps = 1:10` at the
   top (this folder holds maps 1-10). Finished outputs are skipped, so you
   can stop and start again.
4. Copy the folder mtex_out back to the cluster, to
   /project/community/aiosman/MAPS_DENOISING_Orientation_Noise/datasets/mtex_package/mtex_out
   and run (from /project/community/aiosman):
       bash MAPS_DENOISING_Orientation_Noise/src/import_mtex_results.sh
       python3 MAPS_DENOISING_Orientation_Noise/src/compare_with_mtex.py

Sets: {n_sets}   Maps per set in this folder: {maps}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--out", default=str(ROOT / "datasets/mtex_package"))
    args = p.parse_args()

    out = Path(args.out)
    sets = (ROOT / "configs/test_grid.txt").read_text().split() + ["test_random"]
    (out / "ang").mkdir(parents=True, exist_ok=True)

    n = 0
    for s in sets:
        src = ROOT / f"datasets/noise_{s}"
        dst = out / "ang" / s
        dst.mkdir(parents=True, exist_ok=True)
        for m in args.maps:
            f = src / "ang_files" / f"map_{m:05d}_clean_euler_noisy_noisy.ang"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)        # real copies: the folder leaves the cluster
            n += 1
        if (src / "levels.csv").exists():
            shutil.copy2(src / "levels.csv", dst / "levels.csv")

    shutil.copy2(ROOT / "src/run_mtex_baseline.m", out / "run_mtex_baseline.m")
    (out / "sets.txt").write_text("".join(f"{s}\n" for s in sets))
    maps = (f"{args.maps[0]}-{args.maps[-1]}" if args.maps == list(range(args.maps[0], args.maps[-1] + 1))
            else ", ".join(map(str, args.maps)))
    (out / "README.txt").write_text(README.format(n_sets=len(sets), maps=maps))

    # The zip holds the inputs only: an old mtex_out must not travel back and forth.
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            rel = f.relative_to(out.parent)
            if f.is_file() and "mtex_out" not in rel.parts:
                z.write(f, rel)
    print(f"{len(sets)} sets x {len(args.maps)} maps = {n} .ang files in {out}")
    print(f"zipped: {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
