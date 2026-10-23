#!/usr/bin/env python3
"""
Divide the clean maps (paths.CLEAN_DIR) into train / validation / test.

    test   maps    1-500     the maps MTEX was run on; every noise_* test set uses them
    val    maps  501-1000    model selection and baseline tuning
    train  maps 1001-40000   everything else

Contiguous id ranges are safe here because DREAM.3D generated every map with
the same settings: grains per map are ~52 in every id range (checked
2026-10-11 on 15 maps from each of six ranges).

Two outputs:

  SHARED/configs/splits/<split>.txt          one map id per line. Tracked in git, so
                                      the split is part of the experiment record.
  datasets/splits/<split>/clean_euler/ symlinks to the clean maps, so each split
                                      can be passed to any script as a folder.
                                      Links cost no disk, and the nightly GCS
                                      sync ignores them (gcloud's default), so
                                      the 20 GB is not backed up twice. Rerun
                                      this script to rebuild them.

Broken maps (empty or truncated, from an interrupted generation run) are left
out of every split. Rerunning is safe: existing links are kept.

Usage
-----
    python make_splits.py
"""

import argparse
import os
import re
from pathlib import Path

from paths import CLEAN_DIR

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"test": (1, 500), "val": (501, 1000), "train": (1001, 40000)}
MIN_BYTES = 400_000          # a healthy 128x128 map is 500-525 KB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", default=str(CLEAN_DIR))
    p.add_argument("--lists-dir", default=str(ROOT / "SHARED/configs/splits"))
    p.add_argument("--links-dir", default=str(ROOT / "datasets/splits"))
    args = p.parse_args()

    clean, lists, links = Path(args.clean_dir), Path(args.lists_dir), Path(args.links_dir)
    lists.mkdir(parents=True, exist_ok=True)

    maps, broken = {}, []
    for f in clean.glob("map_*_clean_euler.txt"):
        i = int(re.search(r"map_(\d+)", f.name).group(1))
        if f.stat().st_size >= MIN_BYTES:
            maps[i] = f
        else:
            broken.append(i)

    print(f"{len(maps) + len(broken)} maps in {clean}, {len(broken)} broken: {sorted(broken)}")
    for name, (lo, hi) in SPLITS.items():
        ids = sorted(i for i in maps if lo <= i <= hi)
        (lists / f"{name}.txt").write_text("".join(f"{i}\n" for i in ids))

        out = links / name / "clean_euler"
        out.mkdir(parents=True, exist_ok=True)
        for i in ids:
            link = out / maps[i].name
            if link.is_symlink() and not link.exists():
                link.unlink()                   # the clean maps have moved
            if not link.is_symlink():
                # Relative, so the tree still resolves if the project moves.
                link.symlink_to(os.path.relpath(maps[i], out))
        print(f"  {name:<5} maps {lo:>5}-{hi:<5}: {len(ids):>6} maps -> "
              f"{lists / f'{name}.txt'}, {out}")


if __name__ == "__main__":
    main()
