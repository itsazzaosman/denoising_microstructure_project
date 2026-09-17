#!/usr/bin/env python3
"""
Pack the clean Euler .txt maps into one memmappable .npy array.

Parsing a 512 KB text map costs ~12 ms, so a 12k-map epoch would spend ~2.5
CPU-minutes just re-parsing text it already read last epoch. Packing once into
a float32 array drops that to a memmap read.

The test maps (1-500) are packed separately, together with their PRE-GENERATED
noisy maps and bad-pixel masks, because those are the exact inputs MTEX was
given -- the test set must not be re-noised or the comparison stops being
like-for-like.

Usage
-----
    python prepare_cache.py --shape 128 128 --n-train 12000
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from orientation import load_euler

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "datasets/clean_euler"
NOISY_DIR = ROOT / "datasets/noisy_mis05"
CACHE_DIR = ROOT / "datasets/cache"

TEST_MAX_ID = 500          # maps 1-500 are the MTEX baseline set; never train on them


def map_id(path):
    m = re.search(r"map_(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def _read(args):
    path, n = args
    try:
        e = load_euler(path)
        return (e.astype(np.float32), None) if e.shape[0] == n else (None, "wrong length")
    except Exception as exc:
        return None, str(exc)


def usable(paths, min_bytes=400_000):
    """Drop maps that are obviously truncated before allocating anything.

    A healthy 128x128 map is 500-523 KB. The dataset has a handful of short or
    zero-byte maps left over from an interrupted generation run; filtering them
    here means the output array can be allocated at exactly the right size.
    """
    ok, skipped = [], []
    for f in paths:
        try:
            if Path(f).stat().st_size >= min_bytes:
                ok.append(f)
            else:
                skipped.append(Path(f).name)
        except OSError:
            skipped.append(Path(f).name)
    if skipped:
        print(f"  skipping {len(skipped)} short/unreadable maps: {skipped[:6]}"
              + (" ..." if len(skipped) > 6 else ""))
    return ok


def pack(paths, n, out, jobs, label):
    """Read `paths` in parallel into a single (M, n, 3) float32 .npy.

    Rows are filled contiguously and the map ids are saved alongside, so a read
    that fails anyway leaves unused zero rows at the END of the array rather
    than a hole in the middle. The loader takes its length from the ids file,
    never from the array's shape.
    """
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                    shape=(len(paths), n, 3))
    kept, ids, bad = 0, [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for path, (e, err) in zip(paths, pool.map(_read, [(p, n) for p in paths],
                                                  chunksize=16)):
            if e is None:
                bad.append((Path(path).name, err))
                continue
            arr[kept] = e
            ids.append(map_id(path))
            kept += 1
            if kept % 2000 == 0:
                print(f"  {label}: {kept}/{len(paths)}", flush=True)
    arr.flush()
    del arr

    np.save(out.with_name(out.stem + "_ids.npy"), np.array(ids, dtype=np.int32))
    print(f"  {label}: wrote {kept} maps to {out.name}"
          + (f"   ({len(bad)} unusable: {bad[:4]})" if bad else ""))
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=2, type=int, default=(128, 128), metavar=("NY", "NX"))
    p.add_argument("--n-train", type=int, default=12000,
                   help="how many training maps to pack (0 = every usable map)")
    p.add_argument("--jobs", type=int, default=16)
    args = p.parse_args()

    n = int(np.prod(args.shape))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    every = sorted(CLEAN_DIR.glob("map_*_clean_euler.txt"), key=lambda f: map_id(f) or 0)
    train_paths = usable([f for f in every if (map_id(f) or 0) > TEST_MAX_ID])
    if args.n_train:
        train_paths = train_paths[:args.n_train]

    print(f"clean maps found : {len(every)}")
    print(f"train pool       : ids > {TEST_MAX_ID}, packing {len(train_paths)}")
    pack(train_paths, n, CACHE_DIR / "train_clean.npy", args.jobs, "train")

    # ---- test set: clean + the exact noisy inputs MTEX was given ----
    noisy_paths = sorted(NOISY_DIR.glob("map_*_noisy.txt"), key=lambda f: map_id(f) or 0)
    noisy_paths = [f for f in noisy_paths if (map_id(f) or 0) <= TEST_MAX_ID]
    ids = [map_id(f) for f in noisy_paths]
    clean_paths = [CLEAN_DIR / f"map_{i:05d}_clean_euler.txt" for i in ids]
    good = set(usable(clean_paths))
    keep = [i for i, c in enumerate(clean_paths) if c in good]
    noisy_paths = [noisy_paths[i] for i in keep]
    clean_paths = [clean_paths[i] for i in keep]
    ids = [ids[i] for i in keep]

    print(f"test set         : {len(ids)} maps (ids {min(ids)}-{max(ids)})")
    pack(clean_paths, n, CACHE_DIR / "test_clean.npy", args.jobs, "test clean")
    pack(noisy_paths, n, CACHE_DIR / "test_noisy.npy", args.jobs, "test noisy")

    masks = np.zeros((len(ids), n), dtype=bool)
    missing = 0
    for k, f in enumerate(noisy_paths):
        mp = Path(str(f) + ".badmask.npy")
        if mp.exists():
            masks[k] = np.load(mp).ravel()
        else:
            missing += 1
    np.save(CACHE_DIR / "test_badmask.npy", masks)
    print(f"  test masks: wrote {len(ids)} ({missing} missing)")
    print(f"\ncache in {CACHE_DIR}")


if __name__ == "__main__":
    main()
