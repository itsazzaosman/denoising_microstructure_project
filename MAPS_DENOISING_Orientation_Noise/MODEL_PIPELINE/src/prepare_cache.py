#!/usr/bin/env python3
"""Pack CLEAN Euler-angle text maps into memory-mappable NumPy caches.

Reads map IDs from SHARED/configs/splits/{train,val,test}.txt and writes:
    datasets/cache/split_train_clean.npy
    datasets/cache/split_train_clean_ids.npy
    datasets/cache/split_val_clean.npy
    datasets/cache/split_val_clean_ids.npy
    datasets/cache/split_test_clean.npy
    datasets/cache/split_test_clean_ids.npy

No noise generation, noisy inputs, masks, or quaternion conversion.
Original Euler angles are stored as float32, in the same order returned by
SHARED/src/orientation.py:load_euler.

From MODEL_PIPELINE/src:
    python prepare_cache.py                  # process all three splits
    python prepare_cache.py --split val      # only validation
    python prepare_cache.py --split test --jobs 8
    python prepare_cache.py --all --overwrite  # intentionally replace caches
"""

import argparse
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SHARED" / "src"))
from orientation import load_euler
from paths import CLEAN_DIR

SPLITS_DIR = ROOT / "SHARED" / "configs" / "splits"
CACHE_DIR = ROOT / "datasets" / "cache"
VALID_SPLITS = ("train", "val", "test")


def read_ids(split):
    source = SPLITS_DIR / f"{split}.txt"
    if not source.is_file():
        raise FileNotFoundError(f"Missing split file: {source}")
    try:
        ids = [int(x) for x in source.read_text().split()]
    except ValueError as exc:
        raise ValueError(f"Non-integer map ID in {source}") from exc
    if not ids:
        raise ValueError(f"No map IDs in {source}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate map IDs in {source}")
    if any(i < 1 for i in ids):
        raise ValueError(f"Invalid map ID in {source}: IDs must be positive")
    if split == "test" and any(i > 500 for i in ids):
        raise ValueError(f"Test split {source} must only contain IDs 1–500")
    if split == "val" and any(not 501 <= i <= 1000 for i in ids):
        raise ValueError(f"Validation split {source} must only contain IDs 501–1000")
    if split == "train" and any(i <= 1000 for i in ids):
        raise ValueError(f"Training split {source} must only contain IDs >= 1001")
    return ids


def read_map(task):
    map_id, pixels = task
    path = CLEAN_DIR / f"map_{map_id:05d}_clean_euler.txt"
    try:
        data = np.asarray(load_euler(path))
        if data.shape != (pixels, 3):
            raise ValueError(f"shape {data.shape}; expected ({pixels}, 3)")
        if not np.isfinite(data).all():
            raise ValueError("NaN or infinity in Euler angles")
        return data.astype(np.float32), None
    except Exception as exc:
        return None, f"map {map_id} ({path}): {exc}"


def pack_split(split, ids, pixels, jobs, overwrite):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data_path = CACHE_DIR / f"split_{split}_clean.npy"
    ids_path = CACHE_DIR / f"split_{split}_clean_ids.npy"
    if not overwrite and (data_path.exists() or ids_path.exists()):
        raise FileExistsError(
            f"Cache already exists for {split}: {data_path} / {ids_path}. "
            "Use --overwrite only if you intend to replace it."
        )

    # Write temporary files first; invalid data never replaces an existing cache.
    token = uuid.uuid4().hex
    temp_data = CACHE_DIR / f".split_{split}_clean_{token}.npy"
    temp_ids = CACHE_DIR / f".split_{split}_clean_ids_{token}.npy"
    arr = None
    try:
        arr = np.lib.format.open_memmap(
            temp_data, mode="w+", dtype=np.float32,
            shape=(len(ids), pixels, 3)
        )
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            tasks = ((map_id, pixels) for map_id in ids)
            for index, (data, error) in enumerate(pool.map(read_map, tasks, chunksize=16)):
                if error is not None:
                    raise ValueError(error)
                arr[index] = data
                if (index + 1) % 2000 == 0:
                    print(f"  {split}: {index + 1}/{len(ids)}", flush=True)
        arr.flush()
        del arr
        arr = None
        np.save(temp_ids, np.asarray(ids, dtype=np.int32))
        # Publish finished caches. IDs and rows are always in identical order.
        temp_data.replace(data_path)
        temp_ids.replace(ids_path)
        print(f"  {split}: saved {len(ids)} maps -> {data_path}")
        print(f"         IDs -> {ids_path}")
    finally:
        if arr is not None:
            del arr
        temp_data.unlink(missing_ok=True)
        temp_ids.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--split", choices=VALID_SPLITS, help="Pack one split")
    selection.add_argument("--all", action="store_true", help="Pack all splits (default)")
    parser.add_argument("--shape", nargs=2, type=int, default=(128, 128), metavar=("NY", "NX"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cache files")
    args = parser.parse_args()
    if args.jobs < 1 or any(s < 1 for s in args.shape):
        parser.error("--jobs and --shape dimensions must be positive")

    selected = (args.split,) if args.split else VALID_SPLITS
    split_ids = {split: read_ids(split) for split in selected}
    # Check uniqueness across selected splits, in addition to split ID ranges.
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            overlap = set(split_ids[a]) & set(split_ids[b])
            if overlap:
                raise ValueError(f"Map IDs shared by {a} and {b}: {sorted(overlap)[:10]}")
    # Check output collisions before writing any split.
    if not args.overwrite:
        existing = [str(CACHE_DIR / f"split_{split}_clean{suffix}.npy")
                    for split in selected for suffix in ("", "_ids")
                    if (CACHE_DIR / f"split_{split}_clean{suffix}.npy").exists()]
        if existing:
            raise FileExistsError("Output cache already exists; use --overwrite to replace:\n" + "\n".join(existing))

    print(f"Clean Euler directory: {CLEAN_DIR}")
    print(f"Splits directory: {SPLITS_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    for split in selected:
        print(f"Packing {split}: {len(split_ids[split])} map IDs")
        pack_split(split, split_ids[split], args.shape[0] * args.shape[1], args.jobs, args.overwrite)
    print("Done. Only clean Euler angles were cached; no noise or quaternions.")


if __name__ == "__main__":
    main()
