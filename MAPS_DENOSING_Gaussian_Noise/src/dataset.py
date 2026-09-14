import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class EBSD_Dataset(Dataset):
    """Paired (noisy, clean) EBSD orientation maps.

    Filenames are shared between the two directories, so the clean listing
    drives the index and the noisy path is derived from it.

    split:
        "train" / "val" carve a deterministic hold-out out of the file list
        using `seed`, so the two splits never overlap and stay identical across
        runs and across resumes. "all" uses every file (what the old code did).

    augment:
        Random horizontal/vertical flips and 90-degree rotations, applied
        identically to both images of a pair. These are physically valid here:
        IPF-Z colouring depends on which crystal direction lies parallel to the
        sample Z axis, and none of these in-plane operations move Z, so every
        pixel keeps its original colour. Only the spatial arrangement changes.
        Should be off for validation.

    cache:
        Load every image into RAM once as uint8, removing the per-epoch PNG
        decode that can otherwise starve the GPU. Worth enabling if you see low
        GPU utilisation; costs about 3.9 GB for the full dataset, inside the
        32 GB the batch job asks for.

        Filled eagerly here in the parent process, into two preallocated
        contiguous arrays. That detail matters: DataLoader workers are forked,
        so they inherit these arrays copy-on-write and all share one copy. A
        lazily-filled per-worker dict would instead give each of the 8 workers
        its own full copy and blow straight past the memory limit.

    limit:
        Keep only the first N files of the split. Debug aid - applied before
        the cache is filled, so `--limit 32 --cache` stays instant.
    """

    def __init__(self, clean_dir, noisy_dir, split="all", val_fraction=0.05,
                 seed=42, augment=False, cache=False, limit=0):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.augment = augment
        self.cache = cache

        all_filenames = sorted(os.listdir(clean_dir))
        if not all_filenames:
            raise RuntimeError(f"no files found in {clean_dir}")

        if split == "all":
            self.image_filenames = all_filenames
        elif split in ("train", "val"):
            # Seeded permutation -> the same split every time, no state on disk.
            order = np.random.default_rng(seed).permutation(len(all_filenames))
            n_val = max(1, int(round(len(all_filenames) * val_fraction)))
            keep = order[:n_val] if split == "val" else order[n_val:]
            self.image_filenames = [all_filenames[i] for i in sorted(keep)]
        else:
            raise ValueError(f"split must be 'train', 'val' or 'all', got {split!r}")

        if limit:
            self.image_filenames = self.image_filenames[:limit]

        self._clean_cache = None
        self._noisy_cache = None
        if cache:
            self._build_cache(split)

    def _build_cache(self, split):
        n = len(self.image_filenames)
        probe = np.asarray(Image.open(
            os.path.join(self.clean_dir, self.image_filenames[0])
        ).convert("RGB"))
        h, w, c = probe.shape

        gib = 2 * n * h * w * c / 1024 ** 3
        print(f"Caching {n} {split} pairs into RAM ({gib:.1f} GiB)...", flush=True)

        self._clean_cache = np.empty((n, h, w, c), dtype=np.uint8)
        self._noisy_cache = np.empty((n, h, w, c), dtype=np.uint8)

        for i, name in enumerate(self.image_filenames):
            self._clean_cache[i] = np.asarray(
                Image.open(os.path.join(self.clean_dir, name)).convert("RGB"))
            self._noisy_cache[i] = np.asarray(
                Image.open(os.path.join(self.noisy_dir, name)).convert("RGB"))
            if (i + 1) % 5000 == 0:
                print(f"  cached {i+1}/{n}", flush=True)

        print(f"Cache ready ({split}).", flush=True)

    def __len__(self):
        return len(self.image_filenames)

    def _load(self, idx):
        if self._clean_cache is not None:
            return self._clean_cache[idx], self._noisy_cache[idx]

        img_name = self.image_filenames[idx]
        clean = np.asarray(
            Image.open(os.path.join(self.clean_dir, img_name)).convert("RGB")
        )
        noisy = np.asarray(
            Image.open(os.path.join(self.noisy_dir, img_name)).convert("RGB")
        )
        return clean, noisy

    def __getitem__(self, idx):
        clean, noisy = self._load(idx)

        # HWC uint8 -> CHW float in [0, 1]. Done by hand rather than with
        # transforms.ToTensor() so the cached uint8 arrays feed straight in.
        clean_tensor = torch.from_numpy(clean.copy()).permute(2, 0, 1).float().div_(255.0)
        noisy_tensor = torch.from_numpy(noisy.copy()).permute(2, 0, 1).float().div_(255.0)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                clean_tensor = torch.flip(clean_tensor, dims=[2])
                noisy_tensor = torch.flip(noisy_tensor, dims=[2])
            if torch.rand(1).item() < 0.5:
                clean_tensor = torch.flip(clean_tensor, dims=[1])
                noisy_tensor = torch.flip(noisy_tensor, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                clean_tensor = torch.rot90(clean_tensor, k, dims=[1, 2])
                noisy_tensor = torch.rot90(noisy_tensor, k, dims=[1, 2])

        return noisy_tensor.contiguous(), clean_tensor.contiguous()
