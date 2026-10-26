#!/usr/bin/env python3
"""On-the-fly EBSD orientation dataset for training and validation ONLY.

Clean caches (created by prepare_cache.py):
  datasets/cache/split_train_clean.npy + split_train_clean_ids.npy
  datasets/cache/split_val_clean.npy   + split_val_clean_ids.npy

Per access, sample scatter_max ~ Uniform(1,15) degrees and
misindex_fraction ~ Uniform(0.01,0.20). Then generate new noise per pixel.
TrainMaps includes orientation + spatial augmentation and fresh noise on every
access; ValMaps has no augmentation and one fixed noise draw per map (seeded by
its map ID), reused every epoch.
Return (noisy_quaternions, clean_quaternions, misindexed_mask) as tensors:
  (H,W,4), (H,W,4), (H,W).
Euler input: Bunge ZXZ radians. Quaternion order: (w,x,y,z).
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'datasets' / 'cache'

# Default noise sampling: continuous, per map access.
SCATTER_RANGE_DEG = (1.0, 15.0)
MISINDEX_RANGE = (0.01, 0.20)
      

def euler_to_quat(e):
    """Bunge Euler radians -> quaternion (w,x,y,z); same convention as old loader."""
    phi1, Phi, phi2 = e[..., 0], e[..., 1], e[..., 2]
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack((c * np.cos(sigma), -s * np.cos(delta),
                  -s * np.sin(delta), -c * np.sin(sigma)), axis=-1)
    return np.where(q[..., :1] < 0, -q, q)


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def random_quats(n, rng):
    """Uniform SO(3) quaternions (Shoemake), scalar-first order."""
    u1, u2, u3 = rng.random((3, n))
    return np.stack((np.sqrt(u1)*np.cos(2*np.pi*u3),
                     np.sqrt(1-u1)*np.sin(2*np.pi*u2),
                     np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                     np.sqrt(u1)*np.sin(2*np.pi*u3)), axis=-1)


def small_rotations(n, scatter_deg, rng):
    axes = rng.normal(size=(n, 3))
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-15)
    half = np.deg2rad(rng.uniform(0.0, scatter_deg, size=n)) * 0.5
    return np.concatenate((np.cos(half)[:, None], axes*np.sin(half)[:, None]), axis=-1)


def cubic_disorientation_deg(q1, q2):
    """Cubic m-3m disorientation, matching orientation_torch shortcut.

    Assumes scalar-first unit quaternions. Works on arbitrary broadcastable
    arrays with final dimension 4.
    """
    conjugate = np.array([1., -1., -1., -1.])
    m = quat_mul(q2, q1 * conjugate)
    a = np.sort(np.abs(m), axis=-1)[..., ::-1]
    c = np.maximum.reduce((a[..., 0], (a[..., 0] + a[..., 1]) / np.sqrt(2),
                           a.sum(axis=-1) / 2))
    return np.rad2deg(2 * np.arccos(np.clip(c, -1, 1)))


def boundary_of(q_grid, tol_deg=0.1):
    """Four-neighbor, cubic-symmetry-aware boundary pixels on CLEAN maps."""
    H, W, _ = q_grid.shape
    out = np.zeros((H, W), dtype=bool)
    horizontal = cubic_disorientation_deg(q_grid[:, :-1], q_grid[:, 1:]) > tol_deg
    vertical = cubic_disorientation_deg(q_grid[:-1, :], q_grid[1:, :]) > tol_deg
    out[:, :-1] |= horizontal
    out[:, 1:] |= horizontal
    out[:-1, :] |= vertical
    out[1:, :] |= vertical
    return out


def corrupt(q_clean, shape, rng, scatter_deg, misindex, boundary_bias=4.0):
    """Scatter every pixel, then replace selected pixels with random orientations.

    misindex is a fraction in [0, 1], not a percentage.
    """
    n = q_clean.shape[0]
    noisy = quat_mul(small_rotations(n, scatter_deg, rng), q_clean)
    bad = np.zeros(n, dtype=bool)
    n_bad = int(round(misindex * n))
    if n_bad:
        if boundary_bias > 1:
            bnd = boundary_of(q_clean.reshape(*shape, 4)).reshape(-1)
            weights = np.where(bnd, boundary_bias, 1.0).astype(np.float64)
            indices = rng.choice(n, size=n_bad, replace=False, p=weights/weights.sum())
        else:
            indices = rng.choice(n, size=n_bad, replace=False)
        noisy[indices] = random_quats(n_bad, rng)
        bad[indices] = True
    return noisy, bad

class _CleanMaps(Dataset):
    """Shared clean-cache reader and on-the-fly noise generator.

    Each access samples independently:
      scatter maximum in [scatter_min, scatter_max] degrees;
      misindexing fraction in [misindex_min, misindex_max].

    These are continuous uniform distributions. Each pixel then receives its own
    scatter rotation with an angle in [0, selected scatter maximum].
    """

    def __init__(self, *, split, augment, shape=(128, 128),
                 scatter_min=1.0, scatter_max=15.0,
                 misindex_min=0.01, misindex_max=0.20,
                 boundary_bias=4.0, limit=0, cache_dir=None, fixed_noise=False):
        self.split = split
        self.augment = augment
        self.shape = tuple(shape)
        self.scatter_min, self.scatter_max = float(scatter_min), float(scatter_max)
        self.misindex_min, self.misindex_max = float(misindex_min), float(misindex_max)
        self.boundary_bias = float(boundary_bias)
        # True: one fixed random draw per map, seeded by its map ID, reused every
        # epoch (validation). False: a fresh draw on every access (training).
        self.fixed_noise = bool(fixed_noise)
        if not (0 <= self.scatter_min <= self.scatter_max and
                np.isfinite(self.scatter_min) and np.isfinite(self.scatter_max)):
            raise ValueError('Expected 0 <= scatter_min <= scatter_max (finite degrees)')
        if not (0 <= self.misindex_min <= self.misindex_max <= 1 and
                np.isfinite(self.misindex_min) and np.isfinite(self.misindex_max)):
            raise ValueError('Expected 0 <= misindex_min <= misindex_max <= 1 (fractions)')
        if not (np.isfinite(self.boundary_bias) and self.boundary_bias >= 1):
            raise ValueError('boundary_bias must be finite and >= 1')
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError('shape must be (H, W) with positive dimensions')
        if limit < 0:
            raise ValueError('limit must be >= 0')

        cache = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        stem = f'split_{split}_clean'
        maps_path = cache / f'{stem}.npy'
        ids_path = cache / f'{stem}_ids.npy'
        if not maps_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f'Missing {maps_path} or {ids_path}; run prepare_cache.py')
        self.maps = np.load(maps_path, mmap_mode='r')
        ids = np.load(ids_path)
        n_pixels = int(np.prod(self.shape))
        if self.maps.ndim != 3 or self.maps.shape[1:] != (n_pixels, 3):
            raise ValueError(f'Unexpected cache shape: {self.maps.shape}, expected (N,{n_pixels},3)')
        if len(self.maps) != len(ids):
            raise ValueError(f'Cache/ID count mismatch for {stem}')
        if len(set(map(int, ids))) != len(ids):
            raise ValueError(f'Duplicate map IDs in {ids_path}')
        self.ids = ids[:limit] if limit else ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.fixed_noise:
            rng = np.random.default_rng(np.random.SeedSequence([2026, int(self.ids[index])]))
        else:
            # Each DataLoader worker receives a reproducible PyTorch worker seed.
            # Reuse a generator within the worker so consecutive maps differ.
            if not hasattr(self, "_rng"):
                self._rng = np.random.default_rng(torch.initial_seed())
            rng = self._rng
        euler = np.asarray(self.maps[index], dtype=np.float64)
        if not np.all(np.isfinite(euler)):
            raise ValueError(f'Non-finite Euler angles in map ID {self.ids[index]}')
        clean = euler_to_quat(euler)

        if self.augment:
            # Apply the same random global orientation change to the whole clean map.
            # RIGHT multiplication follows the convention in the original project.
            global_rotation = random_quats(1, rng)[0]
            clean = quat_mul(clean, global_rotation)

        scatter_deg = rng.uniform(self.scatter_min, self.scatter_max)
        misindex_fraction = rng.uniform(self.misindex_min, self.misindex_max)
        noisy, bad = corrupt(clean, self.shape, rng,
                             scatter_deg=scatter_deg,
                             misindex=misindex_fraction,
                             boundary_bias=self.boundary_bias)

        H, W = self.shape
        clean = clean.reshape(H, W, 4)
        noisy = noisy.reshape(H, W, 4)
        bad = bad.reshape(H, W)

        if self.augment:
            # Identical spatial transformations for input, target and bad-pixel mask.
            if rng.random() < 0.5:
                clean, noisy, bad = clean[:, ::-1], noisy[:, ::-1], bad[:, ::-1]
            if rng.random() < 0.5:
                clean, noisy, bad = clean[::-1], noisy[::-1], bad[::-1]
            k = int(rng.integers(0, 4))
            if k:
                clean = np.rot90(clean, k, axes=(0, 1))
                noisy = np.rot90(noisy, k, axes=(0, 1))
                bad = np.rot90(bad, k, axes=(0, 1))

        return (torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(clean, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(bad, dtype=np.bool_)))


class TrainMaps(_CleanMaps):
    """Training split: new continuous noise every access, WITH augmentation."""

    def __init__(self, *, shape=(128, 128), scatter_min=1.0, scatter_max=15.0,
                 misindex_min=0.01, misindex_max=0.20,
                 boundary_bias=4.0, limit=0, cache_dir=None):
        super().__init__(split='train', augment=True, shape=shape,
                         scatter_min=scatter_min, scatter_max=scatter_max,
                         misindex_min=misindex_min, misindex_max=misindex_max,
                         boundary_bias=boundary_bias, limit=limit, cache_dir=cache_dir)


class ValMaps(_CleanMaps):
    """Validation split: one fixed noise draw per map (seeded by its ID, the same
    every epoch), NO augmentation."""

    def __init__(self, *, shape=(128, 128), scatter_min=1.0, scatter_max=15.0,
                 misindex_min=0.01, misindex_max=0.20,
                 boundary_bias=4.0, limit=0, cache_dir=None):
        super().__init__(split='val', augment=False, fixed_noise=True, shape=shape,
                         scatter_min=scatter_min, scatter_max=scatter_max,
                         misindex_min=misindex_min, misindex_max=misindex_max,
                         boundary_bias=boundary_bias, limit=limit, cache_dir=cache_dir)




# #!/usr/bin/env python3
# """On-the-fly EBSD orientation dataset for training and validation ONLY.

# Clean caches (created by prepare_cache.py):
#   datasets/cache/split_train_clean.npy + split_train_clean_ids.npy
#   datasets/cache/split_val_clean.npy   + split_val_clean_ids.npy

# Per access, sample scatter_max ~ Uniform(1,15) degrees and
# misindex_fraction ~ Uniform(0.01,0.20). Then generate new noise per pixel.
# TrainMaps includes orientation + spatial augmentation and fresh noise on every
# access; ValMaps has no augmentation and one fixed noise draw per map (seeded by
# its map ID), reused every epoch.
# Return (noisy_quaternions, clean_quaternions, misindexed_mask) as tensors:
#   (H,W,4), (H,W,4), (H,W).
# Euler input: Bunge ZXZ radians. Quaternion order: (w,x,y,z).
# """
# from pathlib import Path
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# ROOT = Path(__file__).resolve().parents[2]
# CACHE_DIR = ROOT / 'datasets' / 'cache'

# # Default noise sampling: continuous, per map access.
# SCATTER_RANGE_DEG = (1.0, 15.0)
# MISINDEX_RANGE = (0.01, 0.20)
      

# def euler_to_quat(e):
#     """Bunge Euler radians -> quaternion (w,x,y,z); same convention as old loader."""
#     phi1, Phi, phi2 = e[..., 0], e[..., 1], e[..., 2]
#     sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
#     c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
#     q = np.stack((c * np.cos(sigma), -s * np.cos(delta),
#                   -s * np.sin(delta), -c * np.sin(sigma)), axis=-1)
#     return np.where(q[..., :1] < 0, -q, q)


# def quat_mul(a, b):
#     aw, ax, ay, az = np.moveaxis(a, -1, 0)
#     bw, bx, by, bz = np.moveaxis(b, -1, 0)
#     return np.stack((aw*bw-ax*bx-ay*by-az*bz,
#                      aw*bx+ax*bw+ay*bz-az*by,
#                      aw*by-ax*bz+ay*bw+az*bx,
#                      aw*bz+ax*by-ay*bx+az*bw), axis=-1)


# def random_quats(n, rng):
#     """Uniform SO(3) quaternions, scalar-first (w, x, y, z)."""
#     u1, u2, u3 = rng.random((3, n))

#     return np.stack((
#         np.sqrt(u1) * np.cos(2 * np.pi * u3),
#         np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
#         np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
#         np.sqrt(u1) * np.sin(2 * np.pi * u3),
#     ), axis=-1)


# def small_rotations(n, scatter_deg, rng):
#     axes = rng.normal(size=(n, 3))
#     axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-15)
#     half = np.deg2rad(rng.uniform(0.0, scatter_deg, size=n)) * 0.5
#     return np.concatenate((np.cos(half)[:, None], axes*np.sin(half)[:, None]), axis=-1)


# def boundary_of(q_grid, atol=1e-6):
#     """4-neighbor differences on piecewise-constant synthetic maps.

#     Checks q vs +/-q; NOT a cubic-symmetry-aware misorientation threshold.
#     """
#     def same(a, b):
#         return np.all(np.isclose(a, b, atol=atol, rtol=0), axis=-1) | \
#                np.all(np.isclose(a, -b, atol=atol, rtol=0), axis=-1)
#     H, W, _ = q_grid.shape
#     out = np.zeros((H, W), dtype=bool)
#     horizontal = ~same(q_grid[:, :-1], q_grid[:, 1:])
#     vertical = ~same(q_grid[:-1, :], q_grid[1:, :])
#     out[:, :-1] |= horizontal
#     out[:, 1:] |= horizontal
#     out[:-1, :] |= vertical
#     out[1:, :] |= vertical
#     return out


# def corrupt(q_clean, shape, rng, scatter_deg, misindex, boundary_bias=4.0):
#     """Scatter every pixel, then replace selected pixels with random orientations.

#     misindex is a fraction in [0, 1], not a percentage.
#     """
#     n = q_clean.shape[0]
#     noisy = quat_mul(small_rotations(n, scatter_deg, rng), q_clean)
#     bad = np.zeros(n, dtype=bool)
#     n_bad = int(round(misindex * n))
#     if n_bad:
#         if boundary_bias > 1:
#             bnd = boundary_of(q_clean.reshape(*shape, 4)).reshape(-1)
#             weights = np.where(bnd, boundary_bias, 1.0).astype(np.float64)
#             indices = rng.choice(n, size=n_bad, replace=False, p=weights/weights.sum())
#         else:
#             indices = rng.choice(n, size=n_bad, replace=False)
#         noisy[indices] = random_quats(n_bad, rng)
#         bad[indices] = True
#     return noisy, bad

# class _CleanMaps(Dataset):
#     """Shared clean-cache reader and on-the-fly noise generator.

#     Each access samples independently:
#       scatter maximum in [scatter_min, scatter_max] degrees;
#       misindexing fraction in [misindex_min, misindex_max].

#     These are continuous uniform distributions. Each pixel then receives its own
#     scatter rotation with an angle in [0, selected scatter maximum].
#     """

#     def __init__(self, *, split, augment, shape=(128, 128),
#                  scatter_min=1.0, scatter_max=15.0,
#                  misindex_min=0.01, misindex_max=0.20,
#                  boundary_bias=4.0, limit=0, cache_dir=None, fixed_noise=False):
#         self.split = split
#         self.augment = augment
#         self.shape = tuple(shape)
#         self.scatter_min, self.scatter_max = float(scatter_min), float(scatter_max)
#         self.misindex_min, self.misindex_max = float(misindex_min), float(misindex_max)
#         self.boundary_bias = float(boundary_bias)
#         # True: one fixed random draw per map, seeded by its map ID, reused every
#         # epoch (validation). False: a fresh draw on every access (training).
#         self.fixed_noise = bool(fixed_noise)
#         if not (0 <= self.scatter_min <= self.scatter_max and
#                 np.isfinite(self.scatter_min) and np.isfinite(self.scatter_max)):
#             raise ValueError('Expected 0 <= scatter_min <= scatter_max (finite degrees)')
#         if not (0 <= self.misindex_min <= self.misindex_max <= 1 and
#                 np.isfinite(self.misindex_min) and np.isfinite(self.misindex_max)):
#             raise ValueError('Expected 0 <= misindex_min <= misindex_max <= 1 (fractions)')
#         if not (np.isfinite(self.boundary_bias) and self.boundary_bias >= 1):
#             raise ValueError('boundary_bias must be finite and >= 1')
#         if len(self.shape) != 2 or min(self.shape) < 1:
#             raise ValueError('shape must be (H, W) with positive dimensions')
#         if limit < 0:
#             raise ValueError('limit must be >= 0')

#         cache = Path(cache_dir) if cache_dir is not None else CACHE_DIR
#         stem = f'split_{split}_clean'
#         maps_path = cache / f'{stem}.npy'
#         ids_path = cache / f'{stem}_ids.npy'
#         if not maps_path.is_file() or not ids_path.is_file():
#             raise FileNotFoundError(f'Missing {maps_path} or {ids_path}; run prepare_cache.py')
#         self.maps = np.load(maps_path, mmap_mode='r')
#         ids = np.load(ids_path)
#         n_pixels = int(np.prod(self.shape))
#         if self.maps.ndim != 3 or self.maps.shape[1:] != (n_pixels, 3):
#             raise ValueError(f'Unexpected cache shape: {self.maps.shape}, expected (N,{n_pixels},3)')
#         if len(self.maps) != len(ids):
#             raise ValueError(f'Cache/ID count mismatch for {stem}')
#         if len(set(map(int, ids))) != len(ids):
#             raise ValueError(f'Duplicate map IDs in {ids_path}')
#         self.ids = ids[:limit] if limit else ids

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         rng = np.random.default_rng(int(self.ids[index]) if self.fixed_noise else None)
#         euler = np.asarray(self.maps[index], dtype=np.float64)
#         if not np.all(np.isfinite(euler)):
#             raise ValueError(f'Non-finite Euler angles in map ID {self.ids[index]}')
#         clean = euler_to_quat(euler)

#         if self.augment:
#             # Apply the same random global orientation change to the whole clean map.
#             # RIGHT multiplication follows the convention in the original project.
#             global_rotation = random_quats(1, rng)[0]
#             clean = quat_mul(clean, global_rotation)

#         scatter_deg = rng.uniform(self.scatter_min, self.scatter_max)
#         misindex_fraction = rng.uniform(self.misindex_min, self.misindex_max)
#         noisy, bad = corrupt(clean, self.shape, rng,
#                              scatter_deg=scatter_deg,
#                              misindex=misindex_fraction,
#                              boundary_bias=self.boundary_bias)

#         H, W = self.shape
#         clean = clean.reshape(H, W, 4)
#         noisy = noisy.reshape(H, W, 4)
#         bad = bad.reshape(H, W)

#         if self.augment:
#             # Identical spatial transformations for input, target and bad-pixel mask.
#             if rng.random() < 0.5:
#                 clean, noisy, bad = clean[:, ::-1], noisy[:, ::-1], bad[:, ::-1]
#             if rng.random() < 0.5:
#                 clean, noisy, bad = clean[::-1], noisy[::-1], bad[::-1]
#             k = int(rng.integers(0, 4))
#             if k:
#                 clean = np.rot90(clean, k, axes=(0, 1))
#                 noisy = np.rot90(noisy, k, axes=(0, 1))
#                 bad = np.rot90(bad, k, axes=(0, 1))

#         return (torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32)),
#                 torch.from_numpy(np.ascontiguousarray(clean, dtype=np.float32)),
#                 torch.from_numpy(np.ascontiguousarray(bad, dtype=np.bool_)))


# class TrainMaps(_CleanMaps):
#     """Training split: new continuous noise every access, WITH augmentation."""

#     def __init__(self, *, shape=(128, 128), scatter_min=1.0, scatter_max=15.0,
#                  misindex_min=0.01, misindex_max=0.20,
#                  boundary_bias=4.0, limit=0, cache_dir=None):
#         super().__init__(split='train', augment=True, shape=shape,
#                          scatter_min=scatter_min, scatter_max=scatter_max,
#                          misindex_min=misindex_min, misindex_max=misindex_max,
#                          boundary_bias=boundary_bias, limit=limit, cache_dir=cache_dir)


# class ValMaps(_CleanMaps):
#     """Validation split: one fixed noise draw per map (seeded by its ID, the same
#     every epoch), NO augmentation."""

#     def __init__(self, *, shape=(128, 128), scatter_min=1.0, scatter_max=15.0,
#                  misindex_min=0.01, misindex_max=0.20,
#                  boundary_bias=4.0, limit=0, cache_dir=None):
#         super().__init__(split='val', augment=False, fixed_noise=True, shape=shape,
#                          scatter_min=scatter_min, scatter_max=scatter_max,
#                          misindex_min=misindex_min, misindex_max=misindex_max,
#                          boundary_bias=boundary_bias, limit=limit, cache_dir=cache_dir)
