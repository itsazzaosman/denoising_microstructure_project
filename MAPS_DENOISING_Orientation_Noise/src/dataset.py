#!/usr/bin/env python3
"""
Paired (noisy, clean) orientation maps.

Training noise is generated ON THE FLY rather than read from disk. There are
~34k clean maps and only 500 pre-noised ones, so re-corrupting a fresh sample
every epoch turns one pass over the data into effectively unlimited data, and
stops the network memorising one fixed corruption pattern. It also avoids
writing ~17 GB of noisy maps onto a volume that is already 80% full.

The TEST set does the opposite: it serves the exact pre-generated noisy maps
from datasets/noisy_mis05/, because those are the files MTEX was given. Re-
noising the test set would make the comparison with the filter baseline
meaningless.

Everything is returned as quaternions, (H, W, 4), plus the bad-pixel mask.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "datasets/cache"

# Matches how datasets/noisy_mis05 was generated: 0-1 deg scatter on every
# pixel, 5% of pixels replaced outright, 4x more likely on a grain boundary.
DEFAULT_NOISE = dict(scatter_deg=1.0, misindex=0.05, boundary_bias=4.0)


# ---------------------------------------------------------------------------
# numpy quaternion helpers (kept local so DataLoader workers import nothing heavy)
# ---------------------------------------------------------------------------


def euler_to_quat(e):
    phi1, Phi, phi2 = e[..., 0], e[..., 1], e[..., 2]
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack([c * np.cos(sigma), -s * np.cos(delta),
                  -s * np.sin(delta), -c * np.sin(sigma)], axis=-1)
    return np.where(q[..., :1] < 0, -q, q)


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def random_quats(n, rng):
    """Uniform random orientations (Shoemake)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    return np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=-1)


def small_rotations(n, max_deg, rng):
    """Random-axis rotations with angle uniform in [0, max_deg]."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.radians(rng.random(n) * max_deg) / 2.0
    return np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)


def boundary_of(q_grid):
    """(H, W, 4) clean quats -> (H, W) bool, True where a 4-neighbour differs."""
    same = lambda a, b: np.all(np.isclose(a, b, atol=1e-6), axis=-1)
    H, W, _ = q_grid.shape
    m = np.zeros((H, W), dtype=bool)
    dh = ~same(q_grid[:, :-1], q_grid[:, 1:])
    dv = ~same(q_grid[:-1, :], q_grid[1:, :])
    m[:, :-1] |= dh
    m[:, 1:] |= dh
    m[:-1, :] |= dv
    m[1:, :] |= dv
    return m


def corrupt(q_clean, shape, rng, scatter_deg=1.0, misindex=0.05, boundary_bias=4.0):
    """
    Apply the two-part EBSD noise model to a (N, 4) clean quaternion map.

    Returns (noisy quats, bad-pixel bool mask). Mirrors add_ebsd_noise.py, but
    works in quaternions end to end -- the .txt pipeline round-trips through
    Euler angles, which costs two conversions per map for nothing.
    """
    n = q_clean.shape[0]
    q = q_clean.copy()

    if scatter_deg > 0:
        q = quat_mul(small_rotations(n, scatter_deg, rng), q)

    bad = np.zeros(n, dtype=bool)
    n_bad = int(round(misindex * n))
    if n_bad > 0:
        if boundary_bias and boundary_bias > 1:
            on_bnd = boundary_of(q_clean.reshape(*shape, 4)).ravel()
            w = np.where(on_bnd, float(boundary_bias), 1.0)
            idx = rng.choice(n, size=n_bad, replace=False, p=w / w.sum())
        else:
            idx = rng.choice(n, size=n_bad, replace=False)
        q[idx] = random_quats(n_bad, rng)
        bad[idx] = True

    return q, bad


# ---------------------------------------------------------------------------


class TrainMaps(Dataset):
    """
    Clean maps from the cache, corrupted fresh on every access.

    Augmentation, both physically exact:

      * a global random rotation applied to every orientation in the map. The
        denoising task is exactly equivariant to this -- q -> q * g leaves every
        pairwise misorientation unchanged -- and it stops the network learning
        whatever orientation distribution DREAM.3D happened to emit.
        Note the side: crystal SYMMETRY acts on the left, so a sample rotation
        must act on the RIGHT or it would not commute with it.

      * flips and 90-degree rotations of the pixel grid, with the orientation
        values left alone. This rearranges grain shapes without inventing any
        new crystallography.

    split: "train"/"val" carve a deterministic hold-out from the cached maps,
    so the val maps are never trained on and stay the same across resumes.
    """

    def __init__(self, split="train", val_fraction=0.02, seed=42, augment=True,
                 shape=(128, 128), limit=0, noise=None, fixed_noise=False):
        self.shape = tuple(shape)
        self.augment = augment
        self.noise = dict(DEFAULT_NOISE if noise is None else noise)
        self.fixed_noise = fixed_noise

        path = CACHE / "train_clean.npy"
        if not path.exists():
            raise RuntimeError(f"{path} missing -- run prepare_cache.py first")
        self.maps = np.load(path, mmap_mode="r")
        ids = np.load(CACHE / "train_clean_ids.npy")
        n_total = len(ids)                       # never trust the array's shape

        order = np.random.default_rng(seed).permutation(n_total)
        n_val = max(1, int(round(n_total * val_fraction)))
        keep = order[:n_val] if split == "val" else order[n_val:]
        self.index = np.sort(keep)
        if limit:
            self.index = self.index[:limit]
        self.ids = ids[self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = int(self.index[i])
        e = np.asarray(self.maps[row], dtype=np.float64)
        q = euler_to_quat(e)

        # A fixed seed per map makes validation noise reproducible epoch to
        # epoch, so a moving val score means the MODEL moved, not the noise.
        rng = (np.random.default_rng(1_000_003 + row) if self.fixed_noise
               else np.random.default_rng())

        if self.augment:
            g = random_quats(1, rng)[0]
            q = quat_mul(q, g[None, :])          # sample rotation: RIGHT multiply

        noisy, bad = corrupt(q, self.shape, rng, **self.noise)

        H, W = self.shape
        q = q.reshape(H, W, 4)
        noisy = noisy.reshape(H, W, 4)
        bad = bad.reshape(H, W)

        if self.augment:
            if rng.random() < 0.5:
                q, noisy, bad = q[:, ::-1], noisy[:, ::-1], bad[:, ::-1]
            if rng.random() < 0.5:
                q, noisy, bad = q[::-1], noisy[::-1], bad[::-1]
            k = int(rng.integers(0, 4))
            if k:
                q = np.rot90(q, k, (0, 1))
                noisy = np.rot90(noisy, k, (0, 1))
                bad = np.rot90(bad, k, (0, 1))

        return (torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(bad)))


class TestMaps(Dataset):
    """
    The 500 maps MTEX was given, with their exact pre-generated noise.

    No augmentation and no re-noising: these are the held-out maps the baseline
    table is computed on, so the model must see byte-identical inputs.
    """

    def __init__(self, shape=(128, 128), limit=0):
        self.shape = tuple(shape)
        for name in ("test_clean.npy", "test_noisy.npy", "test_badmask.npy"):
            if not (CACHE / name).exists():
                raise RuntimeError(f"{CACHE/name} missing -- run prepare_cache.py first")
        self.clean = np.load(CACHE / "test_clean.npy", mmap_mode="r")
        self.noisy = np.load(CACHE / "test_noisy.npy", mmap_mode="r")
        self.bad = np.load(CACHE / "test_badmask.npy", mmap_mode="r")
        self.ids = np.load(CACHE / "test_clean_ids.npy")
        if limit:
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        H, W = self.shape
        q_clean = euler_to_quat(np.asarray(self.clean[i], dtype=np.float64)).reshape(H, W, 4)
        q_noisy = euler_to_quat(np.asarray(self.noisy[i], dtype=np.float64)).reshape(H, W, 4)
        bad = np.asarray(self.bad[i]).reshape(H, W)
        return (torch.from_numpy(np.ascontiguousarray(q_noisy, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(q_clean, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(bad)),
                int(self.ids[i]))
