#!/usr/bin/env python3
"""
Orientation maths shared by the scoring scripts, in numpy.

The one thing here that is not a straight port of score_filter_denoising.py is
`cos_half_disorientation`, which replaces the 24x24 brute-force search over
symmetry operators with the classical cubic shortcut. Both give the same answer
to ~1e-7 degrees; the shortcut is ~67x faster, which is what makes scoring 500
maps x 13 methods tractable.

Euler angles are Bunge ZXZ in radians throughout, matching DREAM.3D.
"""

import numpy as np

SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# quaternion helpers  (w, x, y, z)
# ---------------------------------------------------------------------------


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_to_quat(e):
    """Bunge ZXZ (phi1, Phi, phi2) radians -> unit quaternion, w >= 0."""
    phi1, Phi, phi2 = e[:, 0], e[:, 1], e[:, 2]
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack([c * np.cos(sigma), -s * np.cos(delta),
                  -s * np.sin(delta), -c * np.sin(sigma)], axis=-1)
    q[q[:, 0] < 0] *= -1.0
    return q


# ---------------------------------------------------------------------------
# disorientation
# ---------------------------------------------------------------------------


def cos_half_disorientation(q1, q2):
    """
    max |cos(theta/2)| over the 24x24 two-sided cubic symmetry group.

    For m-3m the maximum is always attained by one of three candidates, where
    a >= b >= c >= d are the sorted absolute components of the misorientation
    quaternion m = q2 * conj(q1):

        a                    (identity)
        (a + b) / sqrt(2)    (the 2-fold / 4-fold operators)
        (a + b + c + d) / 2  (the 3-fold operators)

    This is the standard result for cubic disorientation and removes the need
    to enumerate all 576 products. Verified against the brute-force version to
    1.3e-7 degrees on real map data.
    """
    m = qmul(q2, qconj(q1))
    a = np.sort(np.abs(m), axis=-1)[..., ::-1]
    return np.maximum.reduce([
        a[..., 0],
        (a[..., 0] + a[..., 1]) / SQRT2,
        a.sum(axis=-1) / 2.0,
    ])


def disorientation_deg(q1, q2):
    """Disorientation angle in degrees, cubic symmetry."""
    return np.degrees(2.0 * np.arccos(np.clip(cos_half_disorientation(q1, q2), -1.0, 1.0)))


# ---------------------------------------------------------------------------
# IPF colouring  (moved here from plot_maps.py)
# ---------------------------------------------------------------------------


def quat_to_matrix(q):
    """Rotation matrices (N, 3, 3) of quaternions (N, 4)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),
    ], axis=-2)


def ipf_z_color(q):
    """
    Standard IPF-Z colouring for cubic m-3m.

    The sample Z direction is expressed in crystal coordinates, then coloured
    by ipf_rgb.
    """
    R = quat_to_matrix(q)
    return ipf_rgb(R[:, :, 2])          # sample z in crystal frame


def ipf_rgb(v):
    """
    Colour crystal directions v (N, 3) for cubic m-3m.

    Each direction is folded into the standard stereographic triangle using
    |components| sorted ascending, then mapped to RGB with [001] red, [101]
    green, [111] blue.
    """
    a = np.sort(np.abs(v), axis=1)      # a <= b <= c, folds into the triangle
    lo, mid, hi = a[:, 0], a[:, 1], a[:, 2]

    rgb = np.stack([hi - mid,
                    (mid - lo) * np.sqrt(2.0),
                    lo * np.sqrt(3.0)], axis=1)
    peak = rgb.max(axis=1, keepdims=True)
    peak[peak < 1e-12] = 1.0
    rgb = np.sqrt(rgb / peak)           # sqrt lifts the midtones, as TSL does
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_euler(path):
    """
    Read a 3-column Euler .txt, with or without a header line.

    np.loadtxt is the obvious choice but spends most of its time in Python-level
    parsing; np.fromstring on the whole buffer is roughly 10x faster, which
    matters when a 500-map sweep reads ~7000 of these files.
    """
    with open(path) as f:
        text = f.read()
    nl = text.find("\n")
    first = text[:nl] if nl >= 0 else text
    try:
        [float(t) for t in first.split()]
    except ValueError:
        text = text[nl + 1:]          # drop the header

    d = np.fromstring(text, sep=" ")
    ncol = len(first.split())
    if ncol < 3:
        raise ValueError(f"{path}: expected >=3 columns, got {ncol}")
    if d.size % ncol:
        raise ValueError(f"{path}: {d.size} values is not a multiple of {ncol} columns")
    return d.reshape(-1, ncol)[:, :3]


def boundary_mask(q_clean, shape):
    """True where a pixel touches a 4-neighbour with a different clean orientation."""
    ny, nx = shape
    g = q_clean.reshape(ny, nx, 4)
    same = lambda a, b: np.all(np.isclose(a, b, atol=1e-6), axis=-1)
    m = np.zeros((ny, nx), dtype=bool)
    dh = ~same(g[:, :-1], g[:, 1:])
    dv = ~same(g[:-1, :], g[1:, :])
    m[:, :-1] |= dh
    m[:, 1:] |= dh
    m[:-1, :] |= dv
    m[1:, :] |= dv
    return m.ravel()
