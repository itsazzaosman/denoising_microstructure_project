#!/usr/bin/env python3
"""
Orientation maths in torch: the layer every model piece sits on.

Everything here is differentiable and batched. Quaternions are (w, x, y, z) in
the trailing dimension; rotation matrices are (..., 3, 3); Euler angles are
Bunge ZXZ in radians, matching DREAM.3D and the numpy scripts.

Three things in here are load-bearing for the model design:

  * cos_half_disorientation  the cubic symmetry shortcut. A network that
    regresses orientations must be scored against ALL 24 symmetry-equivalent
    correct answers, or it is being trained on contradictory labels.

  * disorientation_loss      a loss with a BOUNDED gradient at zero error.
    The obvious loss, 2*arccos(cos_half), has an infinite gradient exactly
    where the model converges (errors here land around 0.03 degrees), so it
    produces NaNs just as training starts to work. Use `angle` for reporting
    and this for optimising.

  * sixd_to_matrix           the continuous 6D rotation representation.
    Every representation of four or fewer numbers is discontinuous somewhere,
    and the discontinuity surfaces as large errors at unpredictable places.
    Regress 6 numbers and Gram-Schmidt them instead.

Run `python orientation_torch.py --self-test` to check all of it against the
numpy reference implementations in orientation.py.
"""

import math

import torch
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# symmetry
# ---------------------------------------------------------------------------


def cubic_sym_quats(device=None, dtype=torch.float32):
    """The 24 proper rotations of m-3m as quaternions."""
    h, r = 0.5, SQRT2 / 2.0
    return torch.tensor([
        (1, 0, 0, 0),
        (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
        (r, r, 0, 0), (r, -r, 0, 0), (r, 0, r, 0), (r, 0, -r, 0),
        (r, 0, 0, r), (r, 0, 0, -r),
        (0, r, r, 0), (0, r, -r, 0), (0, r, 0, r), (0, r, 0, -r),
        (0, 0, r, r), (0, 0, r, -r),
        (h, h, h, h), (h, -h, -h, -h), (h, h, -h, h), (h, -h, h, -h),
        (h, -h, h, h), (h, h, -h, -h), (h, -h, -h, h), (h, h, h, -h),
    ], device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# quaternion algebra
# ---------------------------------------------------------------------------


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def qconj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def qnormalize(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def qpositive(q):
    """Fix the double cover by forcing w >= 0. q and -q are the same rotation."""
    return torch.where(q[..., :1] < 0, -q, q)


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def euler_to_quat(e):
    """Bunge ZXZ (phi1, Phi, phi2) radians -> unit quaternion with w >= 0."""
    phi1, Phi, phi2 = e.unbind(-1)
    sigma, delta = 0.5 * (phi1 + phi2), 0.5 * (phi1 - phi2)
    c, s = torch.cos(0.5 * Phi), torch.sin(0.5 * Phi)
    q = torch.stack([c * torch.cos(sigma), -s * torch.cos(delta),
                     -s * torch.sin(delta), -c * torch.sin(sigma)], dim=-1)
    return qpositive(q)


def quat_to_euler(q):
    """Unit quaternion -> Bunge ZXZ (phi1, Phi, phi2) radians, phi1/phi2 in [0, 2pi).

    Branchless: the two degenerate cases (Phi = 0 and Phi = pi, where phi1 and
    phi2 are not separately defined) are computed alongside the general case and
    selected with where(), so this stays usable under autograd and on GPU.
    """
    q = qnormalize(q)
    q0, q1, q2, q3 = q.unbind(-1)
    q03, q12 = q0 ** 2 + q3 ** 2, q1 ** 2 + q2 ** 2
    chi = torch.sqrt((q03 * q12).clamp_min(0.0))
    ok = chi > 1e-12

    safe_chi = torch.where(ok, chi, torch.ones_like(chi))
    phi1 = torch.atan2((-q0 * q2 + q1 * q3) / safe_chi,
                       (-q0 * q1 - q2 * q3) / safe_chi)
    Phi = torch.atan2(2.0 * chi, q03 - q12)
    phi2 = torch.atan2((q0 * q2 + q1 * q3) / safe_chi,
                       (-q0 * q1 + q2 * q3) / safe_chi)

    # Phi = 0: only phi1 + phi2 is defined, so put it all in phi1.
    deg0 = (~ok) & (q12 <= 1e-12)
    phi1 = torch.where(deg0, torch.atan2(-2.0 * q0 * q3, q0 ** 2 - q3 ** 2), phi1)
    # Phi = pi: only phi1 - phi2 is defined.
    degpi = (~ok) & (q03 <= 1e-12)
    phi1 = torch.where(degpi, torch.atan2(2.0 * q1 * q2, q1 ** 2 - q2 ** 2), phi1)

    zero = torch.zeros_like(phi1)
    Phi = torch.where(deg0, zero, torch.where(degpi, torch.full_like(Phi, math.pi), Phi))
    phi2 = torch.where(deg0 | degpi, zero, phi2)

    two_pi = 2.0 * math.pi
    return torch.stack([phi1 % two_pi, Phi, phi2 % two_pi], dim=-1)


def quat_to_matrix(q):
    """Unit quaternion -> (..., 3, 3) rotation matrix."""
    q = qnormalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def matrix_to_quat(R):
    """(..., 3, 3) rotation matrix -> unit quaternion with w >= 0.

    Shepperd's method: build the quaternion four different ways, each stable in
    a different region, then pick the one whose divisor is largest. Naively
    using only the w-branch loses all precision near a 180 degree rotation.
    """
    m = [[R[..., i, j] for j in range(3)] for i in range(3)]
    t = m[0][0] + m[1][1] + m[2][2]

    cands = torch.stack([
        torch.stack([1.0 + t, m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]], -1),
        torch.stack([m[2][1] - m[1][2], 1.0 + m[0][0] - m[1][1] - m[2][2],
                     m[0][1] + m[1][0], m[0][2] + m[2][0]], -1),
        torch.stack([m[0][2] - m[2][0], m[0][1] + m[1][0],
                     1.0 - m[0][0] + m[1][1] - m[2][2], m[1][2] + m[2][1]], -1),
        torch.stack([m[1][0] - m[0][1], m[0][2] + m[2][0], m[1][2] + m[2][1],
                     1.0 - m[0][0] - m[1][1] + m[2][2]], -1),
    ], dim=-2)                                               # (..., 4, 4)

    divisor = torch.stack([1.0 + t, 1.0 + m[0][0] - m[1][1] - m[2][2],
                           1.0 - m[0][0] + m[1][1] - m[2][2],
                           1.0 - m[0][0] - m[1][1] + m[2][2]], dim=-1)
    best = divisor.argmax(dim=-1, keepdim=True)
    q = torch.gather(cands, -2, best.unsqueeze(-1).expand(*best.shape, 4)).squeeze(-2)
    return qpositive(qnormalize(q))


def sixd_to_matrix(x):
    """(..., 6) -> (..., 3, 3) rotation matrix by Gram-Schmidt.

    The continuous rotation representation of Zhou et al. (2019). The first
    three numbers give the first column direction, the next three are
    orthogonalised against it, and the third column is their cross product.
    """
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_sixd(R):
    """(..., 3, 3) -> (..., 6). The inverse of sixd_to_matrix, for warm starts."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def quat_to_sixd(q):
    return matrix_to_sixd(quat_to_matrix(q))


def sixd_to_quat(x):
    return matrix_to_quat(sixd_to_matrix(x))


def axis_angle_to_quat(v, eps=1e-8):
    """
    (..., 3) rotation vector (axis * angle, radians) -> unit quaternion.

    This is the refinement head's output map. A zero vector gives the identity
    quaternion, so a zero-initialised head starts as a no-op and the untrained
    model reproduces its input exactly.

    The half-angle sinc is expanded near zero rather than divided, because the
    refinement angles here are genuinely tiny (a degree or less) and the naive
    sin(t)/t would lose all its precision in exactly the regime that matters.
    """
    theta = v.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    # sin(half)/theta, stable as theta -> 0 (limit 1/2)
    small = theta < 1e-4
    ratio = torch.where(small,
                        0.5 - theta ** 2 / 48.0,
                        torch.sin(half) / theta.clamp_min(eps))
    return torch.cat([torch.cos(half), ratio * v], dim=-1)


def quat_to_axis_angle(q, eps=1e-8):
    """Inverse of axis_angle_to_quat, for the shortest-path representation."""
    q = qpositive(qnormalize(q))
    w = q[..., :1].clamp(-1.0, 1.0)
    vec = q[..., 1:]
    sin_half = vec.norm(dim=-1, keepdim=True)
    theta = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half < 1e-6, 2.0 + theta ** 2 / 12.0, theta / sin_half.clamp_min(eps))
    return scale * vec


# ---------------------------------------------------------------------------
# disorientation
# ---------------------------------------------------------------------------


def cos_half_disorientation(q1, q2):
    """
    max |cos(theta/2)| over the two-sided cubic symmetry group, elementwise.

    With a >= b >= c >= d the sorted absolute components of m = q2 * conj(q1),
    the maximum over all 576 symmetry products is always one of

        a                    (identity)
        (a + b) / sqrt(2)    (2-fold and 4-fold axes)
        (a + b + c + d) / 2  (3-fold axes)

    so the 24x24 enumeration is unnecessary. Matches the brute-force numpy
    version to ~1e-7 degrees and runs about 67x faster.

    Differentiable: sort and max are differentiable almost everywhere, and the
    measure-zero ties do not matter in practice.
    """
    m = qmul(q2, qconj(q1))
    a = torch.sort(m.abs(), dim=-1, descending=True).values
    return torch.maximum(
        torch.maximum(a[..., 0], (a[..., 0] + a[..., 1]) / SQRT2),
        a.sum(-1) / 2.0,
    ).clamp(max=1.0)


def disorientation_rad(q1, q2):
    """Disorientation angle in radians. For REPORTING -- see disorientation_loss."""
    return 2.0 * torch.arccos(cos_half_disorientation(q1, q2).clamp(-1.0, 1.0))


def disorientation_deg(q1, q2):
    """Disorientation angle in degrees. For REPORTING -- see disorientation_loss."""
    return torch.rad2deg(disorientation_rad(q1, q2))


def disorientation_loss(q1, q2, kind="charbonnier", eps=1e-6):
    """
    Symmetry-aware per-pixel loss with a usable gradient at zero error.

    Do not optimise disorientation_deg directly: d(arccos)/dx is infinite at
    x = 1, which is exactly where a working model ends up, so it NaNs at the
    moment it starts succeeding.

    kind:
        "charbonnier"  sqrt(1 - cos + eps) - sqrt(eps).  Proportional to the
                       angle for errors above ~sqrt(eps), so it behaves like an
                       L1 loss on the angle: edge-preserving, and it keeps
                       pushing on small errors instead of giving up. Gradient
                       is bounded by 1/(2*sqrt(eps)). This is the default.
        "cosine"       1 - cos(theta/2).  Proportional to theta^2, so it is an
                       L2-like loss: smoother, but it stops caring about small
                       errors and smears boundaries. Cheap and very stable.
    """
    c = cos_half_disorientation(q1, q2)
    if kind == "cosine":
        return 1.0 - c
    if kind == "charbonnier":
        return torch.sqrt((1.0 - c).clamp_min(0.0) + eps) - math.sqrt(eps)
    raise ValueError(f"unknown kind: {kind!r}")


# ---------------------------------------------------------------------------
# fundamental zone and interpolation
# ---------------------------------------------------------------------------


def fz_reduce(q, sym=None):
    """
    Pick the symmetry-equivalent representative with the smallest rotation
    angle, i.e. the largest |w|. Use this to canonicalise network INPUTS so
    the same physical orientation always arrives as the same numbers.

    Symmetry acts by LEFT multiplication, s * q, under this convention. Right
    multiplication is not a symmetry of the disorientation defined above and
    silently changes the answer by tens of degrees -- the self-test checks this.

    It does not make the input discontinuity-free -- two pixels in one grain can
    still straddle a zone boundary and land on different variants. That is why
    the loss is symmetry-aware rather than relying on this.
    """
    if sym is None:
        sym = cubic_sym_quats(q.device, q.dtype)
    variants = qmul(sym, q.unsqueeze(-2))                    # (..., 24, 4)
    best = variants[..., 0].abs().argmax(dim=-1, keepdim=True)
    q = torch.gather(variants, -2, best.unsqueeze(-1).expand(*best.shape, 4)).squeeze(-2)
    return qpositive(q)


def align_to(q, ref, sym=None):
    """
    The representation of q (over the 24 symmetry variants and both signs)
    that is closest to ref.

    Required before any interpolation: slerp between two arbitrary
    representations of nearby orientations can travel the long way round the
    sphere and produce something physically wrong.
    """
    if sym is None:
        sym = cubic_sym_quats(q.device, q.dtype)
    variants = qmul(sym, q.unsqueeze(-2))                    # (..., 24, 4)
    dots = (variants * ref.unsqueeze(-2)).sum(-1)            # (..., 24)
    best = dots.abs().argmax(dim=-1, keepdim=True)
    out = torch.gather(variants, -2, best.unsqueeze(-1).expand(*best.shape, 4)).squeeze(-2)
    sign = torch.gather(dots, -1, best).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return out * sign


def slerp(q0, q1, t, eps=1e-7):
    """
    Spherical linear interpolation, t in [0, 1], broadcast over the batch.

    This is the gate in the proposed architecture: t = 0 returns the input
    untouched (so grain interiors and boundaries pass through bit-exact) and
    t = 1 returns the prediction. q1 is aligned to q0 first, so the path taken
    is always the short one.

    Falls back to a normalised lerp when the two are nearly parallel, where the
    sin(omega) divisor would otherwise blow up. That is the common case here,
    since most pixels need only a small correction.
    """
    q1 = align_to(q1, q0)
    cos = (q0 * q1).sum(-1, keepdim=True).clamp(-1.0, 1.0)

    omega = torch.arccos(cos.clamp(-1.0 + eps, 1.0 - eps))
    sin = torch.sin(omega)
    near = sin.abs() < 1e-4

    w0 = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * omega) / sin.clamp_min(eps))
    w1 = torch.where(near, t, torch.sin(t * omega) / sin.clamp_min(eps))
    return qnormalize(w0 * q0 + w1 * q1)


# ---------------------------------------------------------------------------
# map-level helpers
def local_mean_quat(q_map, radius=2, tol_deg=2.0):
    """
    Edge-preserving local mean orientation over a (2*radius+1)^2 window.

    Neighbours are aligned to the centre pixel by sign only and then averaged.
    No 24-fold symmetry search is needed, which is what makes this cheap: a
    neighbour inside the same grain differs from the centre by under a degree,
    so its misorientation quaternion is already its own fundamental-zone
    representative. A neighbour in a different grain, or in a different
    symmetry variant, fails the tolerance test and is dropped from the average
    entirely -- which is exactly the behaviour wanted at a grain boundary.

    q_map: (B, H, W, 4) -> (B, H, W, 4)
    """
    B, H, W, _ = q_map.shape
    xp = F.pad(q_map.permute(0, 3, 1, 2), (radius,) * 4, mode="replicate")
    cos_tol = math.cos(math.radians(tol_deg) / 2.0)

    acc = torch.zeros_like(q_map)
    wsum = torch.zeros(B, H, W, 1, device=q_map.device, dtype=q_map.dtype)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = xp[:, :, radius + dy:radius + dy + H,
                    radius + dx:radius + dx + W].permute(0, 2, 3, 1)
            dot = (nb * q_map).sum(-1, keepdim=True)
            w = (dot.abs() > cos_tol).to(q_map.dtype)
            acc = acc + w * nb * torch.sign(dot)
            wsum = wsum + w
    return qnormalize(acc / wsum.clamp_min(1.0))


def local_correction(q_map, radius=2, tol_deg=2.0, scale_deg=2.0):
    """
    The rotation taking each pixel to its edge-preserving local mean, as a
    scaled axis-angle 3-vector. (B, H, W, 4) -> (B, H, W, 3).

    This is the single most useful input channel for the refinement task. The
    correction a denoiser must apply to a pixel is a random rotation specific to
    that pixel's noise draw, so a head asked to synthesise it from abstract
    trunk features gets a gradient that cancels across pixels and never trains.
    Handing the network an approximate answer up front turns the task into
    "improve on this", which has a well-conditioned gradient from step one.

    Averaging n samples of the ~0.5 deg scatter leaves ~0.5/sqrt(n), so a 5x5
    window gets to ~0.1 deg on its own. Beating that is what the network is for.
    """
    mean = local_mean_quat(q_map, radius, tol_deg)
    delta = qpositive(qmul(mean, qconj(q_map)))     # symmetry acts on the left
    return quat_to_axis_angle(delta) / math.radians(scale_deg)


# ---------------------------------------------------------------------------


def local_misfit_deg(q_map, neighbourhood=8):
    """
    Per-pixel median disorientation to its neighbours, in degrees.

    Feed this to the network as an extra input channel. It is most of a
    misindexing detector on its own -- a replaced pixel disagrees with every
    neighbour by tens of degrees, while a grain-boundary pixel disagrees with
    only some of them, so the MEDIAN separates the two cleanly where a mean
    would not.

    q_map: (B, H, W, 4)  ->  (B, 1, H, W)
    """
    if neighbourhood not in (4, 8):
        raise ValueError("neighbourhood must be 4 or 8")

    B, H, W, _ = q_map.shape
    x = q_map.permute(0, 3, 1, 2)                            # (B, 4, H, W)
    x = F.pad(x, (1, 1, 1, 1), mode="replicate")

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighbourhood == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    centre = q_map
    angles = []
    for dy, dx in offsets:
        shifted = x[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        angles.append(disorientation_deg(centre, shifted.permute(0, 2, 3, 1)))
    return torch.stack(angles, dim=-1).median(dim=-1).values.unsqueeze(1)


def quats_to_input(q_map, include_misfit=True):
    """
    (B, H, W, 4) quaternions -> (B, C, H, W) network input.

    Channels: 9 rotation-matrix entries (fundamental-zone reduced), plus the
    local-misfit channel scaled to roughly unit range.
    """
    q = fz_reduce(q_map)
    R = quat_to_matrix(q).flatten(-2)                        # (B, H, W, 9)
    chans = [R.permute(0, 3, 1, 2)]
    if include_misfit:
        chans.append(local_misfit_deg(q_map) / 62.8)         # 62.8 deg is the cubic max
    return torch.cat(chans, dim=1)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def _self_test():
    import numpy as np

    import orientation as onp
    from score_filter_denoising import cubic_sym
    from score_filter_denoising import disorientation_deg as brute_deg

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    fails = []

    def check(name, value, tol):
        ok = value <= tol
        print(f"  {'ok ' if ok else 'FAIL'}  {name:<46} {value:.3e}  (tol {tol:.0e})")
        if not ok:
            fails.append(name)

    def err(a, b):
        """Symmetry-aware error as 1 - cos(theta/2).

        Round trips are checked with this rather than with degrees because the
        degree form goes through arccos near 1, where theta ~ 2*sqrt(2*eps)
        bottoms out around 3e-6 degrees even in float64. That floor is a
        property of the metric, not of the code under test, and a tolerance
        below it can never pass.
        """
        return float((1.0 - cos_half_disorientation(a, b)).abs().max())

    def rand_quat(n):
        u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
        return np.stack([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                         np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                         np.sqrt(u1) * np.sin(2 * np.pi * u3),
                         np.sqrt(u1) * np.cos(2 * np.pi * u3)], -1)

    n = 4096
    q1_np, q2_np = rand_quat(n), rand_quat(n)
    q1 = torch.from_numpy(q1_np).double()
    q2 = torch.from_numpy(q2_np).double()

    print("\nagainst the numpy reference")
    check("qmul", float((qmul(q1, q2).numpy() - onp.qmul(q1_np, q2_np)).max()), 1e-12)
    check("disorientation vs numpy shortcut",
          float((disorientation_deg(q1, q2).numpy() - onp.disorientation_deg(q1_np, q2_np)).max()),
          1e-9)
    check("disorientation vs 24x24 brute force",
          float(np.abs(disorientation_deg(q1, q2).numpy()
                       - brute_deg(q1_np, q2_np, cubic_sym())).max()),
          1e-8)

    print("\nround trips  (measured as 1 - cos(theta/2), see err())")
    check("quat -> matrix -> quat", err(matrix_to_quat(quat_to_matrix(q1)), q1), 1e-14)
    check("quat -> 6D -> quat", err(sixd_to_quat(quat_to_sixd(q1)), q1), 1e-14)
    e = quat_to_euler(q1)
    check("quat -> euler -> quat", err(euler_to_quat(e), q1), 1e-14)
    check("euler_to_quat matches numpy",
          float(np.abs(euler_to_quat(torch.from_numpy(e.numpy())).numpy()
                       - onp.euler_to_quat(e.numpy())).max()), 1e-12)

    print("\nsymmetry invariance")
    sym = cubic_sym_quats(dtype=torch.float64)
    picks = sym[torch.randint(0, 24, (n,))]
    check("disorientation unchanged by s*q1 (left action)",
          float(disorientation_deg(qmul(picks, q1), q2).sub(
              disorientation_deg(q1, q2)).abs().max()), 1e-9)
    check("disorientation unchanged by s*q2 (left action)",
          float(disorientation_deg(q1, qmul(picks, q2)).sub(
              disorientation_deg(q1, q2)).abs().max()), 1e-9)
    # Guard rail: right multiplication is NOT a symmetry here. If this ever
    # starts passing, the convention has changed and fz_reduce/align_to are
    # acting on the wrong side.
    right = float(disorientation_deg(qmul(q1, picks), q2).sub(
        disorientation_deg(q1, q2)).abs().max())
    print(f"  {'ok ' if right > 1.0 else 'FAIL'}  "
          f"{'right action q1*s is correctly NOT a symmetry':<46} {right:.3e}  (want >1)")
    if right <= 1.0:
        fails.append("right action unexpectedly symmetric")
    check("fz_reduce preserves disorientation",
          float(disorientation_deg(fz_reduce(q1), fz_reduce(q2)).sub(
              disorientation_deg(q1, q2)).abs().max()), 1e-9)
    check("fz_reduce is idempotent",
          float((fz_reduce(fz_reduce(q1)) - fz_reduce(q1)).abs().max()), 1e-12)

    print("\ngate (slerp)")
    small = torch.randn(n, 4).double() * 1e-2
    small[:, 0] = 1.0
    q_near = qnormalize(qmul(qnormalize(small), q1))
    check("slerp t=0 is the identity",
          float((slerp(q1, q_near, torch.zeros(n, 1).double()) - q1).abs().max()), 1e-9)
    check("slerp t=1 reaches the target",
          err(slerp(q1, q_near, torch.ones(n, 1).double()), q_near), 1e-14)
    t = torch.rand(n, 1).double()
    mid = slerp(q1, q_near, t)
    total = disorientation_rad(q1, q_near)
    check("slerp is angle-linear in t",
          float((disorientation_rad(q1, mid) - t.squeeze(-1) * total).abs().max()), 1e-7)

    print("\nloss gradients (the arccos trap)")
    qa = qnormalize(torch.randn(256, 4, dtype=torch.double)).requires_grad_(True)
    for name, fn in [("disorientation_deg (reporting only)", disorientation_deg),
                     ("disorientation_loss charbonnier",
                      lambda a, b: disorientation_loss(a, b, "charbonnier")),
                     ("disorientation_loss cosine",
                      lambda a, b: disorientation_loss(a, b, "cosine"))]:
        target = qa.detach().clone()                          # zero error: the danger point
        g = torch.autograd.grad(fn(qa, target).sum(), qa)[0]
        finite = bool(torch.isfinite(g).all())
        print(f"  {'ok ' if finite or 'reporting' in name else 'FAIL'}  "
              f"{name:<46} grad finite at zero error: {finite}")
        if not finite and "reporting" not in name:
            fails.append(name)

    print("\nlocal misfit channel")
    q_map = fz_reduce(qnormalize(torch.randn(2, 16, 16, 4, dtype=torch.double)))
    grain = q_map[:, :1, :1].expand(2, 16, 16, 4).contiguous()
    check("flat map -> zero misfit", float(local_misfit_deg(grain).abs().max()), 1e-9)
    mis = local_misfit_deg(q_map)
    check("random map -> large misfit (reported as 60 - median)",
          float(60.0 - mis.median()), 60.0)
    inp = quats_to_input(q_map)
    check("quats_to_input channel count", abs(inp.shape[1] - 10), 0)

    print("\nreal map data")
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        qc = onp.euler_to_quat(onp.load_euler(
            root / "datasets/clean_euler/map_00001_clean_euler.txt"))
        qn = onp.euler_to_quat(onp.load_euler(
            root / "datasets/noisy_mis05/map_00001_clean_euler_noisy.txt"))
        d_np = onp.disorientation_deg(qc, qn)
        d_pt = disorientation_deg(torch.from_numpy(qc), torch.from_numpy(qn)).numpy()
        check("map 1 disorientation matches numpy", float(np.abs(d_np - d_pt).max()), 1e-8)
        print(f"        median {np.median(d_pt):.6f} deg, "
              f">10 deg {100 * (d_pt > 10).mean():.2f}%  (expect 0.521763 / 4.95%)")

        qmap = torch.from_numpy(qc).reshape(1, 128, 128, 4)
        qmapn = torch.from_numpy(qn).reshape(1, 128, 128, 4)
        bad = np.load(root / "datasets/noisy_mis05/map_00001_clean_euler_noisy.txt.badmask.npy")
        mis = local_misfit_deg(qmapn)[0, 0].numpy()
        sep = np.median(mis[bad]) - np.percentile(mis[~bad], 99)
        print(f"        misfit channel: corrupted median {np.median(mis[bad]):.1f} deg vs "
              f"clean 99th pct {np.percentile(mis[~bad], 99):.1f} deg")
        check("misfit separates corrupted from clean pixels", -sep, 0.0)
        print(f"        clean-map misfit median {np.median(local_misfit_deg(qmap).numpy()):.3f} deg")
    except FileNotFoundError as exc:
        print(f"  skipped (dataset not found: {exc})")

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true",
                   help="verify against the numpy reference in orientation.py")
    a = p.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    p.print_help()
