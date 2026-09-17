#!/usr/bin/env python3
"""
Gated U-Net denoiser for EBSD orientation maps.

The design follows from the shape of the noise. 95% of pixels carry a
sub-degree wobble and need a small refinement; 5% have been replaced outright
and need a new orientation entirely. One regression head serves neither well,
so the network has three:

    refine    a bounded small rotation applied to the input  (handles scatter)
    replace   a full orientation, from scratch               (handles misindexing)
    gate      how much to trust the input at this pixel      (chooses between them)

    q_out = slerp(refine(q_in), replace(x), sigmoid(gate))

Two properties fall out of this that a plain regression U-Net does not have:

  * Where the gate is closed the output IS the input, bit for bit. Grain
    interiors and boundaries survive untouched, so the model cannot win the
    ALL column by quietly blurring the BOUNDARY column -- the usual failure of
    a smoothing filter. The refinement is additionally capped at a few degrees,
    so even an open refine path cannot drag a pixel across a boundary.

  * The untrained model is the identity. Epoch 0 scores the noisy input rather
    than noise, so the first validation number is meaningful and any
    improvement is unambiguously the model's doing.

Orientations are regressed in the continuous 6D representation: every
representation of four or fewer numbers is discontinuous somewhere, and the
discontinuity shows up as large errors at unpredictable places.

The gate is supervised directly from the bad-pixel mask the noise generator
records, which is the one thing this model knows that the MTEX pre-cleaning
step has to guess.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from orientation_torch import (axis_angle_to_quat, disorientation_loss,
                               fz_reduce, local_correction, local_misfit_deg,
                               qmul, qnormalize, quat_to_matrix, sixd_to_quat,
                               slerp)


# Degrees that the local-correction channel is normalised by, so it lands in
# roughly [-1, 1]. Also used to undo that scaling in forward().
CORR_SCALE_DEG = 2.0


def _norm(channels, groups=8):
    """GroupNorm, so behaviour does not change when the batch size does."""
    return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)


class DoubleConv(nn.Module):
    """(conv -> norm -> SiLU) x 2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample + conv, not ConvTranspose2d, which checkerboards
    visibly against the flat grain interiors."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class GatedOrientationUNet(nn.Module):
    """
    (B, H, W, 4) noisy quaternions -> dict of (B, H, W, ...) predictions.

    max_refine_deg caps the refinement path. The scatter noise is under a
    degree, so 5 is generous; the point of the cap is that no amount of
    training can turn the refine path into a smoother.
    """

    def __init__(self, base=64, max_refine_deg=5.0, gate_bias=-4.0, use_misfit=True,
                 gate_deadzone=0.05, mean_radius=4):
        super().__init__()
        self.use_misfit = use_misfit
        self.max_refine = math.radians(max_refine_deg)
        self.gate_deadzone = gate_deadzone
        self.mean_radius = mean_radius

        in_ch = 9 + (3 if mean_radius else 0) + (1 if use_misfit else 0)
        self.inc = DoubleConv(in_ch, base)                 # 128
        self.down1 = Down(base, base * 2)                  # 64
        self.down2 = Down(base * 2, base * 4)              # 32
        self.down3 = Down(base * 4, base * 8)              # 16
        self.up1 = Up(base * 8, base * 4, base * 4)        # 32
        self.up2 = Up(base * 4, base * 2, base * 2)        # 64
        self.up3 = Up(base * 2, base, base)                # 128

        self.head_refine = nn.Conv2d(base, 3, 1)
        self.head_replace = nn.Conv2d(base, 6, 1)
        self.head_gate = nn.Conv2d(base, 1, 1)

        # Identity at initialisation: no refinement, and a gate that is closed
        # (sigmoid(-4) ~ 0.018) so the output starts as the input.
        nn.init.zeros_(self.head_refine.weight)
        nn.init.zeros_(self.head_refine.bias)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.constant_(self.head_gate.bias, gate_bias)
        # The replace head starts near the identity rotation rather than at
        # zero, which Gram-Schmidt cannot normalise.
        nn.init.zeros_(self.head_replace.weight)
        with torch.no_grad():
            self.head_replace.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def build_input(self, q_noisy):
        """(B, H, W, 4) -> ((B, C, H, W) features, (B, H, W, 3) local correction)."""
        R = quat_to_matrix(fz_reduce(q_noisy)).flatten(-2)          # (B,H,W,9)
        chans = [R.permute(0, 3, 1, 2)]
        corr = None
        if self.mean_radius:
            # The rotation to the edge-preserving local mean: an approximate
            # answer for the refinement task, handed over rather than left for
            # the trunk to rediscover. Used twice -- as an input channel, and
            # as the base the refine head composes onto in forward().
            # No gradient: this is a fixed function of the input, not of any
            # parameter. Letting autograd track the 7x7 neighbour loop builds a
            # ~50-node graph per forward for nothing, and it is computed in
            # fp32 so autocast cannot quietly halve the precision of the one
            # signal the refinement depends on.
            with torch.no_grad():
                corr = local_correction(q_noisy.float(), radius=self.mean_radius,
                                        scale_deg=CORR_SCALE_DEG).detach()
            chans.append(corr.permute(0, 3, 1, 2))
        if self.use_misfit:
            # 62.8 deg is the largest possible cubic disorientation, so this
            # lands in [0, 1] without needing dataset statistics.
            with torch.no_grad():
                chans.append(local_misfit_deg(q_noisy.float()).detach() / 62.8)
        return torch.cat(chans, dim=1), corr

    def forward(self, q_noisy):
        x, corr = self.build_input(q_noisy)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)

        # The geometry runs in fp32 even under autocast: quaternion products,
        # Gram-Schmidt and slerp all lose meaningful precision in bf16, and the
        # errors being chased here are ~0.03 degrees.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = y.float()
            q_in = qnormalize(q_noisy.float())

            # Refinement is composed ON TOP of the edge-preserving local mean,
            # not predicted from scratch. Feeding the local correction in as an
            # input channel was not enough: the refine head reads the trunk
            # OUTPUT, seven conv blocks downstream of the input and shaped by
            # the much larger gate and replace losses, so the signal did not
            # survive the trip (head weights stayed at 2e-4 after a full epoch).
            # Composing here gives the head a good starting point it can only
            # improve on, and makes the untrained model the local-mean filter
            # (~0.08 deg) rather than the identity (~0.50 deg).
            v = torch.tanh(self.head_refine(y)).permute(0, 2, 3, 1) * self.max_refine
            q_learned = axis_angle_to_quat(v)
            if self.mean_radius:
                q_mean = axis_angle_to_quat(corr * math.radians(CORR_SCALE_DEG))
                q_refine = qnormalize(qmul(q_learned, qmul(q_mean, q_in)))
            else:
                q_refine = qnormalize(qmul(q_learned, q_in))

            q_replace = sixd_to_quat(self.head_replace(y).permute(0, 2, 3, 1))

            gate_logit = self.head_gate(y).permute(0, 2, 3, 1)      # (B,H,W,1)
            # Dead zone: a plain sigmoid never reaches zero, so a "closed" gate
            # of even 1e-3 still slerps a pixel ~0.06 deg toward the replace
            # head -- larger than the error being chased. Subtracting the dead
            # zone makes closed mean exactly closed, so untouched pixels are
            # bit-exact. The gate head keeps training regardless, because its
            # BCE term supervises gate_logit directly rather than this.
            eps = self.gate_deadzone
            gate = F.relu(torch.sigmoid(gate_logit) - eps) / (1.0 - eps)

            q_out = slerp(q_refine, q_replace, gate)

        return {"q": q_out, "q_refine": q_refine, "q_replace": q_replace,
                "gate": gate.squeeze(-1), "gate_logit": gate_logit.squeeze(-1)}


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def boundary_mask(q_clean, tol_deg=0.1):
    """(B, H, W, 4) exact clean quats -> (B, H, W) bool at grain boundaries.

    Computed from the CLEAN map, which is piecewise constant, so any non-zero
    neighbour disorientation is a real boundary and the tolerance is only there
    to absorb float error.
    """
    from orientation_torch import disorientation_deg
    d = torch.zeros(q_clean.shape[:3], device=q_clean.device, dtype=q_clean.dtype)
    m = torch.zeros(q_clean.shape[:3], dtype=torch.bool, device=q_clean.device)
    dh = disorientation_deg(q_clean[:, :, :-1], q_clean[:, :, 1:]) > tol_deg
    dv = disorientation_deg(q_clean[:, :-1], q_clean[:, 1:]) > tol_deg
    m[:, :, :-1] |= dh
    m[:, :, 1:] |= dh
    m[:, :-1] |= dv
    m[:, 1:] |= dv
    return m


def denoise_loss(out, q_clean, bad, bnd, w_bad=5.0, w_bnd=2.0, w_gate=1.0,
                 w_refine=0.5, w_replace=0.5, kind="charbonnier", eps=1e-10):
    """
    Weighted symmetry-aware loss plus direct supervision of each head.

    The per-pixel weights map straight onto the two columns of the scoring
    table that matter: corrupted pixels and boundary pixels.

    The two auxiliary terms exist because the gate starts closed. Without them
    the replace head would receive gradient only through a multiplier of ~0.02
    and would barely train; supervising each head on the pixels it is
    responsible for removes that chicken-and-egg.

    eps sets where the charbonnier stops behaving like an L1 loss on the angle
    and turns L2-like, at theta = 2*sqrt(2*eps) radians. The default 1e-6 turns
    over at 0.16 deg -- above the ~0.03 deg being chased here, which cuts the
    gradient in the target regime by ~77x. 1e-10 moves the turnover to
    0.0016 deg, leaving the whole useful range L1-like. It barely changes the
    gradient at 0.5 deg (2.5 vs 2.4), so early training is unaffected; it only
    sharpens the endgame.
    """
    bad_f = bad.float()
    per_px = disorientation_loss(out["q"], q_clean, kind, eps)
    w = 1.0 + w_bad * bad_f + w_bnd * bnd.float()
    main = (per_px * w).sum() / w.sum().clamp_min(1.0)

    gate = F.binary_cross_entropy_with_logits(out["gate_logit"], bad_f)

    def masked(pred, mask):
        mf = mask.float()
        if mf.sum() < 1:
            return pred.sum() * 0.0
        return (disorientation_loss(pred, q_clean, kind, eps) * mf).sum() / mf.sum()

    refine = masked(out["q_refine"], ~bad)      # the 95% it is responsible for
    replace = masked(out["q_replace"], bad)     # the 5% it is responsible for

    total = main + w_gate * gate + w_refine * refine + w_replace * replace
    return total, {"main": main.item(), "gate": gate.item(),
                   "refine": refine.item(), "replace": replace.item()}
