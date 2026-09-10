"""PSNR and SSIM in plain torch.

Neither scikit-image nor torchmetrics is installed in the `diffusion` env, and
these run on-GPU on a whole batch at once, which is faster than round-tripping
through numpy during validation.

Both expect float tensors in [0, 1], shaped (N, C, H, W).
"""

import torch
import torch.nn.functional as F


def psnr(predicted, target, data_range=1.0, eps=1e-10):
    """Per-image PSNR in dB, returned as a (N,) tensor.

    Averaged per image rather than over the flattened batch: a single batch-wide
    MSE lets one bad image dominate and is not comparable across batch sizes.
    """
    mse = ((predicted - target) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(data_range ** 2 / (mse + eps))


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(predicted, target, data_range=1.0, window_size=11, sigma=1.5):
    """Mean SSIM per image, returned as a (N,) tensor.

    Standard Wang et al. formulation: an 11x11 Gaussian window, applied
    per-channel (depthwise) and averaged over channels.
    """
    channels = predicted.shape[1]
    window = _gaussian_window(
        window_size, sigma, channels, predicted.device, predicted.dtype
    )
    pad = window_size // 2

    def filt(x):
        return F.conv2d(x, window, padding=pad, groups=channels)

    mu_p, mu_t = filt(predicted), filt(target)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p = filt(predicted ** 2) - mu_p_sq
    sigma_t = filt(target ** 2) - mu_t_sq
    sigma_pt = filt(predicted * target) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)
