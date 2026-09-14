"""PSNR and SSIM in plain torch.

Neither scikit-image nor torchmetrics is installed in the `ebsd` env, and
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
    data_range is the difference between the maximum and minimum possible values in the image. For example, if your images are in [0, 1] (normalized), data_range=1.0; if your images are in [0, 255], data_range=255.0.
    """
    # each image has the shape of (batch_size or the number of images, channels, height, width)
    # flatten(1) start at dim 1 which is channels and multiply all the channels and height and width together to get a single value for each image
    # therefore after the flattern(1) the shape of the tensor will be (batch_size, channels * height * width)
    # then we take the mean of each image to get the MSE for each image and dim(1) means we are taking the mean across the channels * height * width dimension for each image in the batch
    # therefore we will get a tensor of shape (batch_size,) which is the MSE for each image in the batch
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

    def filt(x): # Apply the Gaussian window to the image 
        # do not look at each pixl but look at the surrounding pixels and take a weighted average of the surrounding pixels to get a new value for each pixel
        return F.conv2d(x, window, padding=pad, groups=channels)

    mu_p, mu_t = filt(predicted), filt(target)
    # What is the local average brightness around this pixel in the predicted/target image?
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p = filt(predicted ** 2) - mu_p_sq
    sigma_t = filt(target ** 2) - mu_t_sq
    sigma_pt = filt(predicted * target) - mu_pt

    c1 = (0.01 * data_range) ** 2 # constants to stabilize the division with weak denominator
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)
