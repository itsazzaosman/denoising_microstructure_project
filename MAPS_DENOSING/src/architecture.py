import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net denoiser (the model used for training - see train.py)
#
# Why a U-Net and not the VAE below: denoising an EBSD map is a deterministic,
# pixel-aligned regression. The clean map is piecewise-constant with sharp grain
# boundaries at arbitrary pixel positions. A global latent vector (the VAE's
# 256-d bottleneck) cannot encode boundary positions precisely, so boundaries
# come out smeared. Skip connections let that high-frequency spatial structure
# bypass the bottleneck entirely.
#
# The net predicts a *residual* (the correction to apply to the noisy input)
# rather than re-synthesising the map from scratch. The input is already 95% of
# the answer, so learning the small correction is a much easier target - this is
# the DnCNN trick.
# ---------------------------------------------------------------------------


def _norm(channels, groups=8):
    # GroupNorm rather than BatchNorm so behaviour is independent of batch size
    # (lets you drop the batch size on a smaller GPU without retuning anything).
    return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)


class DoubleConv(nn.Module):
    """(conv -> norm -> SiLU) x 2, the standard U-Net building block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Halve the resolution, then two convs."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Double the resolution, concatenate the skip, then two convs.

    Bilinear upsample + 3x3 conv instead of ConvTranspose2d: transposed convs
    produce checkerboard artefacts, which are especially visible against the
    flat grain interiors in these maps.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        # Guard against odd input sizes; a no-op for the 128x128 maps.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNetDenoiser(nn.Module):
    """Noisy EBSD map -> clean EBSD map.

    Three downsamples, so a 128x128 input reaches a 16x16 bottleneck. Deeper
    would buy little here: the noise is spatially uncorrelated, so the useful
    context for any pixel is local (grain-sized), not global.
    """

    def __init__(self, in_channels=3, out_channels=3, base=64, residual=True):
        super().__init__()
        self.residual = residual

        self.inc = DoubleConv(in_channels, base)            # 128x128
        self.down1 = Down(base, base * 2)                   # 64x64
        self.down2 = Down(base * 2, base * 4)               # 32x32
        self.down3 = Down(base * 4, base * 8)               # 16x16 (bottleneck)

        self.up1 = Up(base * 8, base * 4, base * 4)         # 32x32
        self.up2 = Up(base * 4, base * 2, base * 2)         # 64x64
        self.up3 = Up(base * 2, base, base)                 # 128x128

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

        # Start the residual branch at zero so the untrained model is the
        # identity: epoch 0 already scores the noisy-input baseline instead of
        # emitting garbage, which makes the first validation numbers meaningful.
        if residual:
            nn.init.zeros_(self.outc.weight)
            nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.outc(y)

        # No output activation during training - clamping here would zero the
        # gradient wherever a prediction lands outside [0, 1]. Callers clamp at
        # inference instead (see inference.py).
        return x + y if self.residual else y


def denoise_loss_function(predicted, target, loss_type="l1"):
    """Mean-reduced reconstruction loss.

    Mean rather than sum: a summed loss scales with batch size, which makes the
    printed number meaningless across configs and inflates gradient magnitudes.

    L1 over L2 because it is robust to the clipped, non-Gaussian residual in
    this dataset and preserves edges rather than blurring them. Charbonnier is
    a smooth L1 variant that sometimes trains a little more stably.
    """
    if loss_type == "l1":
        return F.l1_loss(predicted, target)
    if loss_type == "l2":
        return F.mse_loss(predicted, target)
    if loss_type == "charbonnier":
        return torch.sqrt((predicted - target) ** 2 + 1e-6).mean()
    raise ValueError(f"unknown loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# Original VAE. No longer used by train.py, kept so the earlier experiment is
# still runnable. See the note at the top of this file for why it was replaced.
# ---------------------------------------------------------------------------

class EBSD_VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(EBSD_VAE, self).__init__()

        # 1. Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Latent Space Vectors
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # 2. Decoder
        self.dec_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 256, 8, 8)

        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        # sample=False takes the latent mean, making inference deterministic.
        z = self.reparameterize(mu, logvar) if sample else mu
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss_function(reconstructed, original, mu, logvar, beta=1.0):
    recon_loss = F.l1_loss(reconstructed, original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence
