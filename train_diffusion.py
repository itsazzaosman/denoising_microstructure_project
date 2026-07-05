import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from abc import abstractmethod
from torchvision import datasets, transforms
import torchvision.utils as vutils
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import wandb
import datetime

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Beta Schedules ───────────────────────────────────────────────────────────
def linear_beta_schedule(timesteps, start_scale=0.0001, end_scale=0.02):
    return torch.linspace(start_scale, end_scale, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()

# ─── Timestep Embedding ───────────────────────────────────────────────────────
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# ─── UNet Building Blocks ─────────────────────────────────────────────────────
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, t):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, t):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x

def norm_layer(channels):
    return nn.GroupNorm(32, channels)

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale).softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v).reshape(B, -1, H, W)
        return self.proj(h) + x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1) if use_conv else nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)

class UNetModel(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, out_channels=3,
                 num_res_blocks=2, attention_resolutions=(8, 16), dropout=0,
                 channel_mult=(1, 2, 4, 8), conv_resample=True, num_heads=4):
        super().__init__()
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch, ds = model_channels, 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + down_block_chans.pop(), model_channels * mult, time_embed_dim, dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.up_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)

# ─── Gaussian Diffusion ───────────────────────────────────────────────────────
class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        return mean, self._extract(self.posterior_variance, t, x_t.shape), self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        pred_noise = model(x_t, t)
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1., 1.)
        return self.q_posterior_mean_variance(x_recon, x_t, t)

    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        mean, _, log_var = self.p_mean_variance(model, x_t, t, clip_denoised)
        noise = torch.randn_like(x_t)
        mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        return mean + mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
        return img

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size))

    def train_losses(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        return F.mse_loss(model(x_noisy, t), noise)

# ─── Config ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 128
CHANNELS = 3
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TIMESTEPS = 500

timestamp = datetime.datetime.now().strftime("%b%d_%H-%M")
run_name = f"DDPM_ep{NUM_EPOCHS}_bs{BATCH_SIZE}_{timestamp}"

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(
    project="microstructure-ddpm", 
    name=run_name,
    config={
        "image_size": IMAGE_SIZE,
        "channels": CHANNELS,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "timesteps": TIMESTEPS,
        "model_channels": 128
    }
)

BASE_DIR = "/project/community/aiosman/diffusion_project"


# BASE_DIR           = "/mnt/microstructures-vol"
train_dataset_path = os.path.join(BASE_DIR, "dataset_split/train")
test_dataset_path  = os.path.join(BASE_DIR, "dataset_split/test")
ckpt_dir           = os.path.join(BASE_DIR, "checkpoints")
sample_dir         = os.path.join(BASE_DIR, "samples")
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset     = datasets.ImageFolder(train_dataset_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=2, pin_memory=True)

# ─── Model ────────────────────────────────────────────────────────────────────
model = UNetModel(
    in_channels=CHANNELS,
    model_channels=128,
    out_channels=CHANNELS,
    channel_mult=(1, 2, 4, 8),
    attention_resolutions=[16, 8]
).to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=500)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Resume from checkpoint if one exists
start_epoch = 0
# latest_ckpt = None
# if os.path.exists(ckpt_dir):
#     ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
#     if ckpts:
#         latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])

# if latest_ckpt:
#     print(f"Resuming from checkpoint: {latest_ckpt}")
#     ckpt = torch.load(latest_ckpt, map_location=device)
#     model.load_state_dict(ckpt['model_state_dict'])
#     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#     start_epoch = ckpt['epoch']
#     print(f"Resumed from epoch {start_epoch}")

# ─── Training ─────────────────────────────────────────────────────────────────
def validate_diffusion(model, diffusion, dataloader):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss = diffusion.train_losses(model, x, t)
            total_val_loss += loss.item()
    model.train()
    return total_val_loss / len(dataloader)
def train_diffusion(model, diffusion, dataloader, optimizer, num_epochs, start_epoch=0):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    model.train()

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss = diffusion.train_losses(model, x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wandb.log({"batch_loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        val_loss = validate_diffusion(model, gaussian_diffusion, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "avg_epoch_loss": avg_loss, "val_loss": val_loss})


        # Save checkpoint every 50 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.sample(model, image_size=IMAGE_SIZE, batch_size=1, channels=CHANNELS)
            grid = vutils.make_grid(samples, nrow=1, normalize=True, value_range=(-1, 1))

            wandb.log({"generated_samples": wandb.Image(grid, caption=f"Epoch {epoch+1}")})
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.title(f"Epoch {epoch+1}")
            plt.savefig(os.path.join(sample_dir, f"samples_epoch_{epoch+1}.png"), bbox_inches='tight')
            plt.close()
            print(f"Saved samples to {sample_dir}/samples_epoch_{epoch+1}.png")
            model.train()

    wandb.finish()

train_diffusion(model, gaussian_diffusion, train_loader, optimizer, NUM_EPOCHS, start_epoch)
print("Training complete!")