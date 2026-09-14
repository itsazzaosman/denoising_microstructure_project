"""Train the U-Net denoiser on paired (noisy, clean) EBSD maps.

    python train.py --epochs 50 --batch-size 64

Where things go (all defined in paths.py):

    checkpoints/best.pth        weights with the highest validation PSNR
    checkpoints/last.pth        most recent epoch, for --resume auto
    checkpoints/history.json    per-epoch metrics, feeds the plots
    visualize/curves/           loss / PSNR / SSIM curves, redrawn every epoch
    visualize/comparisons/      clean | noisy | denoised samples, written at the end

Every epoch reports validation PSNR/SSIM next to the noisy input's own score,
so you can always see whether the model is beating "do nothing".
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from architecture import UNetDenoiser, denoise_loss_function
from dataset import EBSD_Dataset
from metrics import psnr, ssim
from paths import (CLEAN_DIR, NOISY_DIR, CHECKPOINTS_DIR, HISTORY_JSON,
                   CURVES_DIR, COMPARISONS_DIR, ensure_dirs)
from visualize import plot_curves, save_comparison


def parse_args():
    p = argparse.ArgumentParser(description="EBSD map denoiser training")
    p.add_argument("--clean-dir", default=CLEAN_DIR)
    p.add_argument("--noisy-dir", default=NOISY_DIR)
    p.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    p.add_argument("--curves-dir", default=CURVES_DIR)
    p.add_argument("--comparisons-dir", default=COMPARISONS_DIR)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--loss", default="l1", choices=["l1", "l2", "charbonnier"])

    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true", help="disable flip/rot90 augmentation")
    p.add_argument("--cache", action="store_true", help="hold the dataset in RAM (~3.9 GB)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--resume", default="", help="checkpoint to resume from, or 'auto' for last.pth")
    p.add_argument("--n-samples", type=int, default=4,
                   help="comparison figures to write when training finishes")
    p.add_argument("--limit", type=int, default=0, help="debug: cap the number of training images")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    """Validation pass. Returns model PSNR/SSIM and the noisy input's baseline.

    The baseline is the score of doing nothing at all. If model PSNR is not
    comfortably above it, the model is not helping - a falling training loss on
    its own will not tell you that.
    """
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "base_psnr": 0.0, "base_ssim": 0.0}
    n = 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            predicted = model(noisy)

        # Metrics in fp32 on the clamped output - that is what inference writes
        # out, so this is the number that reflects the saved PNGs.
        predicted = predicted.float().clamp(0.0, 1.0)

        totals["loss"] += denoise_loss_function(predicted, clean).item() * noisy.size(0)
        totals["psnr"] += psnr(predicted, clean).sum().item()
        totals["ssim"] += ssim(predicted, clean).sum().item()
        totals["base_psnr"] += psnr(noisy, clean).sum().item()
        totals["base_ssim"] += ssim(noisy, clean).sum().item()
        n += noisy.size(0)

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def write_sample_comparisons(model, val_set, device, out_dir, n_samples):
    """Write clean | noisy | denoised figures for the first few val maps."""
    if n_samples <= 0:
        return []
    model.eval()
    ensure_dirs(out_dir)
    written = []

    for i in range(min(n_samples, len(val_set))):
        noisy, clean = val_set[i]
        name = val_set.image_filenames[i]

        batch = noisy.unsqueeze(0).to(device)
        denoised = model(batch).float().clamp(0, 1)
        clean_b = clean.unsqueeze(0).to(device)

        scores = {
            "noisy": (psnr(batch, clean_b).item(), ssim(batch, clean_b).item()),
            "denoised": (psnr(denoised, clean_b).item(), ssim(denoised, clean_b).item()),
        }
        out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_comparison.png")
        save_comparison(clean, noisy, denoised[0].cpu(), out_path,
                        title=name, scores=scores)
        written.append(out_path)

    return written


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.checkpoints_dir, args.curves_dir, args.comparisons_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 on Ampere and newer avoids the loss-scaling failure modes of fp16.
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device} | AMP: {amp_dtype if amp_dtype else 'off'}")

    train_set = EBSD_Dataset(
        args.clean_dir, args.noisy_dir, split="train",
        val_fraction=args.val_fraction, seed=args.seed,
        augment=not args.no_augment, cache=args.cache, limit=args.limit,
    )
    val_set = EBSD_Dataset(
        args.clean_dir, args.noisy_dir, split="val",
        val_fraction=args.val_fraction, seed=args.seed,
        augment=False, cache=args.cache,
    )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = UNetDenoiser(base=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNetDenoiser | {n_params/1e6:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    # GradScaler is only needed for fp16; bf16 has the range to go without.
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    start_epoch = 0
    best_psnr = float("-inf")
    history = []

    history_path = os.path.join(args.checkpoints_dir, os.path.basename(HISTORY_JSON))
    last_path = os.path.join(args.checkpoints_dir, "last.pth")
    best_path = os.path.join(args.checkpoints_dir, "best.pth")

    resume_path = last_path if args.resume == "auto" else args.resume
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", best_psnr)
        # Reload history too, so the curves continue rather than restarting.
        if os.path.isfile(history_path):
            with open(history_path) as f:
                history = [h for h in json.load(f) if h["epoch"] <= ckpt["epoch"]]
        print(f"Resumed from {resume_path} at epoch {start_epoch} (best PSNR {best_psnr:.3f} dB)")
    elif args.resume and args.resume != "auto":
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()

        for step, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                predicted = model(noisy)
                loss = denoise_loss_function(predicted, clean, args.loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * noisy.size(0)
            seen += noisy.size(0)
            if step % 100 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} loss {running/seen:.5f}", flush=True)

        scheduler.step()
        train_loss = running / max(seen, 1)
        val = evaluate(model, val_loader, device, amp_dtype)

        print(
            f"Epoch {epoch+1}/{args.epochs} | {time.time()-t0:.0f}s | "
            f"train {args.loss} {train_loss:.5f} | val L1 {val['loss']:.5f} | "
            f"PSNR {val['psnr']:.2f} dB (baseline {val['base_psnr']:.2f}, "
            f"+{val['psnr']-val['base_psnr']:.2f}) | "
            f"SSIM {val['ssim']:.4f} (baseline {val['base_ssim']:.4f})",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_psnr": val["psnr"],
            "val_ssim": val["ssim"],
            "baseline_psnr": val["base_psnr"],
            "baseline_ssim": val["base_ssim"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Redrawn every epoch (overwriting), so you can watch the curves during
        # a long run instead of waiting for it to finish.
        plot_curves(history, args.curves_dir)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_psnr": max(best_psnr, val["psnr"]),
            "args": vars(args),
        }
        # Written every epoch so a 12h SLURM wall-clock kill costs one epoch,
        # not the whole run. Restart with --resume auto.
        torch.save(ckpt, last_path)
        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            torch.save(ckpt, best_path)
            print(f"  new best: {best_psnr:.3f} dB -> best.pth", flush=True)

    print(f"\nTraining complete. Best val PSNR: {best_psnr:.3f} dB")

    # Reload the best weights so the sample figures show the best model, not
    # whatever the last epoch happened to land on.
    if os.path.isfile(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)["model"]
        )
    samples = write_sample_comparisons(
        model, val_set, device, args.comparisons_dir, args.n_samples
    )

    print(f"  weights      -> {args.checkpoints_dir}")
    print(f"  curves       -> {args.curves_dir}")
    if samples:
        print(f"  comparisons  -> {args.comparisons_dir} ({len(samples)} figures)")


if __name__ == "__main__":
    main()
