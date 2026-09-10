"""Run a trained checkpoint and write clean | noisy | denoised comparison figures.

Default - 4 maps from the validation split (data the model never trained on),
one figure each, into visualize/comparisons/:

    python inference.py

Specific maps by filename:

    python inference.py --maps Map_00001.png Map_02500.png

All of them stacked into a single figure:

    python inference.py --num 6 --grid

Score a larger batch without writing figures, and save the bare denoised PNGs:

    python inference.py --num 500 --no-figures --save-denoised ../outputs/denoised

An image from outside the dataset:

    python inference.py --input /some/other/noisy.png --output-name my_map
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from architecture import UNetDenoiser
from dataset import EBSD_Dataset
from metrics import psnr, ssim
from paths import (CLEAN_DIR, NOISY_DIR, BEST_CHECKPOINT, COMPARISONS_DIR,
                   ensure_dirs)
from visualize import save_comparison, save_grid

IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="Denoise EBSD maps and visualise the result")
    p.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    p.add_argument("--clean-dir", default=CLEAN_DIR)
    p.add_argument("--noisy-dir", default=NOISY_DIR)
    p.add_argument("--out-dir", default=COMPARISONS_DIR)

    # Which maps to run - pick one of these three.
    p.add_argument("--maps", nargs="+", default=None,
                   help="specific filenames from the dataset, e.g. Map_00001.png")
    p.add_argument("--num", type=int, default=4,
                   help="take the first N maps of the validation split (default)")
    p.add_argument("--input", default="",
                   help="a noisy image or directory outside the dataset")

    p.add_argument("--grid", action="store_true",
                   help="one combined figure with a row per map, instead of one file each")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the figures, just report the scores")
    p.add_argument("--save-denoised", default="",
                   help="also write the bare denoised PNGs to this directory")
    p.add_argument("--output-name", default="",
                   help="basename for the figure when using --input on a single file")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    return p.parse_args()


def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}\nRun train.py first."
        )

    # weights_only=True: never unpickle arbitrary objects out of a .pth file.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Rebuild with the width the checkpoint was trained at, not the default.
    base = ckpt.get("args", {}).get("base_channels", 64)
    model = UNetDenoiser(base=base)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    best = ckpt.get("best_psnr")
    print(f"Loaded {checkpoint_path} (epoch {ckpt.get('epoch', '?')}"
          + (f", best val PSNR {best:.2f} dB" if isinstance(best, float) else "") + ")")
    return model


def to_tensor(path):
    array = np.array(Image.open(path).convert("RGB"))  # np.array copies -> writable
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def save_png(tensor, path):
    array = (tensor.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def resolve_targets(args):
    """Work out which maps to run, as a list of (name, noisy_path, clean_path).

    clean_path is None when there is no ground truth, in which case the figure
    shows only the noisy input and the denoised output.
    """
    if args.input:
        if os.path.isdir(args.input):
            names = sorted(f for f in os.listdir(args.input) if f.lower().endswith(IMAGE_EXTS))
            if args.num:
                names = names[: args.num]
            return [(n, os.path.join(args.input, n),
                     _clean_or_none(args.clean_dir, n)) for n in names]

        name = args.output_name or os.path.splitext(os.path.basename(args.input))[0]
        return [(name, args.input, _clean_or_none(args.clean_dir, os.path.basename(args.input)))]

    if args.maps:
        names = args.maps
    else:
        # Default: the validation split, so you are always looking at maps the
        # model was not trained on.
        val_set = EBSD_Dataset(args.clean_dir, args.noisy_dir, split="val",
                               val_fraction=args.val_fraction, seed=args.seed)
        names = val_set.image_filenames[: args.num]

    return [(n, os.path.join(args.noisy_dir, n),
             _clean_or_none(args.clean_dir, n)) for n in names]


def _clean_or_none(clean_dir, filename):
    path = os.path.join(clean_dir, filename) if clean_dir else ""
    return path if path and os.path.isfile(path) else None


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)

    targets = resolve_targets(args)
    if not targets:
        raise RuntimeError("no maps to process - check --input / --maps / --num")
    print(f"Processing {len(targets)} map(s) on {device}")

    if args.save_denoised:
        ensure_dirs(args.save_denoised)
    if not args.no_figures:
        ensure_dirs(args.out_dir)

    rows, written = [], []
    scores = {"psnr": [], "ssim": [], "base_psnr": [], "base_ssim": []}

    for start in range(0, len(targets), args.batch_size):
        chunk = targets[start : start + args.batch_size]

        noisy = torch.stack([to_tensor(p) for _, p, _ in chunk]).to(device)
        # Deterministic: a single forward pass, no sampling anywhere.
        denoised = model(noisy).float().clamp(0, 1)

        # Ground truth only exists if every map in this chunk has a clean file.
        has_clean = all(c is not None for _, _, c in chunk)
        clean = (torch.stack([to_tensor(c) for _, _, c in chunk]).to(device)
                 if has_clean else None)

        if clean is not None:
            per_psnr, per_ssim = psnr(denoised, clean), ssim(denoised, clean)
            base_psnr, base_ssim = psnr(noisy, clean), ssim(noisy, clean)
            scores["psnr"] += per_psnr.tolist()
            scores["ssim"] += per_ssim.tolist()
            scores["base_psnr"] += base_psnr.tolist()
            scores["base_ssim"] += base_ssim.tolist()

        for i, (name, _, _) in enumerate(chunk):
            stem = os.path.splitext(name)[0]

            if args.save_denoised:
                save_png(denoised[i], os.path.join(args.save_denoised, f"{stem}.png"))

            if args.no_figures:
                continue

            panel_scores = None
            if clean is not None:
                panel_scores = {
                    "noisy": (base_psnr[i].item(), base_ssim[i].item()),
                    "denoised": (per_psnr[i].item(), per_ssim[i].item()),
                }

            clean_cpu = clean[i].cpu() if clean is not None else torch.zeros_like(noisy[i].cpu())
            if args.grid:
                rows.append((stem, clean_cpu, noisy[i].cpu(), denoised[i].cpu(), panel_scores))
            else:
                out_path = os.path.join(args.out_dir, f"{stem}_comparison.png")
                save_comparison(clean_cpu, noisy[i].cpu(), denoised[i].cpu(),
                                out_path, title=name, scores=panel_scores)
                written.append(out_path)

    if args.grid and rows:
        out_path = os.path.join(args.out_dir, "comparison_grid.png")
        save_grid(rows, out_path, title="Clean vs noisy vs denoised")
        written.append(out_path)

    for path in written:
        print(f"  wrote {path}")
    if args.save_denoised:
        print(f"  denoised PNGs -> {args.save_denoised}")

    if scores["psnr"]:
        m = {k: float(np.mean(v)) for k, v in scores.items()}
        print(
            f"\nOver {len(scores['psnr'])} map(s):\n"
            f"  PSNR  {m['psnr']:6.2f} dB   (noisy input: {m['base_psnr']:.2f} dB, "
            f"gain +{m['psnr']-m['base_psnr']:.2f})\n"
            f"  SSIM  {m['ssim']:6.4f}      (noisy input: {m['base_ssim']:.4f}, "
            f"gain +{m['ssim']-m['base_ssim']:.4f})"
        )


if __name__ == "__main__":
    main()
