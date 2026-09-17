#!/usr/bin/env python3
"""
Train the gated orientation U-Net.

Validation reports the same four numbers as score_dataset.py -- ALL,
CORRUPTED, BOUNDARY and >10 deg -- computed per map then aggregated across
maps, so the training log is directly comparable to the MTEX baseline table
without any mental conversion.

Model selection uses `gross% + ALL_median`, not the loss. The loss is a
weighted sum with auxiliary terms and its scale says nothing about whether the
model beats a filter; the two quantities in that score are the ones the
baseline is judged on, and the gross-error rate deliberately dominates because
that is where a learned method has room to win.

Outputs (under the project root):
    checkpoints/best.pth, last.pth, history.json
    figures/training_curves.png   redrawn every epoch, safe to look at mid-run
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from architecture import GatedOrientationUNet, boundary_mask, denoise_loss
from dataset import TrainMaps
from orientation_torch import disorientation_deg

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--max-refine-deg", type=float, default=5.0)
    p.add_argument("--mean-radius", type=int, default=4,
                   help="radius of the edge-preserving local mean the refine "
                        "head composes onto; 0 disables it")
    p.add_argument("--loss", default="charbonnier", choices=["charbonnier", "cosine"])
    p.add_argument("--loss-eps", type=float, default=1e-10,
                   help="charbonnier epsilon; the loss turns L2-like below "
                        "2*sqrt(2*eps) rad, so this must sit well under the "
                        "error you are trying to reach")
    p.add_argument("--w-bad", type=float, default=5.0)
    p.add_argument("--w-bnd", type=float, default=2.0)
    p.add_argument("--w-gate", type=float, default=1.0)
    p.add_argument("--w-refine", type=float, default=5.0)
    p.add_argument("--w-replace", type=float, default=0.5)
    p.add_argument("--no-misfit", action="store_true",
                   help="drop the local-misfit input channel (ablation)")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", default="", help="checkpoint path, or 'auto' for last.pth")
    p.add_argument("--limit", type=int, default=0, help="debug: cap training maps")
    p.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    """Per-map medians, then the median across maps -- as score_dataset.py does."""
    model.eval()
    per_map = {"all": [], "bad": [], "bnd": [], "gross": [],
               "in_all": [], "in_gross": []}
    tp = fp = fn = 0

    for q_noisy, q_clean, bad in loader:
        q_noisy = q_noisy.to(device, non_blocking=True)
        q_clean = q_clean.to(device, non_blocking=True)
        bad = bad.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model(q_noisy)

        d = disorientation_deg(out["q"].float(), q_clean)
        d_in = disorientation_deg(q_noisy, q_clean)
        bnd = boundary_mask(q_clean)

        for b in range(d.shape[0]):
            db, bb, nb = d[b], bad[b], bnd[b]
            per_map["all"].append(db.median().item())
            per_map["gross"].append(100.0 * (db > 10).float().mean().item())
            per_map["in_all"].append(d_in[b].median().item())
            per_map["in_gross"].append(100.0 * (d_in[b] > 10).float().mean().item())
            if bb.any():
                per_map["bad"].append(db[bb].median().item())
            if nb.any():
                per_map["bnd"].append(db[nb].median().item())

        pred = out["gate"] > 0.5
        tp += (pred & bad).sum().item()
        fp += (pred & ~bad).sum().item()
        fn += (~pred & bad).sum().item()

    res = {k: float(np.median(v)) if v else float("nan") for k, v in per_map.items()}
    res["gate_precision"] = tp / max(tp + fp, 1)
    res["gate_recall"] = tp / max(tp + fn, 1)
    res["score"] = res["gross"] + res["all"]
    return res


def draw_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))

    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].set_yscale("log")

    ax[1].plot(ep, [h["val"]["all"] for h in history], label="model")
    ax[1].axhline(history[0]["val"]["in_all"], ls="--", c="grey", label="noisy input")
    ax[1].axhline(0.026, ls=":", c="crimson", label="clean-spline baseline")
    ax[1].set_title("val ALL median (deg)"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7)

    ax[2].plot(ep, [h["val"]["gross"] for h in history], label="model")
    ax[2].axhline(1.48, ls=":", c="crimson", label="MTEX pre-clean baseline")
    ax[2].set_title("val >10 deg (%)"); ax[2].set_yscale("log"); ax[2].legend(fontsize=7)

    ax[3].plot(ep, [h["val"]["bnd"] for h in history], label="boundary")
    ax[3].plot(ep, [h["val"]["bad"] for h in history], label="corrupted")
    ax[3].axhline(0.029, ls=":", c="crimson", label="baseline boundary")
    ax[3].set_title("val median by region (deg)"); ax[3].set_yscale("log"); ax[3].legend(fontsize=7)

    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    # On a CPU-only allocation torch defaults to one thread per visible core,
    # which oversubscribes once the DataLoader workers are also running. Leave
    # the workers their share.
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)
    if slurm_cpus and not torch.cuda.is_available():
        torch.set_num_threads(max(1, slurm_cpus - args.num_workers))
        print(f"CPU run: {torch.get_num_threads()} compute threads, "
              f"{args.num_workers} loader workers", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device} | AMP: {amp_dtype or 'off'}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_set = TrainMaps(split="train", val_fraction=args.val_fraction, seed=args.seed,
                          augment=not args.no_augment, limit=args.limit)
    # Validation noise is fixed per map, so a change in the val score is the
    # model moving and not a different draw of noise.
    val_set = TrainMaps(split="val", val_fraction=args.val_fraction, seed=args.seed,
                        augment=False, fixed_noise=True)
    print(f"train maps: {len(train_set)}   val maps: {len(val_set)}   "
          f"(ids {train_set.ids.min()}-{train_set.ids.max()}, all > 500)", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(2, args.num_workers // 2), pin_memory=True)

    model = GatedOrientationUNet(base=args.base_channels,
                                 max_refine_deg=args.max_refine_deg,
                                 mean_radius=args.mean_radius,
                                 use_misfit=not args.no_misfit).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                     eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    ckpt_dir = Path(args.checkpoints_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_score, history = 0, float("inf"), []
    # Path("") is Path("."), which exists and is a directory -- so guard on the
    # string being non-empty, not on the Path being truthy.
    resume = None
    if args.resume == "auto":
        resume = ckpt_dir / "last.pth"
    elif args.resume:
        resume = Path(args.resume)
    if resume is not None and resume.is_file():
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_score = ck.get("best_score", float("inf"))
        history = ck.get("history", [])
        print(f"resumed from {resume} at epoch {start_epoch}", flush=True)

    print("\nbaseline to beat (MTEX, map 1):  ALL 0.026  CORRUPTED 0.032  "
          "BOUNDARY 0.029  >10deg 1.48%\n", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        parts_sum = {}

        for step, (q_noisy, q_clean, bad) in enumerate(train_loader, 1):
            q_noisy = q_noisy.to(device, non_blocking=True)
            q_clean = q_clean.to(device, non_blocking=True)
            bad = bad.to(device, non_blocking=True)
            bnd = boundary_mask(q_clean)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(q_noisy)
            loss, parts = denoise_loss(out, q_clean, bad, bnd,
                                       w_bad=args.w_bad, w_bnd=args.w_bnd,
                                       w_gate=args.w_gate, w_refine=args.w_refine,
                                       w_replace=args.w_replace, kind=args.loss,
                                       eps=args.loss_eps)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * q_noisy.size(0)
            seen += q_noisy.size(0)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
            if step % 100 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"loss {running/seen:.5f}", flush=True)

        scheduler.step()
        train_loss = running / max(seen, 1)
        val = evaluate(model, val_loader, device, amp_dtype)
        dt = time.time() - t0

        print(f"epoch {epoch+1}/{args.epochs}  {dt/60:.1f} min  loss {train_loss:.5f}  "
              f"[{' '.join(f'{k} {v/max(step,1):.4f}' for k, v in parts_sum.items())}]",
              flush=True)
        print(f"    val  ALL {val['all']:.4f}  CORRUPTED {val['bad']:.4f}  "
              f"BOUNDARY {val['bnd']:.4f}  >10deg {val['gross']:.3f}%   "
              f"gate P {val['gate_precision']:.3f} R {val['gate_recall']:.3f}",
              flush=True)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val": val,
                        "lr": scheduler.get_last_lr()[0], "minutes": dt / 60})
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
        try:
            draw_curves(history, fig_dir / "training_curves.png")
        except Exception as exc:
            print(f"    (curve plot failed: {exc})", flush=True)

        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "best_score": best_score, "history": history, "args": vars(args)}
        torch.save(state, ckpt_dir / "last.pth")
        if val["score"] < best_score:
            best_score = val["score"]
            state["best_score"] = best_score
            torch.save(state, ckpt_dir / "best.pth")
            print(f"    new best (score {best_score:.4f}) -> best.pth", flush=True)

    print(f"\ndone. best score {best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
