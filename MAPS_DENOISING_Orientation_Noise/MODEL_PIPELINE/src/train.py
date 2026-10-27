#!/usr/bin/env python3
"""
Train the gated orientation U-Net.

Validation reports the same four numbers as score_dataset.py -- ALL,
CORRUPTED, BOUNDARY and >10 deg -- computed per map then aggregated across
maps, so the training log is directly comparable to the MTEX baseline table
without any mental conversion.

Model selection minimizes mean cubic-symmetry-aware disorientation (ALL).
Gross errors, boundary errors and errors at misindexed pixels are reported
separately. Fixed validation noise ensures scores are comparable across epochs.

Outputs (under the project root):
    MODEL_PIPELINE/checkpoints/best.pth, last.pth, history.json
    MODEL_PIPELINE/checkpoints/run_info.json     code version, config and settings of every start
    MODEL_PIPELINE/figures/training_curves.png   redrawn every epoch, safe to look at mid-run

Settings can come from an experiment file: `train.py --config
MODEL_PIPELINE/configs/train/<name>.yaml`; flags on the command line override it.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Shared code (orientation maths, paths, scoring) lives in SHARED/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SHARED" / "src"))
from architecture import GatedOrientationUNet, boundary_mask, denoise_loss
from dataset import TrainMaps, ValMaps
from orientation_torch import disorientation_deg
from plotting import draw_curves
from provenance import append_provenance, parse_with_config, provenance

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--max-refine-deg", type=float, default=5.0)
    p.add_argument("--kernel", choices=["learned", "mean"], default="learned",
                   help="how the refinement's local average weights neighbours: "
                        "'learned' (the network picks every weight, no tolerance to "
                        "set) or 'mean' (fixed --mean-tol, the scatter5/7/9 runs)")
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
    # Training uses fresh noise; validation uses fixed corruption per map.
    # For both splits each map has a level drawn from the specified ranges, scatter max uniform in [--scatter-min-deg, --scatter-max-deg] and
    # misindexed fraction uniform in [--misindex-min, --misindex-max]. Minimums
    # above 0 mean every map carries both noise types.
    p.add_argument("--scatter-min-deg", type=float, default=1.0)
    p.add_argument("--scatter-max-deg", type=float, default=15.0)
    p.add_argument("--misindex-min", type=float, default=0.01, help="a fraction, not a percentage")
    p.add_argument("--misindex-max", type=float, default=0.20)
    p.add_argument("--boundary-bias", type=float, default=4.0,
                   help="how much more likely a grain-boundary pixel is to be misindexed")
    p.add_argument("--mean-tol", type=float, default=2.0,
                   help="same-grain tolerance for the local mean, in degrees; "
                        "raise with the scatter level (~1.6x max scatter)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true",
                   help="debugging only: let a run start without a GPU. Real "
                        "training runs on a GPU, never on CPU")
    p.add_argument("--resume", default="", help="checkpoint path, or 'auto' for last.pth")
    p.add_argument("--limit", type=int, default=0, help="debug: cap training maps")
    # Weights & Biases. Log in once with `wandb login`; the API key is never stored here.
    p.add_argument("--wandb-project", default="", help="W&B project; empty = no W&B logging")
    p.add_argument("--wandb-entity", default="", help="W&B user or team; empty = your default")
    p.add_argument("--wandb-name", default="", help="run name in W&B; empty = the config file name")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline"],
                   help="offline: log locally, upload later with `wandb sync`")
    # Relative paths are taken from the project folder (MAPS_DENOISING_Orientation_Noise/).
    p.add_argument("--checkpoints-dir", default="MODEL_PIPELINE/checkpoints",
                   help="where best.pth, last.pth, history.json and run_info.json go")
    p.add_argument("--figures-dir", default="MODEL_PIPELINE/figures",
                   help="where training_curves.png goes")
    return parse_with_config(p, "train")


def start_wandb(args, config, ckpt_dir):
    """The W&B run for this training run, or None when W&B logging is off.

    The run ID is kept in the checkpoint folder, so a resumed job continues the
    same W&B run. Logging never stops training: if W&B cannot reach its server
    (compute nodes may have no internet) it falls back to offline mode, and if
    that fails too, training continues without it.
    """
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("W&B: wandb is not installed in this environment; continuing without it", flush=True)
        return None
    id_file = ckpt_dir / "wandb_run_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:12]
    id_file.write_text(run_id)
    name = args.wandb_name or (Path(config["path"]).stem if config else ckpt_dir.name)
    settings = dict(project=args.wandb_project, entity=args.wandb_entity or None, name=name,
                    id=run_id, resume="allow", dir=str(ckpt_dir),
                    config={**vars(args), "config_file": config["content"] if config else None})
    for mode in dict.fromkeys([args.wandb_mode, "offline"]):
        try:
            run = wandb.init(mode=mode, **settings)
        except Exception as exc:
            print(f"W&B: could not start in {mode} mode ({exc})", flush=True)
            # A failed start keeps the run ID claimed until W&B is shut down,
            # which would make the offline retry fail with "run ID is in use".
            try:
                wandb.finish(exit_code=1)
                wandb.teardown()
            except Exception:
                pass
            continue
        print(f"W&B: logging {mode}, run '{name}' (id {run_id})", flush=True)
        if mode == "offline":
            print(f"W&B: upload it later from a login node with: wandb sync {ckpt_dir / 'wandb'}",
                  flush=True)
        return run
    print("W&B: continuing without it", flush=True)
    return None


def project_path(path):
    """A folder from the config: relative paths are taken from the project folder."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, loss_kw):
    """Per-map means, then the mean across maps -- as score_dataset.py does.

    Also the validation loss: the same denoise_loss, with the same weights, as
    the training loss, so the two curves can be compared directly.
    """
    model.eval()
    per_map = {"all": [], "bad": [], "bnd": [], "gross": [],
               "in_all": [], "in_gross": []}
    tp = fp = fn = 0
    loss_sum, parts_sum, n_maps = 0.0, {}, 0

    for batch in loader:
        q_noisy, q_clean, bad = batch[:3]        # DirMaps adds the map id as a 4th item
        q_noisy = q_noisy.to(device, non_blocking=True)
        q_clean = q_clean.to(device, non_blocking=True)
        bad = bad.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model(q_noisy)

        d = disorientation_deg(out["q"].float(), q_clean)
        d_in = disorientation_deg(q_noisy, q_clean)
        bnd = boundary_mask(q_clean)

        loss, parts = denoise_loss(out, q_clean, bad, bnd, **loss_kw)
        n = q_noisy.size(0)
        loss_sum += loss.item() * n
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v * n
        n_maps += n

        for b in range(d.shape[0]):
            db, bb, nb = d[b], bad[b], bnd[b]
            per_map["all"].append(db.mean().item())
            per_map["gross"].append(100.0 * (db > 10).float().mean().item())
            per_map["in_all"].append(d_in[b].mean().item())
            per_map["in_gross"].append(100.0 * (d_in[b] > 10).float().mean().item())
            if bb.any():
                per_map["bad"].append(db[bb].mean().item())
            if nb.any():
                per_map["bnd"].append(db[nb].mean().item())

        pred = out["gate"] > 0.5
        tp += (pred & bad).sum().item()
        fp += (pred & ~bad).sum().item()
        fn += (~pred & bad).sum().item()

    res = {k: float(np.mean(v)) if v else float("nan") for k, v in per_map.items()}
    res["stat"] = "mean"        # runs before 2026-10-13 stored medians and have no "stat"
    res["gate_precision"] = tp / max(tp + fp, 1)
    res["gate_recall"] = tp / max(tp + fn, 1)
    res["loss"] = loss_sum / max(n_maps, 1)
    res["loss_parts"] = {k: v / max(n_maps, 1) for k, v in parts_sum.items()}
    res["score"] = res["all"]  # mean cubic-symmetry-aware disorientation (degrees)
    return res


def main():
    args, config = parse_args()
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
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("no GPU visible: training runs on a GPU only. Submit "
                         "train_model.sh (it requests --gres=gpu:1); pass "
                         "--allow-cpu only for a quick debug run.")
    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device} | AMP: {amp_dtype or 'off'}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    noise = dict(scatter_min=args.scatter_min_deg, scatter_max=args.scatter_max_deg,
                 misindex_min=args.misindex_min, misindex_max=args.misindex_max,
                 boundary_bias=args.boundary_bias)
    print(f"noise per map: scatter max {args.scatter_min_deg:g}-{args.scatter_max_deg:g} deg, "
          f"misindexed {100 * args.misindex_min:g}-{100 * args.misindex_max:g}%, "
          f"boundary bias {args.boundary_bias:g}x", flush=True)

    # The same process on two caches: training maps (augmented, fresh noise on
    # every access) and validation maps (not augmented, one fixed noise draw per
    # map, so every epoch is scored on the same noisy maps).
    train_set = TrainMaps(limit=args.limit, **noise)
    val_set = ValMaps(**noise)
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("Training and validation caches must not be empty")
    if len(train_set) < args.batch_size:
        raise ValueError("Training data smaller than batch size with drop_last=True")
    overlap = set(train_set.ids.tolist()) & set(val_set.ids.tolist())
    if overlap:
        raise SystemExit(f"{len(overlap)} validation maps are also training maps, "
                         f"e.g. {sorted(overlap)[:5]}")
    print(f"train maps: {len(train_set)} (ids {train_set.ids.min()}-{train_set.ids.max()})   "
          f"val maps: {len(val_set)} (ids {val_set.ids.min()}-{val_set.ids.max()})", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, args.num_workers // 2), pin_memory=True)

    model = GatedOrientationUNet(base=args.base_channels,
                                 max_refine_deg=args.max_refine_deg,
                                 mean_radius=args.mean_radius,
                                 mean_tol_deg=args.mean_tol,
                                 use_misfit=not args.no_misfit,
                                 kernel=args.kernel).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                     eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    ckpt_dir = project_path(args.checkpoints_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = project_path(args.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints -> {ckpt_dir}\nfigures     -> {fig_dir}", flush=True)

    # One record per start, so a run resumed after a code change shows both commits.
    prov = provenance(config, args)
    append_provenance(ckpt_dir / "run_info.json", prov)
    run = start_wandb(args, config, ckpt_dir)
    print(f"code: {prov['git']['commit']}"
          + (f"   config: {config['path']}" if config else ""), flush=True)

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
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_score = ck.get("best_score", float("inf"))
        history = ck.get("history", [])
        print(f"resumed from {resume} at epoch {start_epoch}", flush=True)

    loss_kw = dict(w_bad=args.w_bad, w_bnd=args.w_bnd, w_gate=args.w_gate,
                   w_refine=args.w_refine, w_replace=args.w_replace,
                   kind=args.loss, eps=args.loss_eps)

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
            loss, parts = denoise_loss(out, q_clean, bad, bnd, **loss_kw)

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
        val = evaluate(model, val_loader, device, amp_dtype, loss_kw)
        dt = time.time() - t0

        print(f"epoch {epoch+1}/{args.epochs}  {dt/60:.1f} min  loss {train_loss:.5f}  "
              f"[{' '.join(f'{k} {v/max(step,1):.4f}' for k, v in parts_sum.items())}]",
              flush=True)
        print(f"    val  loss {val['loss']:.5f}  ALL {val['all']:.4f}  CORRUPTED {val['bad']:.4f}  "
              f"BOUNDARY {val['bnd']:.4f}  >10deg {val['gross']:.3f}%   "
              f"gate P {val['gate_precision']:.3f} R {val['gate_recall']:.3f}",
              flush=True)

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "train_parts": {k: v / max(step, 1) for k, v in parts_sum.items()},
                        "val": val,
                        "lr": scheduler.get_last_lr()[0], "minutes": dt / 60,
                        "commit": prov["git"]["commit"]})
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
        try:
            draw_curves({ckpt_dir.name: history}, fig_dir / "training_curves.png")
        except Exception as exc:
            print(f"    (curve plot failed: {exc})", flush=True)

        is_best = val["score"] < best_score
        if is_best:
            best_score = val["score"]

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "history": history,
            "args": vars(args),
            "provenance": prov,
        }

        torch.save(state, ckpt_dir / "last.pth")
        if run is not None:
            run.log({"train/loss": train_loss,
                     **{f"train/{k}": v / max(step, 1) for k, v in parts_sum.items()},
                     "val/loss": val["loss"], "val/ALL": val["all"], "val/BOUNDARY": val["bnd"],
                     "val/CORRUPTED": val["bad"], "val/gross_pct": val["gross"],
                     "val/score": val["score"], "val/noisy_input_ALL": val["in_all"],
                     "val/gate_precision": val["gate_precision"],
                     "val/gate_recall": val["gate_recall"],
                     "lr": scheduler.get_last_lr()[0], "epoch_minutes": dt / 60},
                    step=epoch + 1)
            if is_best:
                run.summary.update({"best_score": best_score, "best_epoch": epoch + 1})

        if is_best:
            torch.save(state, ckpt_dir / "best.pth")
            print(f"    new best (score {best_score:.4f}) -> best.pth", flush=True)

    print(f"\ndone. best score {best_score:.4f}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# Train the gated orientation U-Net.

# Validation reports the same four numbers as score_dataset.py -- ALL,
# CORRUPTED, BOUNDARY and >10 deg -- computed per map then aggregated across
# maps, so the training log is directly comparable to the MTEX baseline table
# without any mental conversion.

# Model selection uses `gross% + ALL_mean`, not the loss. The loss is a
# weighted sum with auxiliary terms and its scale says nothing about whether the
# model beats a filter; the two quantities in that score are the ones the
# baseline is judged on, and the gross-error rate deliberately dominates because
# that is where a learned method has room to win.

# Outputs (under the project root):
#     MODEL_PIPELINE/checkpoints/best.pth, last.pth, history.json
#     MODEL_PIPELINE/checkpoints/run_info.json     code version, config and settings of every start
#     MODEL_PIPELINE/figures/training_curves.png   redrawn every epoch, safe to look at mid-run

# Settings can come from an experiment file: `train.py --config
# MODEL_PIPELINE/configs/train/<name>.yaml`; flags on the command line override it.
# """

# import argparse
# import json
# import os
# import sys
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # Shared code (orientation maths, paths, scoring) lives in SHARED/src.
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SHARED" / "src"))
# from architecture import GatedOrientationUNet, boundary_mask, denoise_loss
# from dataset import TrainMaps, ValMaps
# from orientation_torch import disorientation_deg
# from plot_training import draw_curves
# from provenance import append_provenance, parse_with_config, provenance

# ROOT = Path(__file__).resolve().parents[2]


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--epochs", type=int, default=60)
#     p.add_argument("--batch-size", type=int, default=16)
#     p.add_argument("--lr", type=float, default=2e-4)
#     p.add_argument("--weight-decay", type=float, default=0.0)
#     p.add_argument("--base-channels", type=int, default=32)
#     p.add_argument("--max-refine-deg", type=float, default=5.0)
#     p.add_argument("--kernel", choices=["learned", "mean"], default="learned",
#                    help="how the refinement's local average weights neighbours: "
#                         "'learned' (the network picks every weight, no tolerance to "
#                         "set) or 'mean' (fixed --mean-tol, the scatter5/7/9 runs)")
#     p.add_argument("--mean-radius", type=int, default=4,
#                    help="radius of the edge-preserving local mean the refine "
#                         "head composes onto; 0 disables it")
#     p.add_argument("--loss", default="charbonnier", choices=["charbonnier", "cosine"])
#     p.add_argument("--loss-eps", type=float, default=1e-10,
#                    help="charbonnier epsilon; the loss turns L2-like below "
#                         "2*sqrt(2*eps) rad, so this must sit well under the "
#                         "error you are trying to reach")
#     p.add_argument("--w-bad", type=float, default=5.0)
#     p.add_argument("--w-bnd", type=float, default=2.0)
#     p.add_argument("--w-gate", type=float, default=1.0)
#     p.add_argument("--w-refine", type=float, default=5.0)
#     p.add_argument("--w-replace", type=float, default=0.5)
#     p.add_argument("--no-misfit", action="store_true",
#                    help="drop the local-misfit input channel (ablation)")
#     # Training and validation noise (dataset.py): every access of a map draws its
#     # own level, scatter max uniform in [--scatter-min-deg, --scatter-max-deg] and
#     # misindexed fraction uniform in [--misindex-min, --misindex-max]. Minimums
#     # above 0 mean every map carries both noise types.
#     p.add_argument("--scatter-min-deg", type=float, default=1.0)
#     p.add_argument("--scatter-max-deg", type=float, default=15.0)
#     p.add_argument("--misindex-min", type=float, default=0.01, help="a fraction, not a percentage")
#     p.add_argument("--misindex-max", type=float, default=0.20)
#     p.add_argument("--boundary-bias", type=float, default=4.0,
#                    help="how much more likely a grain-boundary pixel is to be misindexed")
#     p.add_argument("--mean-tol", type=float, default=2.0,
#                    help="same-grain tolerance for the local mean, in degrees; "
#                         "raise with the scatter level (~1.6x max scatter)")
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--num-workers", type=int, default=8)
#     p.add_argument("--no-amp", action="store_true")
#     p.add_argument("--allow-cpu", action="store_true",
#                    help="debugging only: let a run start without a GPU. Real "
#                         "training runs on a GPU, never on CPU")
#     p.add_argument("--resume", default="", help="checkpoint path, or 'auto' for last.pth")
#     p.add_argument("--limit", type=int, default=0, help="debug: cap training maps")
#     p.add_argument("--checkpoints-dir", default=str(ROOT / "MODEL_PIPELINE/checkpoints"))
#     p.add_argument("--figures-dir", default=str(ROOT / "MODEL_PIPELINE/figures"))
#     return parse_with_config(p, "train")


# def set_seed(seed):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# @torch.no_grad()
# def evaluate(model, loader, device, amp_dtype, loss_kw):
#     """Per-map means, then the mean across maps -- as score_dataset.py does.

#     Also the validation loss: the same denoise_loss, with the same weights, as
#     the training loss, so the two curves can be compared directly.
#     """
#     model.eval()
#     per_map = {"all": [], "bad": [], "bnd": [], "gross": [],
#                "in_all": [], "in_gross": []}
#     tp = fp = fn = 0
#     loss_sum, parts_sum, n_maps = 0.0, {}, 0

#     for batch in loader:
#         q_noisy, q_clean, bad = batch[:3]        # DirMaps adds the map id as a 4th item
#         q_noisy = q_noisy.to(device, non_blocking=True)
#         q_clean = q_clean.to(device, non_blocking=True)
#         bad = bad.to(device, non_blocking=True)

#         with torch.autocast(device_type=device.type, dtype=amp_dtype,
#                             enabled=amp_dtype is not None):
#             out = model(q_noisy)

#         d = disorientation_deg(out["q"].float(), q_clean)
#         d_in = disorientation_deg(q_noisy, q_clean)
#         bnd = boundary_mask(q_clean)

#         loss, parts = denoise_loss(out, q_clean, bad, bnd, **loss_kw)
#         n = q_noisy.size(0)
#         loss_sum += loss.item() * n
#         for k, v in parts.items():
#             parts_sum[k] = parts_sum.get(k, 0.0) + v * n
#         n_maps += n

#         for b in range(d.shape[0]):
#             db, bb, nb = d[b], bad[b], bnd[b]
#             per_map["all"].append(db.mean().item())
#             per_map["gross"].append(100.0 * (db > 10).float().mean().item())
#             per_map["in_all"].append(d_in[b].mean().item())
#             per_map["in_gross"].append(100.0 * (d_in[b] > 10).float().mean().item())
#             if bb.any():
#                 per_map["bad"].append(db[bb].mean().item())
#             if nb.any():
#                 per_map["bnd"].append(db[nb].mean().item())

#         pred = out["gate"] > 0.5
#         tp += (pred & bad).sum().item()
#         fp += (pred & ~bad).sum().item()
#         fn += (~pred & bad).sum().item()

#     res = {k: float(np.mean(v)) if v else float("nan") for k, v in per_map.items()}
#     res["stat"] = "mean"        # runs before 2026-10-13 stored medians and have no "stat"
#     res["gate_precision"] = tp / max(tp + fp, 1)
#     res["gate_recall"] = tp / max(tp + fn, 1)
#     res["loss"] = loss_sum / max(n_maps, 1)
#     res["loss_parts"] = {k: v / max(n_maps, 1) for k, v in parts_sum.items()}
#     # With no misindexing there are no corrupted pixels, so "gross" is ~0 and
#     # the score reduces to the ALL statistic -- the right selection criterion
#     # for a scatter-only run.
#     res["score"] = (0.0 if np.isnan(res["gross"]) else res["gross"]) + res["all"]
#     return res


# def main():
#     args, config = parse_args()
#     set_seed(args.seed)

#     # On a CPU-only allocation torch defaults to one thread per visible core,
#     # which oversubscribes once the DataLoader workers are also running. Leave
#     # the workers their share.
#     slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)
#     if slurm_cpus and not torch.cuda.is_available():
#         torch.set_num_threads(max(1, slurm_cpus - args.num_workers))
#         print(f"CPU run: {torch.get_num_threads()} compute threads, "
#               f"{args.num_workers} loader workers", flush=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device.type != "cuda" and not args.allow_cpu:
#         raise SystemExit("no GPU visible: training runs on a GPU only. Submit "
#                          "train_model.sh (it requests --gres=gpu:1); pass "
#                          "--allow-cpu only for a quick debug run.")
#     amp_dtype = None
#     if not args.no_amp and device.type == "cuda":
#         amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
#     print(f"Device: {device} | AMP: {amp_dtype or 'off'}", flush=True)
#     if device.type == "cuda":
#         print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

#     noise = dict(scatter_min=args.scatter_min_deg, scatter_max=args.scatter_max_deg,
#                  misindex_min=args.misindex_min, misindex_max=args.misindex_max,
#                  boundary_bias=args.boundary_bias)
#     print(f"noise per map: scatter max {args.scatter_min_deg:g}-{args.scatter_max_deg:g} deg, "
#           f"misindexed {100 * args.misindex_min:g}-{100 * args.misindex_max:g}%, "
#           f"boundary bias {args.boundary_bias:g}x", flush=True)

#     # The same process on two caches: training maps (augmented, fresh noise on
#     # every access) and validation maps (not augmented, one fixed noise draw per
#     # map, so every epoch is scored on the same noisy maps).
#     train_set = TrainMaps(limit=args.limit, **noise)
#     val_set = ValMaps(**noise)
#     overlap = set(train_set.ids.tolist()) & set(val_set.ids.tolist())
#     if overlap:
#         raise SystemExit(f"{len(overlap)} validation maps are also training maps, "
#                          f"e.g. {sorted(overlap)[:5]}")
#     print(f"train maps: {len(train_set)} (ids {train_set.ids.min()}-{train_set.ids.max()})   "
#           f"val maps: {len(val_set)} (ids {val_set.ids.min()}-{val_set.ids.max()})", flush=True)

#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=True,
#                               drop_last=True, persistent_workers=args.num_workers > 0)
#     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
#                             num_workers=max(2, args.num_workers // 2), pin_memory=True)

#     model = GatedOrientationUNet(base=args.base_channels,
#                                  max_refine_deg=args.max_refine_deg,
#                                  mean_radius=args.mean_radius,
#                                  mean_tol_deg=args.mean_tol,
#                                  use_misfit=not args.no_misfit,
#                                  kernel=args.kernel).to(device)
#     print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
#                                                      eta_min=args.lr * 0.01)
#     scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))

#     ckpt_dir = Path(args.checkpoints_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
#     fig_dir = Path(args.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)

#     # One record per start, so a run resumed after a code change shows both commits.
#     prov = provenance(config, args)
#     append_provenance(ckpt_dir / "run_info.json", prov)
#     print(f"code: {prov['git']['commit']}"
#           + (f"   config: {config['path']}" if config else ""), flush=True)

#     start_epoch, best_score, history = 0, float("inf"), []
#     # Path("") is Path("."), which exists and is a directory -- so guard on the
#     # string being non-empty, not on the Path being truthy.
#     resume = None
#     if args.resume == "auto":
#         resume = ckpt_dir / "last.pth"
#     elif args.resume:
#         resume = Path(args.resume)
#     if resume is not None and resume.is_file():
#         ck = torch.load(resume, map_location=device, weights_only=False)
#         model.load_state_dict(ck["model"])
#         optimizer.load_state_dict(ck["optimizer"])
#         scheduler.load_state_dict(ck["scheduler"])
#         if "scaler" in ck:
#             scaler.load_state_dict(ck["scaler"])
#         start_epoch = ck["epoch"] + 1
#         best_score = ck.get("best_score", float("inf"))
#         history = ck.get("history", [])
#         print(f"resumed from {resume} at epoch {start_epoch}", flush=True)

#     loss_kw = dict(w_bad=args.w_bad, w_bnd=args.w_bnd, w_gate=args.w_gate,
#                    w_refine=args.w_refine, w_replace=args.w_replace,
#                    kind=args.loss, eps=args.loss_eps)

#     for epoch in range(start_epoch, args.epochs):
#         model.train()
#         t0, running, seen = time.time(), 0.0, 0
#         parts_sum = {}

#         for step, (q_noisy, q_clean, bad) in enumerate(train_loader, 1):
#             q_noisy = q_noisy.to(device, non_blocking=True)
#             q_clean = q_clean.to(device, non_blocking=True)
#             bad = bad.to(device, non_blocking=True)
#             bnd = boundary_mask(q_clean)

#             optimizer.zero_grad(set_to_none=True)
#             with torch.autocast(device_type=device.type, dtype=amp_dtype,
#                                 enabled=amp_dtype is not None):
#                 out = model(q_noisy)
#             loss, parts = denoise_loss(out, q_clean, bad, bnd, **loss_kw)

#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()

#             running += loss.item() * q_noisy.size(0)
#             seen += q_noisy.size(0)
#             for k, v in parts.items():
#                 parts_sum[k] = parts_sum.get(k, 0.0) + v
#             if step % 100 == 0:
#                 print(f"  epoch {epoch+1} step {step}/{len(train_loader)} "
#                       f"loss {running/seen:.5f}", flush=True)

#         scheduler.step()
#         train_loss = running / max(seen, 1)
#         val = evaluate(model, val_loader, device, amp_dtype, loss_kw)
#         dt = time.time() - t0

#         print(f"epoch {epoch+1}/{args.epochs}  {dt/60:.1f} min  loss {train_loss:.5f}  "
#               f"[{' '.join(f'{k} {v/max(step,1):.4f}' for k, v in parts_sum.items())}]",
#               flush=True)
#         print(f"    val  loss {val['loss']:.5f}  ALL {val['all']:.4f}  CORRUPTED {val['bad']:.4f}  "
#               f"BOUNDARY {val['bnd']:.4f}  >10deg {val['gross']:.3f}%   "
#               f"gate P {val['gate_precision']:.3f} R {val['gate_recall']:.3f}",
#               flush=True)

#         history.append({"epoch": epoch + 1, "train_loss": train_loss,
#                         "train_parts": {k: v / max(step, 1) for k, v in parts_sum.items()},
#                         "val": val,
#                         "lr": scheduler.get_last_lr()[0], "minutes": dt / 60,
#                         "commit": prov["git"]["commit"]})
#         (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
#         try:
#             draw_curves({ckpt_dir.name: history}, fig_dir / "training_curves.png")
#         except Exception as exc:
#             print(f"    (curve plot failed: {exc})", flush=True)

#         is_best = val["score"] < best_score
#         if is_best:
#             best_score = val["score"]

#         state = {
#             "epoch": epoch,
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict(),
#             "scaler": scaler.state_dict(),
#             "best_score": best_score,
#             "history": history,
#             "args": vars(args),
#             "provenance": prov,
#         }

#         torch.save(state, ckpt_dir / "last.pth")

#         if is_best:
#             torch.save(state, ckpt_dir / "best.pth")
#             print(f"    new best (score {best_score:.4f}) -> best.pth", flush=True)

#     print(f"\ndone. best score {best_score:.4f}", flush=True)


# if __name__ == "__main__":
#     main()
