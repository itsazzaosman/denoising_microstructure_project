#!/usr/bin/env python3
"""
Run the trained model over the held-out maps and write its output in the same
format MTEX produces.

Files land next to the MTEX outputs for the same noise set, named
<stem>__unet.txt, so score_dataset.py picks the model up automatically
alongside the twelve filters and scores it with exactly the same code. No
separate metric path means no chance of the model being flattered by a
different definition.

Usage
-----
    python inference.py --checkpoint ../checkpoints/checkpoints_scatter5/best.pth \
        --clean-dir ../datasets/clean_euler --noisy-dir ../datasets/noise_scatter5

--out defaults to the matching MTEX folder: datasets/noise_scatter5 ->
datasets/mtex_out_scatter5.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from architecture import GatedOrientationUNet, boundary_mask
from dataset import DirMaps, TestMaps
from orientation_torch import disorientation_deg, quat_to_euler
from provenance import append_provenance, provenance

HEADER = "EulerAngles_0 EulerAngles_1 EulerAngles_2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="e.g. checkpoints/checkpoints_scatter5/best.pth")
    p.add_argument("--out", default=None,
                   help="output folder; defaults to the MTEX folder matching "
                        "--noisy-dir (noise_X -> mtex_out_X)")
    p.add_argument("--method-name", default="unet",
                   help="suffix after __ in the output filename; this is the "
                        "name the scoring table will show")
    p.add_argument("--shape", nargs=2, type=int, default=(128, 128))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--clean-dir", default=None,
                   help="with --noisy-dir, read .txt pairs from these directories "
                        "instead of the packed test cache")
    p.add_argument("--noisy-dir", default=None)
    args = p.parse_args()

    if args.out is None:
        noisy = Path(args.noisy_dir) if args.noisy_dir else None
        if noisy is None or not noisy.name.startswith("noise_"):
            p.error("pass --out, or a --noisy-dir named noise_<set> so the "
                    "output folder mtex_out_<set> can be derived from it")
        args.out = str(noisy.parent / noisy.name.replace("noise_", "mtex_out_", 1))

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targs = ck.get("args", {})
    model = GatedOrientationUNet(
        base=targs.get("base_channels", 64),
        max_refine_deg=targs.get("max_refine_deg", 5.0),
        mean_radius=targs.get("mean_radius", 4),
        mean_tol_deg=targs.get("mean_tol", 2.0),
        use_misfit=not targs.get("no_misfit", False),
        kernel=targs.get("kernel", "mean"),     # runs before --kernel existed used the mean
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.checkpoint}  (epoch {ck['epoch']+1}, "
          f"best score {ck.get('best_score', float('nan')):.4f})", flush=True)

    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.noisy_dir:
        if not args.clean_dir:
            p.error("--noisy-dir requires --clean-dir")
        ds = DirMaps(args.clean_dir, args.noisy_dir, shape=tuple(args.shape),
                     limit=args.limit)
    else:
        ds = TestMaps(shape=tuple(args.shape), limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(ds)} maps -> {outdir}/<stem>__{args.method_name}.txt", flush=True)

    stats = {"all": [], "bad": [], "bnd": [], "gross": [], "in_all": [], "in_gross": []}
    t0 = time.time()
    written = 0

    with torch.no_grad():
        for q_noisy, q_clean, bad, ids in loader:
            q_noisy = q_noisy.to(device)
            q_clean = q_clean.to(device)
            bad = bad.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(q_noisy)
            q_out = out["q"].float()

            d = disorientation_deg(q_out, q_clean)
            d_in = disorientation_deg(q_noisy, q_clean)
            bnd = boundary_mask(q_clean)
            euler = quat_to_euler(q_out).cpu().numpy()

            for b in range(q_out.shape[0]):
                mid = int(ids[b])
                stats["all"].append(d[b].mean().item())
                stats["gross"].append(100.0 * (d[b] > 10).float().mean().item())
                stats["in_all"].append(d_in[b].mean().item())
                stats["in_gross"].append(100.0 * (d_in[b] > 10).float().mean().item())
                if bad[b].any():
                    stats["bad"].append(d[b][bad[b]].mean().item())
                if bnd[b].any():
                    stats["bnd"].append(d[b][bnd[b]].mean().item())

                stem = f"map_{mid:05d}_clean_euler_noisy_noisy"
                np.savetxt(outdir / f"{stem}__{args.method_name}.txt",
                           euler[b].reshape(-1, 3), header=HEADER, comments="",
                           fmt="%.8g")
                written += 1
            if written % 100 < args.batch_size:
                print(f"  {written}/{len(ds)}", flush=True)

    # The maps keep MTEX's file format, so the record of which code and which
    # checkpoint made them sits beside them instead.
    trained_with = ck.get("provenance", {}).get("git", {}).get(
        "commit", "unknown (trained before commits were recorded)")
    append_provenance(outdir / f"{args.method_name}.provenance.json",
                      {**provenance(args=args), "maps_written": written,
                       "checkpoint": {"path": str(Path(args.checkpoint).resolve()),
                                      "epoch": ck["epoch"] + 1,
                                      "trained_with_commit": trained_with}})

    avg = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    print(f"\nwrote {written} maps in {time.time()-t0:.0f}s")
    print("=" * 74)
    print(f"{'':<22}{'ALL':>12}{'CORRUPTED':>12}{'BOUNDARY':>12}{'>10 deg':>12}")
    print("-" * 74)
    print(f"{'noisy input':<22}{avg['in_all']:>12.3f}{'':>12}{'':>12}"
          f"{avg['in_gross']:>12.2f}")
    print(f"{args.method_name:<22}{avg['all']:>12.3f}{avg['bad']:>12.3f}"
          f"{avg['bnd']:>12.3f}{avg['gross']:>12.2f}")
    print("=" * 74)
    print("\nnow run:  make score-all NOISE=<set>   (scores this against every MTEX method)")


if __name__ == "__main__":
    main()
