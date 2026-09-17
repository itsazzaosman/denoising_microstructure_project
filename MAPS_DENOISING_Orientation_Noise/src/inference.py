#!/usr/bin/env python3
"""
Run the trained model over the held-out maps and write its output in the same
format MTEX produces.

Files land in mtex_out/ named <stem>__unet.txt, so score_dataset.py picks the
model up automatically alongside the twelve filters and scores it with exactly
the same code. No separate metric path means no chance of the model being
flattered by a different definition.

Usage
-----
    python inference.py --checkpoint ../checkpoints/best.pth --limit 500
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from architecture import GatedOrientationUNet, boundary_mask
from dataset import TestMaps
from orientation_torch import disorientation_deg, quat_to_euler

ROOT = Path(__file__).resolve().parent.parent
HEADER = "EulerAngles_0 EulerAngles_1 EulerAngles_2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/best.pth"))
    p.add_argument("--out", default=str(ROOT / "mtex_out"))
    p.add_argument("--method-name", default="unet",
                   help="suffix after __ in the output filename; this is the "
                        "name the scoring table will show")
    p.add_argument("--shape", nargs=2, type=int, default=(128, 128))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targs = ck.get("args", {})
    model = GatedOrientationUNet(
        base=targs.get("base_channels", 64),
        max_refine_deg=targs.get("max_refine_deg", 5.0),
        mean_radius=targs.get("mean_radius", 4),
        use_misfit=not targs.get("no_misfit", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.checkpoint}  (epoch {ck['epoch']+1}, "
          f"best score {ck.get('best_score', float('nan')):.4f})", flush=True)

    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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
                stats["all"].append(d[b].median().item())
                stats["gross"].append(100.0 * (d[b] > 10).float().mean().item())
                stats["in_all"].append(d_in[b].median().item())
                stats["in_gross"].append(100.0 * (d_in[b] > 10).float().mean().item())
                if bad[b].any():
                    stats["bad"].append(d[b][bad[b]].median().item())
                if bnd[b].any():
                    stats["bnd"].append(d[b][bnd[b]].median().item())

                stem = f"map_{mid:05d}_clean_euler_noisy_noisy"
                np.savetxt(outdir / f"{stem}__{args.method_name}.txt",
                           euler[b].reshape(-1, 3), header=HEADER, comments="",
                           fmt="%.8g")
                written += 1
            if written % 100 < args.batch_size:
                print(f"  {written}/{len(ds)}", flush=True)

    med = {k: float(np.median(v)) if v else float("nan") for k, v in stats.items()}
    print(f"\nwrote {written} maps in {time.time()-t0:.0f}s")
    print("=" * 74)
    print(f"{'':<22}{'ALL':>12}{'CORRUPTED':>12}{'BOUNDARY':>12}{'>10 deg':>12}")
    print("-" * 74)
    print(f"{'noisy input':<22}{med['in_all']:>12.3f}{'42.3':>12}{'':>12}"
          f"{med['in_gross']:>12.2f}")
    print(f"{args.method_name:<22}{med['all']:>12.3f}{med['bad']:>12.3f}"
          f"{med['bnd']:>12.3f}{med['gross']:>12.2f}")
    print(f"{'clean-spline (MTEX)':<22}{0.026:>12.3f}{0.032:>12.3f}"
          f"{0.029:>12.3f}{1.48:>12.2f}")
    print("=" * 74)
    print("\nnow run:  make score-all    (scores this against every MTEX method)")


if __name__ == "__main__":
    main()
