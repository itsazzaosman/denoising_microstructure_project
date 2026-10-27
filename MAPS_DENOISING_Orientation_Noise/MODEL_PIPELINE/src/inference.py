#!/usr/bin/env python3
"""
Run the trained model over the held-out maps and write its output in the same
format MTEX produces.

Files are named <stem>__<method-name>.txt, the naming the MTEX outputs use,
so score_dataset.py scores the model and the filters with exactly the same
code. No separate metric path means no chance of the model being flattered by
a different definition.

Testing is kept apart from training: the noise here is NOT made on the fly.
Each test set is a folder of fixed noisy .txt maps (made once by
add_ebsd_noise.py), the same files MTEX filtered, so the model and MTEX see
exactly the same noisy pixels.

Usage
-----
    python inference.py --checkpoint ../checkpoints/seed1/best.pth \
        --noisy-dir ../../datasets/noise_scatter_misindexing/noise_scatter5_mis3 \
        --method-name blind_both_seed1

--clean-dir defaults to datasets/clean_euler_zip_unzip/clean_euler.
--out defaults to the model results folder of the same test set:
noise_scatter5_mis3 -> datasets/results_test_set_model/scatter5_mis3.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Shared code (orientation maths, paths, scoring) lives in SHARED/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "SHARED" / "src"))
from architecture import GatedOrientationUNet, boundary_mask
from dataset import euler_to_quat
from orientation import load_euler
from orientation_torch import disorientation_deg, quat_to_euler
from paths import CLEAN_DIR, MODEL_RESULTS
from provenance import append_provenance, provenance

HEADER = "EulerAngles_0 EulerAngles_1 EulerAngles_2"


class TestMaps(Dataset):
    """
    Paired (noisy, clean) test maps read straight from .txt folders.

    Returns (noisy_quaternions, clean_quaternions, misindexed_mask, map_id),
    shaped (H,W,4), (H,W,4), (H,W), int. The Euler -> quaternion conversion is
    dataset.euler_to_quat, the one training used.

    The misindexed mask (<noisy>.badmask.npy) is optional: a scatter-only set
    has none, and CORRUPTED is then reported as nan rather than pretending.
    """

    def __init__(self, clean_dir, noisy_dir, shape=(128, 128), limit=0):
        self.shape = tuple(shape)
        mid = lambda p: (int(m.group(1)) if (m := re.search(r"map_(\d+)", p.name)) else None)

        clean = {mid(f): f for f in Path(clean_dir).glob("map_*.txt") if mid(f)}
        pairs = []
        for f in sorted(Path(noisy_dir).glob("map_*_noisy.txt"), key=lambda p: mid(p) or 0):
            i = mid(f)
            if i in clean:
                mask = Path(str(f) + ".badmask.npy")
                pairs.append((i, clean[i], f, mask if mask.exists() else None))
        if limit:
            pairs = pairs[:limit]
        if not pairs:
            raise RuntimeError(f"no clean/noisy pairs between {clean_dir} and {noisy_dir}")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        H, W = self.shape
        mid, cf, nf, mf = self.pairs[i]
        q_clean = euler_to_quat(load_euler(cf).astype(np.float64)).reshape(H, W, 4)
        q_noisy = euler_to_quat(load_euler(nf).astype(np.float64)).reshape(H, W, 4)
        bad = (np.load(mf).reshape(H, W) if mf is not None
               else np.zeros((H, W), dtype=bool))
        return (torch.from_numpy(np.ascontiguousarray(q_noisy, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(q_clean, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(bad, dtype=np.bool_)),
                int(mid))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="e.g. MODEL_PIPELINE/checkpoints/seed1/best.pth")
    p.add_argument("--out", default=None,
                   help="output folder; defaults to the model results folder "
                        "of --noisy-dir (noise_X -> results_test_set_model/X)")
    p.add_argument("--method-name", default="unet",
                   help="suffix after __ in the output filename; this is the "
                        "name the scoring table will show")
    p.add_argument("--shape", nargs=2, type=int, default=(128, 128))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--clean-dir", default=str(CLEAN_DIR),
                   help="folder of the clean map_*.txt files (the ground truth)")
    p.add_argument("--noisy-dir", required=True,
                   help="one test set: a folder of map_*_noisy.txt files, e.g. "
                        "datasets/noise_scatter_misindexing/noise_scatter5_mis3")
    args = p.parse_args()

    if args.out is None:
        noisy = Path(args.noisy_dir)
        if not noisy.name.startswith("noise_"):
            p.error("pass --out, or a --noisy-dir named noise_<set> so the "
                    "output folder results_test_set_model/<set> can be derived from it")
        args.out = str(MODEL_RESULTS / noisy.name[len("noise_"):])

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

    ds = TestMaps(args.clean_dir, args.noisy_dir, shape=tuple(args.shape), limit=args.limit)
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
