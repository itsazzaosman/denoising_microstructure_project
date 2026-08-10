#!/usr/bin/env python3
"""
Visualise the raw EBSPs inside patterns.h5

Produces four views of the same 4125 patterns:

  ebsp_montage.png     a grid of patterns sampled evenly across the scan
  ebsp_raw_vs_proc.png the same patterns before and after background removal
  ebsp_mosaic.png      every pattern tiled in its map position (the striking one)
  ebsp_with_map.png    IPF map with marked points and the pattern from each

Run:
    python3 07_visualize_ebsps.py
    python3 07_visualize_ebsps.py --file patterns.h5 --rows 6 --cols 8
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="patterns.h5")
    p.add_argument("--rows", type=int, default=5, help="montage rows")
    p.add_argument("--cols", type=int, default=8, help="montage columns")
    p.add_argument("--mosaic-max", type=int, default=6000,
                   help="cap on mosaic pixel width/height; downsamples if exceeded")
    p.add_argument("--no-mosaic", action="store_true")
    args = p.parse_args()

    try:
        import kikuchipy as kp
    except ImportError:
        sys.exit("Needs kikuchipy:  pip install kikuchipy")

    print(f"Loading {args.file}")
    s = kp.load(args.file)
    pats = np.asarray(s.data)
    ny, nx, sy, sx = pats.shape
    print(f"  {ny} x {nx} scan points, each pattern {sy} x {sx} px")
    print(f"  dtype {pats.dtype}, range {pats.min()}-{pats.max()}")

    # ------------------------------------------------------------------
    # 1. Montage of patterns sampled across the scan
    # ------------------------------------------------------------------
    r_idx = np.linspace(0, ny - 1, args.rows).astype(int)
    c_idx = np.linspace(0, nx - 1, args.cols).astype(int)

    fig, ax = plt.subplots(args.rows, args.cols,
                           figsize=(args.cols * 1.5, args.rows * 1.6))
    ax = np.atleast_2d(ax)
    for i, r in enumerate(r_idx):
        for j, c in enumerate(c_idx):
            ax[i, j].imshow(pats[r, c], cmap="gray")
            ax[i, j].set_title(f"({r},{c})", fontsize=7)
            ax[i, j].axis("off")
    fig.suptitle("Raw EBSPs sampled across the scan", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/ebsp_montage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/ebsp_montage.png")

    # ------------------------------------------------------------------
    # 2. Raw vs background-corrected
    # ------------------------------------------------------------------
    print("Removing background (on a copy)")
    s2 = s.deepcopy()
    try:
        s2.remove_static_background()
    except Exception as e:
        print(f"  static background skipped: {e}")
    try:
        s2.remove_dynamic_background()
    except Exception as e:
        print(f"  dynamic background skipped: {e}")
    proc = np.asarray(s2.data)

    picks = [(ny // 6, nx // 6), (ny // 3, nx // 2),
             (ny // 2, nx // 3), (2 * ny // 3, 2 * nx // 3)]
    fig, ax = plt.subplots(2, len(picks), figsize=(len(picks) * 3, 6.4))
    for j, (r, c) in enumerate(picks):
        ax[0, j].imshow(pats[r, c], cmap="gray")
        ax[0, j].set_title(f"raw ({r},{c})", fontsize=10)
        ax[0, j].axis("off")
        ax[1, j].imshow(proc[r, c], cmap="gray")
        ax[1, j].set_title("background removed", fontsize=10)
        ax[1, j].axis("off")
    plt.tight_layout()
    plt.savefig("outputs/ebsp_raw_vs_proc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote outputs/ebsp_raw_vs_proc.png")

    # ------------------------------------------------------------------
    # 3. Full mosaic: every pattern tiled at its map position
    # ------------------------------------------------------------------
    if not args.no_mosaic:
        print("Building full mosaic")
        arr = proc.astype(np.float32)
        # normalise each pattern so tiles are comparable
        mn = arr.min(axis=(2, 3), keepdims=True)
        mx = arr.max(axis=(2, 3), keepdims=True)
        arr = (arr - mn) / np.maximum(mx - mn, 1e-9)

        pad = np.pad(arr, ((0, 0), (0, 0), (0, 1), (0, 1)), constant_values=1.0)
        mosaic = pad.transpose(0, 2, 1, 3).reshape(ny * (sy + 1), nx * (sx + 1))
        print(f"  mosaic {mosaic.shape[0]} x {mosaic.shape[1]} px")

        step = 1
        while max(mosaic.shape) // step > args.mosaic_max:
            step += 1
        if step > 1:
            print(f"  downsampling by {step}")
            mosaic = mosaic[::step, ::step]

        fig, axm = plt.subplots(figsize=(14, 14 * mosaic.shape[0] / mosaic.shape[1]))
        axm.imshow(mosaic, cmap="gray", interpolation="nearest")
        axm.set_title(f"All {ny * nx} EBSPs in their map positions", fontsize=14)
        axm.axis("off")
        plt.tight_layout()
        plt.savefig("outputs/ebsp_mosaic.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote outputs/ebsp_mosaic.png")

    # ------------------------------------------------------------------
    # 4. IPF map with marked points and their patterns
    # ------------------------------------------------------------------
    try:
        from orix.plot import IPFColorKeyTSL
        from orix.vector import Vector3d

        xmap = s.xmap
        pg = xmap.phases[0].point_group
        ckey = IPFColorKeyTSL(pg, direction=Vector3d.zvector())
        rgb = ckey.orientation2color(xmap.rotations).reshape(ny, nx, 3)

        eu = np.rad2deg(np.asarray(xmap.rotations.to_euler()).reshape(-1, 3))
        eu[:, 0] %= 360
        eu[:, 1] %= 180
        eu[:, 2] %= 360

        fig = plt.figure(figsize=(15, 4.6))
        axmap = fig.add_subplot(1, len(picks) + 1, 1)
        axmap.imshow(np.clip(rgb, 0, 1))
        axmap.set_title("IPF-Z map", fontsize=11)
        axmap.axis("off")

        colors = ["k", "w", "k", "w"]
        for j, (r, c) in enumerate(picks):
            axmap.plot(c, r, "o", mfc="none", mec=colors[j % 4], ms=11, mew=2)
            axmap.annotate(str(j + 1), (c, r), color=colors[j % 4],
                           fontsize=11, xytext=(5, 5), textcoords="offset points")

            axp = fig.add_subplot(1, len(picks) + 1, j + 2)
            axp.imshow(proc[r, c], cmap="gray")
            p1, P, p2 = eu[r * nx + c]
            axp.set_title(f"{j+1}: ({r},{c})\n{p1:.1f}, {P:.1f}, {p2:.1f}",
                          fontsize=9)
            axp.axis("off")

        plt.tight_layout()
        plt.savefig("outputs/ebsp_with_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote outputs/ebsp_with_map.png")
    except Exception as e:
        print(f"(skipped map figure: {e})")

    print("\nDone.  Open with:  explorer.exe .")


if __name__ == "__main__":
    main()