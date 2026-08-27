#!/usr/bin/env python3
"""
Renders IPF-Z colored orientation maps as PNGs from indexed .ang files.

No DREAM3D/ParaView needed - the .ang file's Euler angles are enough to
compute IPF colors directly with orix, the same technique already used in
06_euler_angle_validation/08_visualize_IPF_colored_map.py, just looped over
the whole batch here instead of one file.

Loops over 11_batch_generation/indexed/{clean,noisy}/*.ang and saves one PNG
per map to 11_batch_generation/ipf_maps/{clean,noisy}/. Resumable - a map
whose PNG already exists is skipped, and one bad file doesn't stop the rest.

Run with (orix installed in the base conda env):
    python3 render_ipf_maps.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")  # no display on this login node
import matplotlib.pyplot as plt
from orix.io import load
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d

HERE = os.path.dirname(os.path.abspath(__file__))
INDEXED_DIR = os.path.join(HERE, "indexed")
OUT_DIR = os.path.join(HERE, "ipf_maps")
VARIANTS = ["clean", "noisy"]


def render_one(ang_path, out_path):
    xmap = load(ang_path)
    symmetry = xmap.phases[0].point_group
    ipf_key = IPFColorKeyTSL(symmetry, direction=Vector3d.zvector())

    rgb = ipf_key.orientation2color(xmap.rotations)
    rgb_map = rgb.reshape(xmap.shape + (3,))

    plt.imsave(out_path, rgb_map)


def main():
    total = converted = skipped = failed = 0

    for variant in VARIANTS:
        in_dir = os.path.join(INDEXED_DIR, variant)
        out_dir = os.path.join(OUT_DIR, variant)
        if not os.path.isdir(in_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)

        for ang_path in sorted(glob.glob(os.path.join(in_dir, "indexed_*.ang"))):
            total += 1
            idx = os.path.basename(ang_path)[len("indexed_"):-len(".ang")]
            out_path = os.path.join(out_dir, f"ipf_{idx}.png")

            if os.path.exists(out_path):
                skipped += 1
                continue

            try:
                render_one(ang_path, out_path)
            except Exception as e:
                print(f"[FAILED] {variant} {idx}: {e}")
                failed += 1
                continue

            converted += 1
            print(f"[{variant} {idx}] -> {out_path}")

    print(
        f"Done. {converted} rendered, {skipped} skipped (already done), "
        f"{failed} failed, out of {total} .ang files found."
    )


if __name__ == "__main__":
    main()
