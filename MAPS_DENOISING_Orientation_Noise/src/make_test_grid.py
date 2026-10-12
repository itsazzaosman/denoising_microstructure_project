#!/usr/bin/env python3
"""
Define the full test grid and write one noise config per test set.

    scatter max (deg)    1  3  5  7  9  12  15
    misindexed (%)       0  3  6  9

Every combination is one test set on maps 1-500: datasets/noise_<name>, with
<name> = scatter<S> when nothing is misindexed (the existing scatter5/7/9 sets
are these cells) and scatter<S>_mis<M> otherwise. 1 deg is a good scan; 15 deg
leaves the grains only just visible, past which the noise approaches the
misorientation between neighbouring grains.

All sets share seed 0, so map i gets the same random tilt axes and the same
random fractions of the maximum angle in every set, and the scatter draw is
identical across the misindexing levels (it is drawn first). The cells
therefore differ only in the amount of each noise type: a paired, factorial
design in which any difference between two cells is the noise, never the draw.

Writes configs/noise/<name>.yaml for every new cell (existing files are left
alone) and configs/test_grid.txt, the list of all 28 names, for loops.

Usage
-----
    python make_test_grid.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCATTER_DEG = [1, 3, 5, 7, 9, 12, 15]
MISINDEX_PCT = [0, 3, 6, 9]


def name(s, m):
    return f"scatter{s}" if m == 0 else f"scatter{s}_mis{m}"


def config(s, m):
    return f"""\
# Test set datasets/noise_{name(s, m)}: maps 1-500, one cell of the test grid
# (see src/make_test_grid.py, which wrote this file).
#
#   make noise NOISE={name(s, m)}
description: Uniform-angle scatter up to {s} deg ({s / 2:g} deg mean), {m}% misindexed, boundary-biased.

noise:                  # add_ebsd_noise.py settings
  shape: [128, 128]
  start: 0              # first map (0 = map_00001)
  limit: 500            # maps 1-500 are the test set
  scatter: {float(s)}{' ' * (13 - len(str(float(s))))}# max tilt angle in deg; angle is uniform in [0, max]
  misindex: {m / 100:.2f}
  boundary_bias: 4.0    # a grain-boundary pixel is 4x more likely to be misindexed
  seed: 0               # shared by every cell of the grid: paired design
  save_mask: true
  ang: true             # .ang copies for MTEX
  step: 1.0
"""


def main():
    out = ROOT / "configs/noise"
    names, written = [], []
    for m in MISINDEX_PCT:
        for s in SCATTER_DEG:
            n = name(s, m)
            names.append(n)
            path = out / f"{n}.yaml"
            if not path.exists():
                path.write_text(config(s, m))
                written.append(n)
    (ROOT / "configs/test_grid.txt").write_text("".join(f"{n}\n" for n in names))
    print(f"{len(names)} test sets in configs/test_grid.txt; wrote {len(written)} new configs, "
          f"kept {len(names) - len(written)} existing")


if __name__ == "__main__":
    main()
