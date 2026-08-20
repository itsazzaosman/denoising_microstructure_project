# 04 — Microstructure demo

**Role:** a standalone illustration, not part of the pipeline's critical
path. Nothing downstream reads its output.

**Input:** [`02_master_pattern/Ni_master_hires.h5`](02_master_pattern.md)
**Output:** three PNGs — a dead end by design

## What it does

`microstructure_demo.py` invents a fake polycrystalline microstructure: a
40×40 grid of pixels assigned to ~12 synthetic grains, each given a random
crystal orientation. It then uses the real master pattern to render what an
EBSD pattern would actually look like for each of those synthetic grains,
and builds an IPF (inverse pole figure) color map from the random
orientations to visualize the fake grain structure.

This exists purely to illustrate the concept of a grain map and IPF
coloring using real diffraction physics, not to test or produce anything the
rest of the pipeline depends on — no real scan, no real orientations, no
downstream consumer.

## Outputs

- `micro_example_patterns.png` — sample simulated patterns, one per
  synthetic grain
- `micro_grain_map.png` — the fake spatial grain layout
- `micro_ipf_colorkey.png` — the color legend for interpreting the grain map
