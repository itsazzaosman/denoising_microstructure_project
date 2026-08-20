# 02 — Master pattern

**Role:** turn the Monte Carlo electron statistics into the full,
orientation-independent diffraction pattern — the expensive calculation
every later simulated pattern is just a re-projection of.

**Input:** [`01_monte_carlo/Ni_MC.h5`](01_monte_carlo.md)
**Output:** `Ni_master_hires.h5` → consumed by
[`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md),
[`04_microstructure_demo`](04_microstructure_demo.md), and
[`08_synthetic_training_data`](08_synthetic_training_data.md)

## What a "master pattern" actually is

Electrons that backscatter out of the crystal near a lattice plane
diffract, producing bands of enhanced/reduced intensity (Kikuchi bands)
whose positions depend only on the crystal's geometry — not on where a
detector happens to sit. The master pattern is this diffraction intensity
computed over the *entire* projection sphere around the crystal at once,
using the Monte Carlo energy/angle statistics as the electron source term
and full dynamical (multi-beam) diffraction theory to compute how those
electrons interfere.

Once this sphere exists, simulating a pattern at any orientation and any
detector geometry is just picking the right patch of the sphere and
projecting it flat — that's exactly what
[`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md) does. This
stage is what makes that cheap.

## The Bethe approximation

Exact dynamical diffraction requires accounting for every possible
diffracted beam simultaneously — computationally intractable. `EMEBSDmaster`
uses the **Bethe potential approximation** (`BetheParameters.nml`) to make
this tractable: beams are sorted into

- **strong** (`c1 = 4.0`) — included in full,
- **weak** (`c2 = 8.0`) — included via perturbation, and
- **ignored** (`c3 = 50.0`, `sgdbdiff = 1.0`) — excitation error too large to
  matter,

based on cutoff thresholds. Both master-pattern runs below share the same
`BetheParameters.nml`.

## Two runs, two resolutions

| File | `npx` | `dmin` | Used for |
|---|---|---|---|
| `EMEBSDmaster.nml` → `Ni_master.h5` | 100 | 0.05 nm | Standard resolution; not used downstream in this repo, kept for comparison |
| `EMEBSDmaster_hires.nml` → `Ni_master_hires.h5` | 500 | 0.03 nm | **The one everything else actually uses** |

`npx` sets the master pattern's angular sampling (2·npx+1 pixels across);
`dmin` is the smallest d-spacing (plane spacing) considered — smaller `dmin`
means more, finer diffracting plane families included. The hi-res version
exists because the standard resolution wasn't fine enough for the actual
pattern simulations that follow.

`EMEBSDmaster.template` is a blank, generic reference copy — not read by
anything.

## Visual check

`view_master.py` opens `Ni_master_hires.h5`, plots it, and saves
`Ni_master.png` — a sanity-check image of the diffraction sphere before any
detector geometry is applied to it.
