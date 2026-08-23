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
detector happens to sit. **The master pattern is this diffraction intensity
computed over the entire projection sphere around the crystal at once**,
using the Monte Carlo energy/angle statistics as the electron source term
and full dynamical (multi-beam) diffraction theory to compute how those
electrons interfere.

Once this sphere exists, simulating a pattern at any orientation and any
detector geometry is just picking the right patch of the sphere and
projecting it flat — that's exactly what
[`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md) does. This
stage is what makes that cheap.

EMEBSDmaster computes that diffracted intensity over the entire projection sphere around the crystal at once, using your Ni_MC.h5 statistics as the electron source term and full dynamical (multi-beam) diffraction theory.

Once that sphere is computed, every later simulated EBSD pattern (steps 03, 04, 08) is just "pick a patch of the sphere for this orientation/detector geometry and project it flat" — cheap. This is the expensive step that makes everything downstream cheap.


So in **short**: input = Monte Carlo statistics + crystal structure + Bethe/resolution settings; output = one HDF5 file carrying the crystal, the original MC data, the new 11-energy-bin diffraction sphere (in two projection styles), and a full provenance trail — everything 03_ebsd_pattern_simulation needs to generate an actual detector-geometry EBSD pattern next.



## Energy bins

`Ni_MC.h5` doesn't record backscattered electrons at a single energy —
electrons enter at `EkeV = 20.0` keV but lose varying amounts of energy
scattering around inside the crystal before escaping. The Monte Carlo step
sorted all 10⁸ exit events into energy buckets set by `MCNi.nml`'s
`EkeV = 20.0` (highest), `Ehistmin = 10.0` (lowest), `Ebinsize = 1.0`
(bucket width) — giving **11 energy bins**: 20, 19, 18, ... down to 10 keV.

`EMEBSDmaster` computes a **full separate diffraction sphere for each of
these 11 energies** rather than one averaged sphere, because diffraction
depends on electron wavelength, which depends on energy — a 20 keV electron
and a 10 keV electron diffract off the same lattice differently (you can
see the program recompute electron wavelength, interaction constant, etc.
for each bin in the run log). This is stored as the `mLPNH`/`mLPSH`
datasets with shape `(numset, numEbins, 2·npx+1, 2·npx+1)` —
`(1, 11, 1001, 1001)` for the hi-res run.

`view_master_energy_bins.py` plots all 11 bins side by side (same
brightness scale) as `Ni_master_energybins.png`. Two things stand out:

- **Same geometric pattern in every panel** — the Kikuchi band symmetry
  only depends on the crystal lattice, not on energy, so all 11 panels
  show the same underlying diffraction geometry.
- **Very different brightness/sharpness per panel** — 10 keV is nearly
  black and noisy, 20 keV is bright and sharp. This reflects electron
  *statistics*, not diffraction physics: electrons exiting near the full
  20 keV barely scattered before escaping (common, strong signal), while
  electrons exiting at only 10 keV had to lose half their energy through
  repeated inelastic scattering first (rare, weak signal).

This is exactly why the energy dimension is kept separate instead of
collapsed: [`03_ebsd_pattern_simulation`](03_ebsd_pattern_simulation.md)
recombines these 11 spheres weighted by the real per-energy electron
counts from `Ni_MC.h5`, not by treating every energy as equally important.

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

Same program (`EMEBSDmaster`), run twice with two different `.nml` config
files — not two different programs. The difference is just
resolution/precision settings:

| | `EMEBSDmaster.nml` | `EMEBSDmaster_hires.nml` |
|---|---|---|
| Output | `Ni_master.h5` | `Ni_master_hires.h5` |
| `npx` | 100 → 201×201 pixel sphere | 500 → 1001×1001 pixel sphere |
| `dmin` | 0.05 nm | 0.03 nm (finer lattice planes included) |
| `nthreads` | 0 (auto) | 16 (explicit) |
| Compute cost | low | much higher (more pixels × more reflections) |
| Used downstream? | No — kept only for comparison | **Yes** — steps 03, 04, 08 all read this one |

Why keep the low-res one at all: it's cheap enough to eyeball quickly, e.g.
to sanity-check that the crystal/Bethe/MC setup is working before
committing to the expensive hi-res run. But it's not fine-grained enough
for actual pattern simulation — that's literally why the hi-res version
exists: the standard resolution wasn't fine enough for the actual pattern
simulations that follow.

Practically: you only need to run `EMEBSDmaster_hires.nml`. The plain
`EMEBSDmaster.nml` is optional/historical, not a dependency of anything
later in your pipeline.

`EMEBSDmaster.template` is a blank, generic reference copy — not read by
anything.

## Running it

CPU/OpenMP only — no GPU needed (`EMEBSDmaster` has no `platid`/`devid`,
just `nthreads`). Only the hi-res run matters in practice:

```bash
sbatch submit_master.sh
```

Takes noticeably longer than the Monte Carlo step — full dynamical
diffraction over a much finer sphere, on CPU rather than GPU. Run it as a
SLURM job (`--cpus-per-task=16`, no `--gres=gpu`) rather than on the login
node if it takes more than a couple of minutes.

## Visual check

`view_master.py` opens `Ni_master_hires.h5`, plots it, and saves
`Ni_master.png` — a sanity-check image of the diffraction sphere before any
detector geometry is applied to it.
