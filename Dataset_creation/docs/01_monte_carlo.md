# 01 — Monte Carlo electron simulation

**Role:** simulate how electrons scatter *inside* the nickel crystal, before
anything about a detector or a specific orientation exists yet.

**Input:** The input to EMMCOpenC is `MCNi.nml` by using [`00_crystal_structure/Ni.xtal`](00_crystal_structure.md)
**Output:** `Ni_MC.h5` 

## The physics, briefly

A beam of electrons hits the sample and immediately starts scattering —
elastically (direction changes, no energy lost "the total kE is conserved") and inelastically (energy
lost to the lattice) — many times before some fraction of them exit back out
of the surface. 

EMsoft's `EMMCOpenCL` program simulates this with a Monte
Carlo method: launch a huge number of individual virtual electrons, track
each one's random walk through the crystal step by step, and record the
statistics of **how they exit (and how many exited) — at what energy, at what angle**.

This step knows nothing yet about crystal orientation or a camera. It's
purely "how does this material scatter electrons of this energy 20keV, entering at
this tilt 70 degree"

## What was actually run

`MCNi.nml` (the config actually used — `EMMCOpenCL.template` alongside it is
a blank, generic reference copy, not read by anything):

| Parameter | Value | Meaning |
|---|---|---|
| `mode` | `'full'` | Full EBSD simulation (vs. `'bse1'`/ECP or `'Ivol'`) |
| `numsx` | 501 | Projection grid resolution |
| `sig` | 70.0° | Sample tilt from horizontal — the standard steep EBSD geometry |
| `EkeV` | 20.0 | Incident beam energy |
| `Ehistmin` | 10.0 | Minimum energy tracked in the exit-energy histogram |
| `totnum_el` | 1×10⁸ | Total incident electrons simulated |

Run with `EMMCOpenCL MCNi.nml`. Output field `dataname` points to
`01_monte_carlo/Ni_MC.h5` (a path relative to `EMdatapathname`, the repo
root — see the [crystal structure doc](00_crystal_structure.md) for why this
one *does* need the folder prefix while `xtalname` doesn't).

## What's in `Ni_MC.h5`

Not an image — a table of electron backscatter statistics: energy histogram
and exit-angle distribution. [`02_master_pattern`](02_master_pattern.md)'s
`EMEBSDmaster` program reads this back via `copyfromenergyfile` rather than
recomputing it, which is the entire point of splitting this into its own
stage: the expensive part (simulating 10⁸ electron trajectories) only has to
happen once, and can be reused across multiple master-pattern resolutions
(both `02_master_pattern`'s standard and hi-res runs read the *same*
`Ni_MC.h5`).
