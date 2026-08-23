# 03 — EBSD pattern simulation

**Role:** project the orientation-independent master pattern onto a specific,
finite detector at specific crystal orientations — this is the step that
finally produces images that look like what a real EBSD camera records.

**Input:** [`02_master_pattern/Ni_master_hires.h5`](02_master_pattern.md) +
an orientation list (angle file)
**Output:** `Ni_EBSD_sim.h5` → consumed by
[`05_noising_training_data`](05_noising_training_data.md) and
[`06_euler_angle_validation`](06_euler_angle_validation.md)

## The detector geometry

A detector isn't just a pixel grid — its position relative to the sample
determines exactly which patch of the master-pattern sphere lands where.
The key parameters:

- **`L`** — distance from the sample to the detector (scintillator)
- **`thetac`** — camera tilt
- **`delta`** — physical size of one detector pixel
- **`xpc`, `ypc`** — the **pattern center**: where the sample-to-detector
  perpendicular actually hits the detector, in pixel units. This is *the*
  critical calibration number — get it wrong and every band sits in the
  wrong place, even with perfect physics upstream.
- **`numsx`, `numsy`** — detector pixel dimensions

## Two runs

| File | Detector | Orientations | Output |
|---|---|---|---|
| `EMEBSD.nml` | 640×480, PC centered (`xpc=ypc=0`) | 5 hand-written test angles (`angles.txt`) | `Ni_EBSD.h5` — smoke test |
| `EMEBSD_ni.nml` | 60×60, PC = `(4.337, 17.273, 1500.8)` matching the real camera | 10,000 orientations from a real Dream3D 100×100 synthetic microstructure (`Nickel_100x100_angles.txt`) | `Ni_EBSD_sim.h5` — **the one everything downstream uses** |

The pattern-center numbers in `EMEBSD_ni.nml` come from
[`06_euler_angle_validation`](06_euler_angle_validation.md) — this stage
can't run meaningfully until that one supplies them first. That dependency
is what makes 06 the one loop in an otherwise linear pipeline: it runs
*before* this stage (to supply inputs) and *after* it (to check the output)
— see that page for the full explanation.

**The angle file changed.** `anglefile` originally pointed at the
4,125-line `ni_angles.txt` (real-scan-derived orientations, also from 06).
It now points at `Nickel_100x100_angles.txt` instead: 10,000 orientations
from a real Dream3D-generated 100×100 grain map
(`Nickel_100x100_Euler.txt`), converted from Dream3D's radians/no-header
export into EMsoft's `eu` / count / degrees format by
`convert_dream3d_angles.py`. `ni_angles.txt` itself is untouched and still
sits in this folder — repoint `anglefile` back to it (and give `datafile`
its own name first, so you don't overwrite this run) if you want the
original real-scan-calibrated dataset back.

`EMEBSD.template` is a blank, generic reference copy — not read by anything.

## Running it

CPU/OpenMP only (`nthreads = 16` in the nml, auto-clamped down to however
many cores are actually available on whatever node you run it on). Fast —
the whole 10,000-pattern run takes about 2-3 seconds, so running it
directly is fine, no need to submit it as a job:

```bash
export PATH="/project/community/aiosman/emsoft_install/src/EMsoftBuild/Release/Bin:$PATH"
EMEBSD /project/community/aiosman/Dataset_creation/03_ebsd_pattern_simulation/EMEBSD_ni.nml
```

Produces `03_ebsd_pattern_simulation/Ni_EBSD_sim.h5` (10,000 × 60 × 60,
8-bit patterns).

## Visual check

`view_patterns.py` opens `Ni_EBSD_sim.h5`, picks 3 patterns from 3
different grains (by walking `Nickel_100x100_angles.txt` until the Euler
angles actually change, since Dream3D repeats one orientation across every
pixel inside a grain), and plots them — each panel titled with its real
`(row, col)` grid position and Euler angles — to `Ni_patterns.png`.

## A parallel run lives in Stage 08

[`08_synthetic_training_data`](08_synthetic_training_data.md) runs this
exact same simulation a second time, with its own copy of the config
(`EMEBSD_30k.nml`) pointed at 30,000 *synthetic* orientations instead of the
4,125 real ones — same master pattern, same detector geometry, different
angle file. That's a deliberate choice, not duplication: it's the fix for
the training-data-scarcity problem discovered in
[`07_training_pipeline`](07_training_pipeline.md).
