# 00 — Crystal structure

**Role:** the starting point of the entire pipeline. Defines *what nickel is*
to every downstream simulation - just the raw material definition.

**Output:** `Ni.xtal` 

## What's in the file

`Ni.xtal` is an HDF5 file (EMsoft's own crystal-structure format) describing
face-centered-cubic (FCC) nickel:

- **Space group 225** (Fm-3m), point group **m-3m**
- **Lattice parameter a = 0.3524 nm** (3.524 Å) — cubic, so a = b = c and all
  angles are 90°
- Atom positions/occupancies for the FCC basis


