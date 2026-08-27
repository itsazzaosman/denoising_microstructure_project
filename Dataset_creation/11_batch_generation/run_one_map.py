#!/usr/bin/env python3
"""
Runs EMEBSD + EMEBSDDI for one map, in either its clean or noisy variant.

Called as:
    python3 run_one_map.py <index> <clean|noisy>

Per-map flow: build a per-map EMEBSD.nml pointing at that map's converted
angle file -> run EMEBSD -> build a per-map EMEBSDDI.nml pointing at the
patterns EMEBSD just wrote -> run EMEBSDDI -> delete the raw pattern file.
Only the final indexed .h5/.ang land in 11_batch_generation/indexed/<variant>/
- the raw simulated patterns are deleted right after indexing (orientation-
maps-only storage decision).

The two variants use different EMsoft nml templates (different beam current /
dwell time / Poisson-noise settings - see docs/04_dictionary_indexing.md):
    clean -> 03_ebsd_pattern_simulation/EMEBSD_ni.nml       + 04_dictionary_indexing/EMEBSDDI.nml
    noisy -> 03_ebsd_pattern_simulation/EMEBSD_noise_ni.nml + 04_dictionary_indexing/EMEBSDDI_noise.nml

Resumable: a map/variant whose indexed .ang already exists is skipped.

Assumes PATH/OCL_ICD_VENDORS are already set by the caller (see
submit_batch_indexing.sh) - this script just invokes EMEBSD/EMEBSDDI by name.
"""

import os
import re
import subprocess
import sys

DATASET_ROOT = "/project/community/aiosman/Dataset_creation"

TEMPLATES = {
    "clean": {
        "emebsd": os.path.join(DATASET_ROOT, "03_ebsd_pattern_simulation", "EMEBSD_ni.nml"),
        "emebsddi": os.path.join(DATASET_ROOT, "04_dictionary_indexing", "EMEBSDDI.nml"),
    },
    "noisy": {
        "emebsd": os.path.join(DATASET_ROOT, "03_ebsd_pattern_simulation", "EMEBSD_noise_ni.nml"),
        "emebsddi": os.path.join(DATASET_ROOT, "04_dictionary_indexing", "EMEBSDDI_noise.nml"),
    },
}

ANGLES_DIR = "11_batch_generation/angles"
PATTERNS_DIR = "11_batch_generation/patterns"
INDEXED_DIR = "11_batch_generation/indexed"
NML_TMP_DIR = "11_batch_generation/_nml_tmp"

IPF_WD = 128
IPF_HT = 128


def set_nml_param(lines, key, value_str):
    """Replace the value of `key = ...,` in a Fortran namelist's lines, in place."""
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$")
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = pattern.sub(rf"\g<1>{value_str}\n", line)
            return
    raise KeyError(f"Parameter '{key}' not found in nml template")


def write_nml(template_path, overrides, out_path):
    with open(template_path) as f:
        lines = f.readlines()
    for key, value_str in overrides.items():
        set_nml_param(lines, key, value_str)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)


def run_emsoft(binary_name, nml_path):
    return subprocess.run(
        [binary_name, nml_path],
        cwd=DATASET_ROOT,
        capture_output=True,
        text=True,
    )


def run_one(index, variant):
    idx = f"{index:05d}"
    templates = TEMPLATES[variant]

    angles_file_abs = os.path.join(DATASET_ROOT, ANGLES_DIR, f"angles_{idx}.txt")
    angles_file_rel = f"{ANGLES_DIR}/angles_{idx}.txt"
    pattern_file_rel = f"{PATTERNS_DIR}/{variant}/sim_{idx}.h5"
    pattern_file_abs = os.path.join(DATASET_ROOT, pattern_file_rel)
    indexed_h5_rel = f"{INDEXED_DIR}/{variant}/indexed_{idx}.h5"
    indexed_ang_rel = f"{INDEXED_DIR}/{variant}/indexed_{idx}.ang"
    indexed_ang_abs = os.path.join(DATASET_ROOT, indexed_ang_rel)

    if os.path.exists(indexed_ang_abs):
        print(f"[{variant} {idx}] already indexed, skipping")
        return True

    if not os.path.exists(angles_file_abs):
        print(f"[{variant} {idx}] no angle file at {angles_file_abs}")
        return False

    os.makedirs(os.path.join(DATASET_ROOT, PATTERNS_DIR, variant), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, INDEXED_DIR, variant), exist_ok=True)

    # 1. Simulate patterns for this map
    emebsd_nml = os.path.join(DATASET_ROOT, NML_TMP_DIR, f"EMEBSD_{variant}_{idx}.nml")
    write_nml(
        templates["emebsd"],
        {
            "anglefile": f"'{angles_file_rel}',",
            "datafile": f"'{pattern_file_rel}',",
        },
        emebsd_nml,
    )
    result = run_emsoft("EMEBSD", emebsd_nml)
    if result.returncode != 0 or not os.path.exists(pattern_file_abs):
        print(f"[{variant} {idx}] EMEBSD FAILED\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}")
        return False

    # 2. Dictionary-index those patterns
    emebsddi_nml = os.path.join(DATASET_ROOT, NML_TMP_DIR, f"EMEBSDDI_{variant}_{idx}.nml")
    write_nml(
        templates["emebsddi"],
        {
            "ipf_wd": f"{IPF_WD},",
            "ipf_ht": f"{IPF_HT},",
            "exptfile": f"'{pattern_file_rel}',",
            "datafile": f"'{indexed_h5_rel}',",
            "angfile": f"'{indexed_ang_rel}',",
        },
        emebsddi_nml,
    )
    result = run_emsoft("EMEBSDDI", emebsddi_nml)
    if result.returncode != 0 or not os.path.exists(indexed_ang_abs):
        print(f"[{variant} {idx}] EMEBSDDI FAILED\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}")
        return False

    # 3. Raw pattern file no longer needed - orientation maps only
    os.remove(pattern_file_abs)

    # Temp nml files no longer needed
    os.remove(emebsd_nml)
    os.remove(emebsddi_nml)

    print(f"[{variant} {idx}] done")
    return True


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("clean", "noisy"):
        sys.exit("usage: run_one_map.py <index> <clean|noisy>")

    index = int(sys.argv[1])
    variant = sys.argv[2]
    ok = run_one(index, variant)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
