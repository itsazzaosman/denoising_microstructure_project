#!/usr/bin/env python3
"""
Runs on Windows, next to your DREAM3D install.

Loops DREAM3D's PipelineRunner.exe over Ni_pipeline_A_generate_microstructure.json,
generating NUM_MAPS distinct 128x128 synthetic Ni microstructures (Pipeline A only).

Each run gets its own Export ASCII Data output path so nothing overwrites the last
one. Resumable: a map whose output file already exists is skipped, so re-running
after a crash/interruption picks up where it left off instead of starting over.

Pipeline B (segmentation) is a separate script that runs later, once these Euler
angle files have been through EMEBSD/EMEBSDDI on the Linux cluster and the
resulting indexed .ang files have come back.

Before running at scale: confirm PipelineRunner's actual CLI flag with
    PipelineRunner.exe --help
and adjust RUNNER_ARGS below if it isn't "-p". Also do the back-to-back
same-JSON test described earlier to confirm Pack Primary Phases / Match
Crystallography really do randomize on their own each run (no Seed field
exists in this pipeline to force it otherwise).

Run with:
    python generate_microstructures_windows.py
"""

import json
import subprocess
import sys
from pathlib import Path

# ======================================================================
# SETTINGS - edit these paths for your machine before running
# ======================================================================
PIPELINE_RUNNER = Path(
    r"C:\Users\Ahmed Alhassan\Downloads\DREAM3D-6.5.171-Win64\DREAM3D-6.5.171-Win64\PipelineRunner.exe"
)
TEMPLATE_JSON = Path(
    r"C:\Users\Ahmed Alhassan\OneDrive\Desktop\Ni_pipeline_A_generate_microstructure.json"
)
OUTPUT_DIR = Path(r"C:\Users\Ahmed Alhassan\OneDrive\Desktop\Ni_dataset\euler_maps")
TEMP_PIPELINE_DIR = Path(r"C:\Users\Ahmed Alhassan\OneDrive\Desktop\Ni_dataset\_pipelines_tmp")
LOG_FILE = Path(r"C:\Users\Ahmed Alhassan\OneDrive\Desktop\Ni_dataset\generate_log.txt")

NUM_MAPS = 10_000
FILENAME_PATTERN = "Euler_{:05d}.txt"  # -> Euler_00001.txt ... Euler_10000.txt

EXPORT_FILTER_NAME = "WriteASCIIData"  # the "Export ASCII Data" step's internal name
RUNNER_ARGS = ["-p"]  # verify against `PipelineRunner.exe --help` first
TIMEOUT_SECONDS = 300
# ======================================================================


def load_template():
    with open(TEMPLATE_JSON, "r") as f:
        return json.load(f)


def set_output_path(pipeline, output_file):
    """Point the Export ASCII Data filter at a fresh output file for this run."""
    for filt in pipeline.values():
        if isinstance(filt, dict) and filt.get("Filter_Name") == EXPORT_FILTER_NAME:
            filt["OutputFilePath"] = str(output_file)
            filt["OutputPath"] = str(output_file.parent)
            return
    raise RuntimeError(f"No filter named '{EXPORT_FILTER_NAME}' found in the pipeline JSON")


def run_one(i):
    output_file = OUTPUT_DIR / FILENAME_PATTERN.format(i)
    if output_file.exists():
        return "skipped", output_file

    pipeline = load_template()
    set_output_path(pipeline, output_file)

    temp_json = TEMP_PIPELINE_DIR / f"pipeline_{i:05d}.json"
    with open(temp_json, "w") as f:
        json.dump(pipeline, f, indent=4)

    try:
        result = subprocess.run(
            [str(PIPELINE_RUNNER), *RUNNER_ARGS, str(temp_json)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        temp_json.unlink(missing_ok=True)
        return "failed", (-1, "", f"timed out after {TIMEOUT_SECONDS}s")

    temp_json.unlink(missing_ok=True)  # don't leave 10,000 temp pipeline files behind

    if result.returncode != 0 or not output_file.exists():
        return "failed", (result.returncode, result.stdout[-2000:], result.stderr[-2000:])

    return "ok", output_file


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    if not PIPELINE_RUNNER.exists():
        sys.exit(f"PipelineRunner.exe not found at {PIPELINE_RUNNER}")
    if not TEMPLATE_JSON.exists():
        sys.exit(f"Template pipeline not found at {TEMPLATE_JSON}")

    ok = skipped = failed = 0
    with open(LOG_FILE, "a") as log:
        for i in range(1, NUM_MAPS + 1):
            status, detail = run_one(i)

            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                returncode, stdout, stderr = detail
                msg = (
                    f"[FAILED] map {i:05d} (exit {returncode})\n"
                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}\n"
                )
                print(msg)
                log.write(msg + "\n")

            if i % 50 == 0 or i == NUM_MAPS:
                progress = f"{i}/{NUM_MAPS}  ok={ok} skipped={skipped} failed={failed}"
                print(progress)
                log.write(progress + "\n")
                log.flush()

    print(f"Done. ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
