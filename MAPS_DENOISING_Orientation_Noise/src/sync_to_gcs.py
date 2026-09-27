from datetime import datetime
from pathlib import Path
import fcntl
import subprocess
import sys

GCLOUD = "/usr/bin/gcloud"

PROJECT = Path(
    "/project/community/aiosman/MAPS_DENOISING_Orientation_Noise"
)

DESTINATION = "gs://cmu-gpucloud-aiosman/DATASET_CHECKPOINTS"

FOLDERS = ("datasets", "checkpoints")


def log(message):
    timestamp = datetime.now().astimezone().isoformat()
    print(f"[{timestamp}] {message}", flush=True)


def main():
    dry_run = "--dry-run" in sys.argv[1:]

    log_dir = Path.home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Keep the lock until both uploads finish.
    with (log_dir / "dataset_checkpoints.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another sync is running. Skipping this run.")
            return

        log("Starting preview." if dry_run else "Starting upload.")
        failed = False

        for folder in FOLDERS:
            source = PROJECT / folder
            destination = f"{DESTINATION}/{folder}"

            if not source.is_dir():
                log(f"ERROR: Source folder does not exist: {source}")
                failed = True
                continue

            log(f"Syncing {source} -> {destination}")

            command = [
                GCLOUD,
                "storage",
                "rsync",
                str(source),
                destination,
                "--recursive",
                "--quiet",
            ]

            if dry_run:
                command.append("--dry-run")

            # No deletion flag: files deleted locally remain in the bucket.
            result = subprocess.run(command)

            if result.returncode != 0:
                log(f"ERROR: Sync failed for {folder}.")
                failed = True
            else:
                log(f"Finished: {folder}")

        if failed:
            log("Sync finished with errors.")
            raise SystemExit(1)

        log("All folders synced successfully.")


if __name__ == "__main__":
    main()