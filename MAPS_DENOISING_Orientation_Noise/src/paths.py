"""
Where the data lives. Python scripts take their folders from here, so
reorganising datasets/ is a change to this file (and to the same two paths in
../../Makefile, eval_model.sh, train_model*.sh, import_mtex_results.sh and
run_mtex_grid.m, which cannot import it).

    datasets/
      clean_euler_zip_unzip/clean_euler/   ground truth: map_XXXXX_clean_euler.txt (40,000)
      noise_scatter_misindexing/noise_<set>/  the 29 noisy test sets (maps 1-500)
      results_test_set_model/<set>/        the network's predictions on each test set
      results_test_set_mtex/<set>/         the MTEX filters' outputs on each test set
      results_test_set_mtex_old/<set>/     MTEX outputs of September 2026 (not compared)
      splits/<split>/clean_euler/          links to the clean maps of each split
      splits/val/noise_random/             the fixed noisy validation set
      cache/                               packed training maps
      mtex_package/                        maps and script for MTEX on another computer
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"

CLEAN_DIR = DATA / "clean_euler_zip_unzip" / "clean_euler"
TEST_SETS = DATA / "noise_scatter_misindexing"
MODEL_RESULTS = DATA / "results_test_set_model"
MTEX_RESULTS = DATA / "results_test_set_mtex"
SPLITS = DATA / "splits"


def clean_map(map_id):
    """The ground-truth map with this id."""
    return CLEAN_DIR / f"map_{map_id:05d}_clean_euler.txt"


def noise_dir(test_set):
    """The folder of one noisy test set, e.g. noise_dir('scatter5_mis6')."""
    return TEST_SETS / f"noise_{test_set}"
