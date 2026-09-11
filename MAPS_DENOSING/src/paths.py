"""Every path in the project, defined once.

Nothing else in src/ hardcodes a directory - import from here instead. If you
move the project or rename a folder, this is the only file to edit.

Layout:

    MAPS_DENOSING/
    |-- datasets/
    |   |-- ni_clean_maps/       ground truth maps
    |   `-- ni_g_noisy_maps/     Gaussian-noised inputs
    |-- checkpoints/             model weights (.pth) + history.json
    |-- visualize/
    |   |-- curves/              loss / PSNR / SSIM plots
    |   |-- comparisons/         clean | noisy | denoised figures
    |   `-- grain_analysis/      grain segmentation + boundary accuracy
    `-- src/                     the code
"""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

# --- inputs ---------------------------------------------------------------
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
CLEAN_DIR = os.path.join(DATASETS_DIR, "ni_clean_maps")
NOISY_DIR = os.path.join(DATASETS_DIR, "ni_g_noisy_maps")

# --- model weights --------------------------------------------------------
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")
BEST_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "best.pth")   # highest val PSNR
LAST_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "last.pth")   # most recent epoch
HISTORY_JSON = os.path.join(CHECKPOINTS_DIR, "history.json")  # per-epoch metrics

# --- visual output --------------------------------------------------------
VISUALIZE_DIR = os.path.join(PROJECT_DIR, "visualize")
CURVES_DIR = os.path.join(VISUALIZE_DIR, "curves")
COMPARISONS_DIR = os.path.join(VISUALIZE_DIR, "comparisons")
GRAIN_ANALYSIS_DIR = os.path.join(VISUALIZE_DIR, "grain_analysis")


def ensure_dirs(*dirs):
    """Create the given directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
