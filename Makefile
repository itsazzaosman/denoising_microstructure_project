BUCKET := gs://cmu-gpucloud-aiosman

# ---- orientation-noise denoising experiment -------------------------------
# PROJ  : the experiment folder
# MAP   : which map to look at, e.g. `make score MAP=00002`
# SHAPE : grid size of the map, rows then columns
# NOISE : which noise set to use, e.g. `make score-all NOISE=scatter7`. It picks
#         the noisy maps (datasets/noise_scatter_misindexing/noise_<NOISE>), the model outputs
#         (datasets/results_test_set_model/<NOISE>), the MTEX outputs
#         (datasets/results_test_set_mtex/<NOISE>) and the model (checkpoints/checkpoints_<NOISE>).
#         Its settings live in configs/noise/<NOISE>.yaml.
# RUN   : a training experiment, configs/train/<RUN>.yaml, whose model is in
#         checkpoints/runs/<RUN>. Without it, `infer` uses the old per-level
#         model in checkpoints/checkpoints_<NOISE>.
# SCATTER, MISINDEX : noise levels for `make preview` (deg, %); several allowed,
#         e.g. `make preview SCATTER="1 5 9" MISINDEX="0 9 15"`
PROJ  := MAPS_DENOISING_Orientation_Noise
MAP   := 00001
SHAPE := 128 128
NOISE := scatter5
RUN   :=
SCATTER  := 5
MISINDEX := 3

# The scoring scripts run on plain numpy, but the torch utilities need the
# `ebsd` env -- base has no torch.
EBSD_PY := $(HOME)/miniconda3/envs/ebsd/bin/python

CLEAN_DIR := $(PROJ)/datasets/clean_euler_zip_unzip/clean_euler
TEST_SETS := $(PROJ)/datasets/noise_scatter_misindexing
NOISY_DIR := $(TEST_SETS)/noise_$(NOISE)
MODEL_OUT := $(PROJ)/datasets/results_test_set_model/$(NOISE)
MTEX_OUT  := $(PROJ)/datasets/results_test_set_mtex/$(NOISE)
RESULTS   := $(MODEL_OUT) $(MTEX_OUT)
CKPT      := $(if $(RUN),$(PROJ)/checkpoints/runs/$(RUN)/best.pth,$(PROJ)/checkpoints/checkpoints_$(NOISE)/best.pth)
FIGURES   := $(PROJ)/figures/$(NOISE)

CLEAN := $(CLEAN_DIR)/map_$(MAP)_clean_euler.txt
NOISY := $(NOISY_DIR)/map_$(MAP)_clean_euler_noisy.txt
MASK  := $(NOISY).badmask.npy

# ---- targets, one file per job --------------------------------------------
# make/cloud.mk  auth, ls, ls-training, ls-datasets, ls-checkpoints, upload, download
# make/data.mk   noise, preview, cache, check-orientation
# make/train.mk  train-gpu, train-cpu, infer, watch
# make/eval.mk   score, plot, score-all, score-grid, maps
# The settings above must stay above this include line, so every file sees them.
# Plain `make` runs this.
.DEFAULT_GOAL := ls
include make/cloud.mk make/data.mk make/train.mk make/eval.mk
