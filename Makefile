BUCKET := gs://cmu-gpucloud-aiosman

# ---- orientation-noise denoising experiment -------------------------------
# PROJ  : the experiment folder
# MAP   : which map to look at, e.g. `make score MAP=00002`
# SHAPE : grid size of the map, rows then columns
# NOISE : which test set `make score` / `make plot-maps` use, e.g. `make plot-maps NOISE=scatter7_mis3`.
#         It picks the noisy maps (datasets/noise_scatter_misindexing/noise_<NOISE>), the
#         model outputs (datasets/results_test_set_model/<SEED>/<NOISE>) and the MTEX outputs
#         (datasets/results_test_set_mtex/<NOISE>). `make noise NOISE=<set>` makes a set
#         from SHARED/configs/noise/<NOISE>.yaml; a val_ name is saved in
#         datasets/noise_scatter_misindexing_val/noise_<NOISE> instead.
# SEED  : which model `make score-grid`, `score` and the plot-* targets use: the folders
#         datasets/results_test_set_model/<SEED> and MODEL_PIPELINE/checkpoints/<SEED>
# SETS  : only these test sets for `make infer` / `make score-grid`, e.g. SETS=scatter5_mis3
# NAME  : the model's name for `make infer` (file names, score tables); empty = config name
# SCATTER, MISINDEX : noise levels for `make preview` (deg, %); several allowed,
#         e.g. `make preview SCATTER="1 5 9" MISINDEX="0 9 15"`
PROJ  := MAPS_DENOISING_Orientation_Noise
# the three code folders of the project
MODEL_DIR  := $(PROJ)/MODEL_PIPELINE
MTEX_DIR   := $(PROJ)/MTEX_PIPELINE
SHARED_DIR := $(PROJ)/SHARED
MAP   := 00001
SHAPE := 128 128
NOISE := scatter5
SEED  := seed1
SETS  :=
NAME  :=
SCATTER  := 5
MISINDEX := 3

# The scoring scripts run on plain numpy, but the torch utilities need the
# `ebsd` env -- base has no torch.
EBSD_PY := $(HOME)/miniconda3/envs/ebsd/bin/python

CLEAN_DIR := $(PROJ)/datasets/clean_euler_zip_unzip/clean_euler
TEST_SETS := $(PROJ)/datasets/noise_scatter_misindexing
VAL_SETS  := $(PROJ)/datasets/noise_scatter_misindexing_val
# A set whose name starts with val_ is a validation set; every other set is a test set.
NOISY_DIR := $(if $(filter val_%,$(NOISE)),$(VAL_SETS),$(TEST_SETS))/noise_$(NOISE)
MODEL_OUT := $(PROJ)/datasets/results_test_set_model/$(SEED)/$(NOISE)
MTEX_OUT  := $(PROJ)/datasets/results_test_set_mtex/$(NOISE)
RESULTS   := $(MODEL_OUT) $(MTEX_OUT)

CLEAN := $(CLEAN_DIR)/map_$(MAP)_clean_euler.txt
NOISY := $(NOISY_DIR)/map_$(MAP)_clean_euler_noisy.txt
MASK  := $(NOISY).badmask.npy

# ---- targets, one file per job --------------------------------------------
# make/cloud.mk  auth, ls, ls-training, ls-datasets, ls-checkpoints, upload, download
# make/data.mk   noise, preview, cache, check-orientation
# make/train.mk  train-gpu, train-cpu, watch
# make/infer.mk  infer       (GPU job: model -> denoised test maps)
# make/score.mk  score-grid  (CPU: model vs MTEX tables, test grid)
# make/eval.mk   score       (one map of one set)
# make/plot.mk   plot-train, plot-zoom, plot-test, plot-maps  (every figure, plotting.py)
# The settings above must stay above this include line, so every file sees them.
# Plain `make` runs this.
.DEFAULT_GOAL := ls
include make/cloud.mk make/data.mk make/train.mk make/infer.mk make/score.mk make/eval.mk make/plot.mk
