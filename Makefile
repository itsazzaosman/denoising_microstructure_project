BUCKET := gs://cmu-gpucloud-aiosman

PROJ  := MAPS_DENOISING_Orientation_Noise
# the code folders of the project
MODEL_DIR  := $(PROJ)/MODEL_PIPELINE
SHARED_DIR := $(PROJ)/SHARED
# MAP, NOISE, SCATTER and MISINDEX are empty on purpose: give them on the command line,
# e.g. make score MAP=00002 NOISE=scatter5_mis3
MAP   :=
SHAPE := 128 128
NOISE :=
SEED  := seed1
SETS  := #  Empty means "all 28 test sets".
NAME  := # Empty means "use the config name in the result files".
SCATTER  :=
MISINDEX :=

EBSD_PY := $(HOME)/miniconda3/envs/ebsd/bin/python

CLEAN_DIR := $(PROJ)/datasets/clean_euler_zip_unzip/clean_euler
TEST_SETS := $(PROJ)/datasets/noise_scatter_misindexing

NOISY_DIR := $(TEST_SETS)/noise_$(NOISE)
MODEL_OUT := $(PROJ)/datasets/results_test_set_model/$(SEED)/$(NOISE)
MTEX_OUT  := $(PROJ)/datasets/results_test_set_mtex/$(NOISE)
RESULTS   := $(MODEL_OUT) $(MTEX_OUT)

CLEAN := $(CLEAN_DIR)/map_$(MAP)_clean_euler.txt
NOISY := $(NOISY_DIR)/map_$(MAP)_clean_euler_noisy.txt
MASK  := $(NOISY).badmask.npy

.DEFAULT_GOAL := ls
include make/cloud.mk make/data.mk make/train.mk make/infer.mk make/score.mk make/plot.mk
