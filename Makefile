BUCKET := gs://cmu-gpucloud-aiosman

# ---- orientation-noise denoising experiment -------------------------------
# PROJ  : the experiment folder
# MAP   : which map to look at, e.g. `make score MAP=00002`
# SHAPE : grid size of the map, rows then columns
# NOISE : which noise set to use, e.g. `make score-all NOISE=scatter7`. It picks
#         the noisy maps (datasets/noise_<NOISE>), the MTEX + model outputs
#         (datasets/mtex_out_<NOISE>) and the model (checkpoints/checkpoints_<NOISE>).
#         Its settings live in configs/noise/<NOISE>.yaml.
# RUN   : a training experiment, configs/train/<RUN>.yaml, whose model is in
#         checkpoints/runs/<RUN>. Without it, `infer` uses the old per-level
#         model in checkpoints/checkpoints_<NOISE>.
PROJ  := MAPS_DENOISING_Orientation_Noise
MAP   := 00001
SHAPE := 128 128
NOISE := scatter5
RUN   :=

# The scoring scripts run on plain numpy, but the torch utilities need the
# `ebsd` env -- base has no torch.
EBSD_PY := $(HOME)/miniconda3/envs/ebsd/bin/python

CLEAN_DIR := $(PROJ)/datasets/clean_euler
NOISY_DIR := $(PROJ)/datasets/noise_$(NOISE)
RESULTS   := $(PROJ)/datasets/mtex_out_$(NOISE)
CKPT      := $(if $(RUN),$(PROJ)/checkpoints/runs/$(RUN)/best.pth,$(PROJ)/checkpoints/checkpoints_$(NOISE)/best.pth)
FIGURES   := $(PROJ)/figures/$(NOISE)

CLEAN := $(CLEAN_DIR)/map_$(MAP)_clean_euler.txt
NOISY := $(NOISY_DIR)/map_$(MAP)_clean_euler_noisy.txt
MASK  := $(NOISY).badmask.npy


.PHONY: auth ls upload download backup score score-all plot maps check-orientation \
        cache noise train-cpu train-gpu infer watch

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/
ls-training:
	gcloud storage ls $(BUCKET)/training/

ls-datasets:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/datasets/
ls-checkpoints:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/checkpoints/

# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install 
upload:
	gcloud storage rsync -r ./DATASET_CHECKPOINTS/datasets $(BUCKET)/DATASET_CHECKPOINTS/datasets
	
# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install
download:
	gcloud storage rsync -r $(BUCKET)/DATASET_CHECKPOINTS/datasets ./DATASET_CHECKPOINTS/datasets

# Print the numbers: how far each denoised map in $(RESULTS) sits from the
# truth. Use this when you want to RANK the filters -- it gives the median
# error per filter, split out over the misindexed pixels (--mask) and the
# grain boundaries, so you can see whether a filter wins overall or only in
# the grain interiors. No pictures, just a table you can paste into notes.
score:
	python $(PROJ)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(RESULTS)/ \
	  --shape  $(SHAPE)

# Draw the pictures into figures/: maps_ipf.png (IPF-Z colour maps, one panel
# per filter, the familiar view) and maps_error.png (per-pixel error on a log
# colour scale, where leftover misindexed pixels glow). Use this when you want
# to SEE what a filter did to the grain boundaries and the speckle, rather
# than just read a median. Note it takes no --mask.
plot:
	mkdir -p $(FIGURES)
	python $(PROJ)/src/plot_maps.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --results $(RESULTS)/ \
	  --shape  $(SHAPE) \
	  --out    $(FIGURES)/

# The same table, but aggregated over every map MTEX has finished, not just
# one. Use this for any claim you intend to defend: single-map medians wobble
# between maps, so the (IQR) column tells you whether a gap between two methods
# is real. This is the number the model has to beat.
# The CSV is named scores_median.csv so it does not overwrite the older,
# mean-based figures/scatter5/scores.csv.
score-all:
	mkdir -p $(FIGURES)
	python $(PROJ)/src/score_dataset.py \
	  --clean-dir $(CLEAN_DIR) \
	  --noisy-dir $(NOISY_DIR) \
	  --results   $(RESULTS)/ \
	  --shape     $(SHAPE) \
	  --jobs      8 \
	  --csv       $(FIGURES)/scores_median.csv

# Check the orientation maths the model will be built on against the numpy
# reference. Run it after touching anything in orientation_torch.py.
check-orientation:
	$(EBSD_PY) $(PROJ)/src/orientation_torch.py --self-test

# Pack the clean maps into one memmappable .npy. Run once; takes about a
# minute. Training reads this instead of re-parsing 12k text files per epoch.
cache:
	$(EBSD_PY) $(PROJ)/src/prepare_cache.py --n-train 12000 --jobs 16

# Build a test set from its experiment file, configs/noise/$(NOISE).yaml ->
# datasets/noise_$(NOISE). Maps that already exist are skipped, so this also
# resumes an interrupted run. Each run is logged in provenance.json there.
noise:
	python $(PROJ)/src/add_ebsd_noise.py --config $(PROJ)/configs/noise/$(NOISE).yaml \
	  $(CLEAN_DIR) $(NOISY_DIR)

# Submit training: `make train-gpu CONFIG=scatter7` trains the experiment in
# configs/train/scatter7.yaml (make passes CONFIG on to the job). The CPU variant exists because every GPU on this cluster was
# allocated with ~470 jobs queued while 330 CPU cores sat idle; at base 32 it
# is slow but not unreasonably so. They write to different checkpoint dirs
# (checkpoints/ and checkpoints_gpu/) so both can run at once.
train-cpu:
	@echo "Training runs on a GPU only (decided 2026-10-12). Use: make train-gpu CONFIG=<name>"; exit 1

train-gpu:
	cd $(PROJ)/src && sbatch --job-name=$(or $(CONFIG),ebsd_orient_gpu) --partition=preempt \
	  --qos=preempt_qos --time=2-00:00:00 train_model.sh --resume auto

# Predict the held-out maps and drop the results into $(RESULTS) as
# <stem>__unet.txt, so score-all compares them against the filters using the
# same code path. Both job scripts already do this when training ends.
infer:
	$(EBSD_PY) $(PROJ)/src/inference.py \
	  --checkpoint $(CKPT) \
	  --clean-dir  $(CLEAN_DIR) \
	  --noisy-dir  $(NOISY_DIR) \
	  --out        $(RESULTS) \
	  --method-name $(or $(RUN),unet)

watch:
	squeue -u $(USER) -o "%.10i %.10P %.20j %.3t %.10M %.6C %.22R"

# Both of the single-map targets, the usual thing to run after a new MTEX sweep.
maps: score plot

# To see what is in the onedrive folder, run this command. It is a bit slow
# rclone lsf "onedrive:Desktop/Research Project - Material Science + Diffusion Models/Pipline For Pictures and Euler Angles Only/"
# if you want to download the whole thing, run this command. It is very slow.
# rclone copyto "onedrive:Desktop/Research Project - Material Science + Diffusion Models/Pipline For Pictures and Euler Angles Only/misorientation_noise_dataset_5deg/mag5/clean_euler.tar" "/project/community/aiosman/MAPS_DENOISING_Orientation_Noise/datasets/clean_euler.tar" --progress


# First run this add_ebsd_noise.py script to generate the noisy dataset that we use for MTEX:
# python src/add_ebsd_noise.py \
  datasets/clean_euler datasets/noise_scatter5 \
  --shape 128 128 --limit 500 \
  --scatter 5.0 --misindex 0 --save-mask --ang

# cd /project/community/aiosman/MAPS_DENOISING_Orientation_Noise

# # all 10 maps at 5° scatter
# python src/plot_mtex_maps.py --scatter 5 --all

# # all 10 maps at all three noise levels
# for s in 5 7 9; do python src/plot_mtex_maps.py --scatter $s --all; done

# # or just one map at a time
# python src/plot_mtex_maps.py --scatter 5 --maps 3

# # optional: zoom into a 48×48 region (rows 40–88, columns 40–88)
# python src/plot_mtex_maps.py --scatter 5 --maps 3 --crop 40 40 48 48


#python src/plot_mtex_maps.py --ang-dir datasets/noise_scatter9/ang_files \
  --results datasets/mtex_out_scatter9_nograins --out figures/mtex_maps/scatter9_nograins --all
