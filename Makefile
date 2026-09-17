BUCKET := gs://cmu-gpucloud-aiosman

# ---- orientation-noise denoising experiment -------------------------------
# PROJ  : the experiment folder
# MAP   : which map to look at, e.g. `make score MAP=00002`
# SHAPE : grid size of the map, rows then columns
PROJ  := MAPS_DENOISING_Orientation_Noise
MAP   := 00001
SHAPE := 128 128

# The scoring scripts run on plain numpy, but the torch utilities need the
# `ebsd` env -- base has no torch.
EBSD_PY := $(HOME)/miniconda3/envs/ebsd/bin/python

CLEAN := $(PROJ)/datasets/clean_euler/map_$(MAP)_clean_euler.txt
NOISY := $(PROJ)/datasets/noisy_mis05/map_$(MAP)_clean_euler_noisy.txt
MASK  := $(NOISY).badmask.npy


.PHONY: auth ls upload download backup score score-all plot maps check-orientation \
        cache train-cpu train-gpu infer watch

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/

# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install 
upload:
	gcloud storage rsync -r ./training $(BUCKET)/training
	
# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install
download:
	gcloud storage rsync -r $(BUCKET)/training ./training

# Print the numbers: how far each denoised map in mtex_out/ sits from the
# truth. Use this when you want to RANK the filters -- it gives the median
# error per filter, split out over the misindexed pixels (--mask) and the
# grain boundaries, so you can see whether a filter wins overall or only in
# the grain interiors. No pictures, just a table you can paste into notes.
score:
	python $(PROJ)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(PROJ)/mtex_out/ \
	  --shape  $(SHAPE)

# Draw the pictures into figures/: maps_ipf.png (IPF-Z colour maps, one panel
# per filter, the familiar view) and maps_error.png (per-pixel error on a log
# colour scale, where leftover misindexed pixels glow). Use this when you want
# to SEE what a filter did to the grain boundaries and the speckle, rather
# than just read a median. Note it takes no --mask.
plot:
	python $(PROJ)/src/plot_maps.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --results $(PROJ)/mtex_out/ \
	  --shape  $(SHAPE) \
	  --out    $(PROJ)/figures/

# The same table, but aggregated over every map MTEX has finished, not just
# one. Use this for any claim you intend to defend: single-map medians wobble
# between maps, so the (IQR) column tells you whether a gap between two methods
# is real. This is the number the model has to beat.
score-all:
	python $(PROJ)/src/score_dataset.py \
	  --clean-dir $(PROJ)/datasets/clean_euler \
	  --noisy-dir $(PROJ)/datasets/noisy_mis05 \
	  --results   $(PROJ)/mtex_out/ \
	  --shape     $(SHAPE) \
	  --jobs      8 \
	  --csv       $(PROJ)/figures/scores.csv

# Check the orientation maths the model will be built on against the numpy
# reference. Run it after touching anything in orientation_torch.py.
check-orientation:
	$(EBSD_PY) $(PROJ)/src/orientation_torch.py --self-test

# Pack the clean maps into one memmappable .npy. Run once; takes about a
# minute. Training reads this instead of re-parsing 12k text files per epoch.
cache:
	$(EBSD_PY) $(PROJ)/src/prepare_cache.py --n-train 12000 --jobs 16

# Submit training. The CPU variant exists because every GPU on this cluster was
# allocated with ~470 jobs queued while 330 CPU cores sat idle; at base 32 it
# is slow but not unreasonably so. They write to different checkpoint dirs
# (checkpoints/ and checkpoints_gpu/) so both can run at once.
train-cpu:
	cd $(PROJ)/src && sbatch train_model_cpu.sh --resume auto

train-gpu:
	cd $(PROJ)/src && sbatch --job-name=ebsd_orient_gpu --partition=preempt \
	  --qos=preempt_qos --time=2-00:00:00 train_model.sh --resume auto

# Predict the held-out maps and drop the results into mtex_out/ as
# <stem>__unet.txt, so score-all compares them against the filters using the
# same code path. Both job scripts already do this when training ends.
infer:
	$(EBSD_PY) $(PROJ)/src/inference.py \
	  --checkpoint $(PROJ)/checkpoints/best.pth

watch:
	squeue -u $(USER) -o "%.10i %.10P %.20j %.3t %.10M %.6C %.22R"

# Both of the single-map targets, the usual thing to run after a new MTEX sweep.
maps: score plot
