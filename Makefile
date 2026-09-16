BUCKET := gs://cmu-gpucloud-aiosman

# ---- orientation-noise denoising experiment -------------------------------
# PROJ  : the experiment folder
# MAP   : which map to look at, e.g. `make score MAP=00002`
# SHAPE : grid size of the map, rows then columns
PROJ  := MAPS_DENOISING_Orientation_Noise
MAP   := 00001
SHAPE := 128 128

CLEAN := $(PROJ)/datasets/clean_euler/map_$(MAP)_clean_euler.txt
NOISY := $(PROJ)/datasets/noisy_mis05/map_$(MAP)_clean_euler_noisy.txt
MASK  := $(NOISY).badmask.npy


.PHONY: auth ls upload download backup score plot maps

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

# Both of the above, the usual thing to run after a new MTEX sweep.
maps: score plot
