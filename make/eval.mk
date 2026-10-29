.PHONY: score plot maps

# Look at ONE map of one test set (model results from results_test_set_model/$(SEED)):
#   make score MAP=00002 NOISE=scatter5_mis3   print each method's error on that map
#   make plot  MAP=00002 NOISE=scatter5_mis3   draw clean / noisy / denoised maps into
#                                              SHARED/figures/<SEED>/<NOISE>/map_<MAP>/
#   make maps  MAP=00002 NOISE=scatter5_mis3   both
score:
	python $(SHARED_DIR)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE)

plot:
	mkdir -p $(FIGURES)
	python $(SHARED_DIR)/src/plot_maps.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE) \
	  --out    $(FIGURES)/


maps: score plot

