.PHONY: score

# Score ONE map of one test set (model results from results_test_set_model/$(SEED)):
#   make score MAP=00002 NOISE=scatter5_mis3   print each method's error on that map
# (its pictures: make plot-maps, in make/plot.mk)
score:
	python $(SHARED_DIR)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE)
