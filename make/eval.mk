.PHONY: score plot score-all score-grid maps

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


score-all:
	mkdir -p $(FIGURES)
	python $(SHARED_DIR)/src/score_dataset.py \
	  --clean-dir $(CLEAN_DIR) \
	  --noisy-dir $(NOISY_DIR) \
	  --results   $(RESULTS) \
	  --shape     $(SHAPE) \
	  --jobs      8 \
	  --csv       $(FIGURES)/scores_mean.csv


# Score the model's test outputs against the MTEX filters, one table per set.
# CPU only (about 1 s per set); prints tables and saves no files. Only maps that every
# method has are scored (MTEX ran on maps 1-10), so the columns compare fairly.
#   make score-grid                      all 28 sets, model in results_test_set_model/$(SEED)
#   make score-grid SETS=scatter5_mis3   one set (several: SETS="scatter5 scatter9_mis6")
score-grid:
	@for S in $(or $(SETS),$$(cat $(SHARED_DIR)/configs/test_grid.txt)); do \
	  M=$(PROJ)/datasets/results_test_set_model/$(SEED)/$$S; \
	  echo; echo "===== $$S ====="; \
	  [ -d $$M ] || { echo "no model results in $$M (run make infer first)"; continue; }; \
	  python $(SHARED_DIR)/src/score_dataset.py --clean-dir $(CLEAN_DIR) \
	    --noisy-dir $(TEST_SETS)/noise_$$S --results $$M $(PROJ)/datasets/results_test_set_mtex/$$S \
	    --shape $(SHAPE) --jobs 1 | sed -n '/^method /,/^====/p'; \
	done


maps: score plot

