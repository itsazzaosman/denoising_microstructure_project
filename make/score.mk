.PHONY: score-grid score

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

# Score ONE map of one test set (model results from results_test_set_model/$(SEED)):
#   make score MAP=00002 NOISE=scatter5_mis3   print each method's error on that map
score:
	@test -n "$(MAP)" && test -n "$(NOISE)" || { echo "set MAP= and NOISE=, e.g. make score MAP=00002 NOISE=scatter5_mis3"; exit 1; }
	python $(SHARED_DIR)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE)

