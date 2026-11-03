.PHONY: score-grid score

# make score-grid SEED=seed1
score-grid:
	@for S in $(or $(SETS),$$(cat $(SHARED_DIR)/configs/test_grid.txt)); do \
	  M=$(PROJ)/datasets/results_test_set_model/$(SEED)/$$S; \
	  echo; echo "===== $$S ====="; \
	  [ -d $$M ] || { echo "no model results in $$M (run make infer first)"; continue; }; \
	  python $(SHARED_DIR)/src/score_dataset.py --clean-dir $(CLEAN_DIR) \
	    --noisy-dir $(TEST_SETS)/noise_$$S --results $$M $(PROJ)/datasets/results_test_set_mtex/$$S \
	    --shape $(SHAPE) --jobs 1 | sed -n '/^method /,/^====/p'; \
	done

# make score MAP=00002 NOISE=scatter5_mis3
score:
	@test -n "$(MAP)" && test -n "$(NOISE)" || { echo "set MAP= and NOISE=, e.g. make score MAP=00002 NOISE=scatter5_mis3"; exit 1; }
	python $(SHARED_DIR)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE)

