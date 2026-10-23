.PHONY: score plot score-all score-grid maps eval-val

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


score-grid:
	for S in $$(cat $(SHARED_DIR)/configs/test_grid.txt) test_random; do \
	  mkdir -p $(SHARED_DIR)/figures/$$S; \
	  python $(SHARED_DIR)/src/score_dataset.py --clean-dir $(CLEAN_DIR) \
	    --noisy-dir $(TEST_SETS)/noise_$$S --results $(PROJ)/datasets/results_test_set_model/$$S $(PROJ)/datasets/results_test_set_mtex/$$S \
	    --shape $(SHAPE) --jobs 8 --all-maps --csv $(SHARED_DIR)/figures/$$S/scores_mean.csv \
	    | sed -n '/^method/,/^====/p'; \
	done


maps: score plot


eval-val:
	cd $(MODEL_DIR)/src && dep=; for R in $(RUNS); do \
	  j=$$(RUN=$$R SETS=val_random sbatch --parsable $${dep:+--dependency=afterok:$$dep} eval_model.sh) || exit 1; \
	  echo "$$R -> job $$j"; dep=$$j; \
	done
