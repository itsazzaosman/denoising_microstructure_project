.PHONY: score plot score-all score-grid maps eval-val eval-test

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


# Test one trained model on every test set (GPU job, eval_model.sh), e.g.
#   make eval-test CONFIG=blind_both_seed1                 now
#   make eval-test CONFIG=blind_both_seed1 AFTER=162403    once training job 162403 ends OK
#   make eval-test CONFIG=blind_both_seed1 SETS="scatter5 scatter5_mis3"   only these sets
eval-test:
	@test -n "$(CONFIG)" || { echo "set CONFIG=<training config>, e.g. CONFIG=blind_both_seed1"; exit 1; }
	cd $(MODEL_DIR)/src/job_scripts && CONFIG=$(CONFIG) SETS="$(SETS)" sbatch \
	  --job-name=eval_$(CONFIG) $(if $(AFTER),--dependency=afterok:$(AFTER)) eval_model.sh


eval-val:
	cd $(MODEL_DIR)/src/job_scripts && dep=; for R in $(RUNS); do \
	  j=$$(CONFIG=$$R SETS=val_random sbatch --parsable $${dep:+--dependency=afterok:$$dep} eval_model.sh) || exit 1; \
	  echo "$$R -> job $$j"; dep=$$j; \
	done
