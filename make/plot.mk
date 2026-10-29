.PHONY: plot-train plot-test plot-maps

# All figures, drawn by MODEL_PIPELINE/src/plotting.py on CPU. SEED picks the model
# (checkpoints/<SEED> and datasets/results_test_set_model/<SEED>); figures go to
# MODEL_PIPELINE/figures/<SEED>/.
#
#   make plot-train                       training curves, figures/<SEED>/training_curves.png
#   make plot-test                        model vs MTEX on the 28 test sets (~1.5 min),
#                                         figures/<SEED>/curves_{all,boundary,corrupted,gross}.png
#   make plot-maps MAP=2 NOISE=scatter5_mis3       one map: clean / noisy / MTEX / model,
#                                         figures/<SEED>/maps/<NOISE>_map<MAP>_{ipf,error,grid}.png
#   make plot-maps MAP=2 NOISE=scatter5_mis3 CROP="40 40 48 48"   zoom in (rows, cols, height, width)
# MTEX has maps 1-10 only; for other maps the grid figure is skipped.

plot-train:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py curves ../checkpoints/$(SEED) \
	  --out ../figures/$(SEED)/training_curves.png

plot-test:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py test --seed $(SEED) $(if $(SETS),--sets $(SETS))

plot-maps:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py maps --seed $(SEED) --set $(NOISE) --map $(MAP) \
	  $(if $(CROP),--crop $(CROP))
