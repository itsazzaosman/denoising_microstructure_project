.PHONY: plot-train plot-zoom plot-test plot-maps

# All figures, drawn by MODEL_PIPELINE/src/plotting.py on CPU. SEED picks the model
# (checkpoints/<SEED> and datasets/results_test_set_model/<SEED>); figures go to
# MODEL_PIPELINE/figures/<SEED>/.
#
#   make plot-train                       training curves, figures/<SEED>/training_curves.png
#   make plot-zoom                        zoomed loss + gate panels, figures/<SEED>/training_zoom.png
#   make plot-zoom EPOCHS=19              epochs 1-19 only, figures/<SEED>/training_zoom_epoch19.png
#   make plot-test                        model vs MTEX on the 28 test sets (~1.5 min),
#                                         figures/<SEED>/curves_{all,boundary,corrupted,gross}.png
#   make plot-maps MAP=2 NOISE=scatter5_mis3       one map: clean / noisy / MTEX / model as a
#                                         3 x 3 grid, figures/<SEED>/maps/<NOISE>_map<MAP>_grid.png
# MTEX has maps 1-10 only, so plot-maps needs MAP between 1 and 10.

plot-train:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py curves ../checkpoints/$(SEED) \
	  --out ../figures/$(SEED)/training_curves.png

plot-zoom:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py zoom ../checkpoints/$(SEED) \
	  $(if $(EPOCHS),--epochs $(EPOCHS))

plot-test:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py test --seed $(SEED) $(if $(SETS),--sets $(SETS))

plot-maps:
	cd $(MODEL_DIR)/src && $(EBSD_PY) plotting.py maps --seed $(SEED) --set $(NOISE) --map $(MAP)
