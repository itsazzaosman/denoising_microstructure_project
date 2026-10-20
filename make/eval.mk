# ---- evaluation: scores and figures ---------------------------------------
# Included by ../Makefile, which sets PROJ, MAP, SHAPE, NOISE and the folders.

.PHONY: score plot score-all score-grid maps

# Print the numbers: how far each denoised map (model and MTEX) sits from the
# truth. Use this when you want to RANK the filters -- it gives the mean
# error per filter, split out over the misindexed pixels (--mask) and the
# grain boundaries, so you can see whether a filter wins overall or only in
# the grain interiors. No pictures, just a table you can paste into notes.
score:
	python $(PROJ)/src/score_filter_denoising.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --mask   $(MASK) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE)

# Draw the pictures into figures/: maps_ipf.png (IPF-Z colour maps, one panel
# per filter, the familiar view) and maps_error.png (per-pixel error on a log
# colour scale, where leftover misindexed pixels glow). Use this when you want
# to SEE what a filter did to the grain boundaries and the speckle, rather
# than just read a mean. Note it takes no --mask.
plot:
	mkdir -p $(FIGURES)
	python $(PROJ)/src/plot_maps.py \
	  --clean  $(CLEAN) \
	  --noisy  $(NOISY) \
	  --results $(RESULTS) \
	  --shape  $(SHAPE) \
	  --out    $(FIGURES)/

# The same table, but aggregated over every map MTEX has finished, not just
# one. Use this for any claim you intend to defend: single-map scores wobble
# between maps, so the (sd) column tells you whether a gap between two methods
# is real. This is the number the model has to beat.
# Every score is a mean: per map, then across maps (decided 2026-10-13).
score-all:
	mkdir -p $(FIGURES)
	python $(PROJ)/src/score_dataset.py \
	  --clean-dir $(CLEAN_DIR) \
	  --noisy-dir $(NOISY_DIR) \
	  --results   $(RESULTS) \
	  --shape     $(SHAPE) \
	  --jobs      8 \
	  --csv       $(FIGURES)/scores_mean.csv

# Re-score every test set (the 28-set grid plus test_random) from the predictions
# already saved in datasets/results_test_set_{model,mtex}/<set>/, without running any model: one
# figures/<set>/scores_mean.csv per set. Run after evaluations or new MTEX
# outputs land. CPU only, about 20 minutes.
score-grid:
	for S in $$(cat $(PROJ)/configs/test_grid.txt) test_random; do \
	  mkdir -p $(PROJ)/figures/$$S; \
	  python $(PROJ)/src/score_dataset.py --clean-dir $(CLEAN_DIR) \
	    --noisy-dir $(TEST_SETS)/noise_$$S --results $(PROJ)/datasets/results_test_set_model/$$S $(PROJ)/datasets/results_test_set_mtex/$$S \
	    --shape $(SHAPE) --jobs 8 --all-maps --csv $(PROJ)/figures/$$S/scores_mean.csv \
	    | sed -n '/^method/,/^====/p'; \
	done

# Both of the single-map targets, the usual thing to run after a new MTEX sweep.
maps: score plot

# cd /project/community/aiosman/MAPS_DENOISING_Orientation_Noise

# # all 10 maps at 5° scatter
# python src/plot_mtex_maps.py --scatter 5 --all

# # all 10 maps at all three noise levels
# for s in 5 7 9; do python src/plot_mtex_maps.py --scatter $s --all; done

# # or just one map at a time
# python src/plot_mtex_maps.py --scatter 5 --maps 3

# # optional: zoom into a 48×48 region (rows 40–88, columns 40–88)
# python src/plot_mtex_maps.py --scatter 5 --maps 3 --crop 40 40 48 48


#python src/plot_mtex_maps.py --ang-dir datasets/noise_scatter9/ang_files \
  --results datasets/results_test_set_mtex_old/scatter5_nograins --out figures/mtex_maps/scatter5_nograins --all
