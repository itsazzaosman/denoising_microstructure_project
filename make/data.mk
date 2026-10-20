# ---- datasets: noisy test sets, training cache, orientation self-test -----
# Included by ../Makefile, which sets PROJ, NOISE, MAP and the folders.

.PHONY: noise preview cache check-orientation

# Build a test set from its experiment file, configs/noise/$(NOISE).yaml ->
# $(NOISY_DIR). Maps that already exist are skipped, so this also
# resumes an interrupted run. Each run is logged in provenance.json there.
noise:
	python $(PROJ)/src/add_ebsd_noise.py --config $(PROJ)/configs/noise/$(NOISE).yaml \
	  $(CLEAN_DIR) $(NOISY_DIR)

# Look at one map at chosen noise levels before making them a test set:
# `make preview MAP=00001 SCATTER="1 5 9" MISINDEX="0 9 15"` draws one panel
# per (scatter, misindex) pair into figures/noise_preview/. Same seed as the
# test sets, so a level already in the grid looks exactly like its files.
preview:
	python $(PROJ)/src/preview_noise.py --map $(MAP) --scatter $(SCATTER) --misindex $(MISINDEX)

# Pack the clean maps into one memmappable .npy. Run once; takes about a
# minute. Training reads this instead of re-parsing 12k text files per epoch.
cache:
	$(EBSD_PY) $(PROJ)/src/prepare_cache.py --n-train 12000 --jobs 16

# Check the orientation maths the model will be built on against the numpy
# reference. Run it after touching anything in orientation_torch.py.
check-orientation:
	$(EBSD_PY) $(PROJ)/src/orientation_torch.py --self-test

# First run this add_ebsd_noise.py script to generate the noisy dataset that we use for MTEX:
# python src/add_ebsd_noise.py \
  datasets/clean_euler datasets/noise_scatter5 \
  --shape 128 128 --limit 500 \
  --scatter 5.0 --misindex 0 --save-mask --ang
