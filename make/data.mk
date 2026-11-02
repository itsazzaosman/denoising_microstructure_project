
.PHONY: noise preview cache check-orientation

noise:
	@test -n "$(NOISE)" || { echo "set NOISE=<test set>, e.g. make noise NOISE=scatter5_mis3"; exit 1; }
	python $(SHARED_DIR)/src/add_ebsd_noise.py --config $(SHARED_DIR)/configs/noise/$(NOISE).yaml \
	  $(CLEAN_DIR) $(NOISY_DIR)
preview:
	@test -n "$(MAP)" && test -n "$(SCATTER)" && test -n "$(MISINDEX)" || { echo "set MAP=, SCATTER= and MISINDEX=, e.g. make preview MAP=7 SCATTER=5 MISINDEX=3"; exit 1; }
	python $(SHARED_DIR)/src/preview_noise.py --map $(MAP) --scatter $(SCATTER) --misindex $(MISINDEX)

cache:
	$(EBSD_PY) $(MODEL_DIR)/src/prepare_cache.py --jobs 16

check-orientation:
	$(EBSD_PY) $(MODEL_DIR)/src/orientation_torch.py --self-test

# First run this add_ebsd_noise.py script to generate the noisy dataset that we use for MTEX:
# python SHARED/src/add_ebsd_noise.py \
  datasets/clean_euler datasets/noise_scatter5 \
  --shape 128 128 --limit 500 \
  --scatter 5.0 --misindex 0 --save-mask --ang
