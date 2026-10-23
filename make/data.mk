
.PHONY: noise preview cache check-orientation

noise:
	python $(SHARED_DIR)/src/add_ebsd_noise.py --config $(SHARED_DIR)/configs/noise/$(NOISE).yaml \
	  $(CLEAN_DIR) $(NOISY_DIR)
preview:
	python $(SHARED_DIR)/src/preview_noise.py --map $(MAP) --scatter $(SCATTER) --misindex $(MISINDEX)

cache:
	$(EBSD_PY) $(MODEL_DIR)/src/prepare_cache.py --n-train 12000 --jobs 16

check-orientation:
	$(EBSD_PY) $(MODEL_DIR)/src/orientation_torch.py --self-test

# First run this add_ebsd_noise.py script to generate the noisy dataset that we use for MTEX:
# python SHARED/src/add_ebsd_noise.py \
  datasets/clean_euler datasets/noise_scatter5 \
  --shape 128 128 --limit 500 \
  --scatter 5.0 --misindex 0 --save-mask --ang
