.PHONY: train-cpu train-gpu infer watch


train-gpu:
	cd $(MODEL_DIR)/src/job_scripts && sbatch --job-name=$(or $(CONFIG),ebsd_orient_gpu) --partition=general \
	  --qos=general_qos --time=12:00:00 train_model.sh --resume auto

	 

infer:
	$(EBSD_PY) $(MODEL_DIR)/src/inference.py \
	  --checkpoint $(CKPT) \
	  --clean-dir  $(CLEAN_DIR) \
	  --noisy-dir  $(NOISY_DIR) \
	  --out        $(MODEL_OUT) \
	  --method-name $(or $(RUN),unet)

watch:
	squeue -u $(USER) -o "%.10i %.10P %.20j %.3t %.10M %.6C %.22R"
