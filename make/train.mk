.PHONY: train-cpu train-gpu watch


train-gpu:
	cd $(MODEL_DIR)/src/job_scripts && sbatch --job-name=$(or $(CONFIG),ebsd_orient_gpu) --partition=general \
	  --qos=general_qos --time=12:00:00 train_model.sh --resume auto

watch:
	squeue -u $(USER) -o "%.10i %.10P %.20j %.3t %.10M %.6C %.22R"
