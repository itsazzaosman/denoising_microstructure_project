.PHONY: train-cpu train-gpu infer watch


train-gpu:
	cd $(MODEL_DIR)/src/job_scripts && sbatch --job-name=$(or $(CONFIG),ebsd_orient_gpu) --partition=general \
	  --qos=general_qos --time=12:00:00 train_model.sh --resume auto

	 

# Inference of one trained model on the 28 fixed-level test sets (GPU job,
# eval_model.sh); results in datasets/results_test_set_model/<checkpoint folder>/<set>/
#   make infer CONFIG=blind_both_seed1                  start now
#   make infer CONFIG=blind_both_seed1 AFTER=162403     wait until training job 162403 ends OK
#   make infer CONFIG=blind_both_seed1 SETS="scatter5 scatter5_mis3"   only these sets
infer:
	@test -n "$(CONFIG)" || { echo "set CONFIG=<training config>, e.g. CONFIG=blind_both_seed1"; exit 1; }
	cd $(MODEL_DIR)/src/job_scripts && CONFIG=$(CONFIG) SETS="$(SETS)" sbatch \
	  --job-name=infer_$(CONFIG) $(if $(AFTER),--dependency=afterok:$(AFTER)) eval_model.sh

watch:
	squeue -u $(USER) -o "%.10i %.10P %.20j %.3t %.10M %.6C %.22R"
