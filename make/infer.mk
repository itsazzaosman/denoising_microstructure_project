.PHONY: infer

# Inference of one trained model on the 28 fixed-level test sets (GPU job,
# eval_model.sh); results in datasets/results_test_set_model/<checkpoint folder>/<set>/
#   make infer CONFIG=blind_both_seed1                  start now
#   make infer CONFIG=blind_both_seed1 AFTER=162403     wait until training job 162403 ends OK
#   make infer CONFIG=blind_both_seed1 SETS="scatter5 scatter5_mis3"   only these sets
#   make infer CONFIG=blind_both_seed1 NAME=unet        name in file names and score tables
#                                                      (default: the config name)
infer:
	@test -n "$(CONFIG)" || { echo "set CONFIG=<training config>, e.g. CONFIG=blind_both_seed1"; exit 1; }
	cd $(MODEL_DIR)/src/job_scripts && CONFIG=$(CONFIG) SETS="$(SETS)" NAME="$(NAME)" sbatch \
	  --job-name=infer_$(CONFIG) $(if $(AFTER),--dependency=afterok:$(AFTER)) eval_model.sh
