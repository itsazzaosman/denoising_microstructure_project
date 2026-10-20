# ---- training and inference -----------------------------------------------
# Included by ../Makefile, which sets PROJ, NOISE, RUN and the folders.

.PHONY: train-cpu train-gpu infer watch

# Submit training: `make train-gpu CONFIG=scatter7` trains the experiment in
# configs/train/scatter7.yaml (make passes CONFIG on to the job). The CPU variant exists because every GPU on this cluster was
# allocated with ~470 jobs queued while 330 CPU cores sat idle; at base 32 it
# is slow but not unreasonably so. They write to different checkpoint dirs
# (checkpoints/ and checkpoints_gpu/) so both can run at once.
train-cpu:
	@echo "Training runs on a GPU only (decided 2026-10-12). Use: make train-gpu CONFIG=<name>"; exit 1

train-gpu:
	cd $(PROJ)/src && sbatch --job-name=$(or $(CONFIG),ebsd_orient_gpu) --partition=preempt \
	  --qos=preempt_qos --time=2-00:00:00 train_model.sh --resume auto

# Predict the held-out maps and drop the results into $(MODEL_OUT) as
# <stem>__unet.txt, so score-all compares them against the filters using the
# same code path. Both job scripts already do this when training ends.
infer:
	$(EBSD_PY) $(PROJ)/src/inference.py \
	  --checkpoint $(CKPT) \
	  --clean-dir  $(CLEAN_DIR) \
	  --noisy-dir  $(NOISY_DIR) \
	  --out        $(MODEL_OUT) \
	  --method-name $(or $(RUN),unet)

watch:
	squeue -u $(USER) -o "%.10i %.10P %.20j %.3t %.10M %.6C %.22R"
