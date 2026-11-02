.PHONY: auth ls ls-datasets ls-checkpoints download

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/
ls-datasets:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/datasets/
ls-checkpoints:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/checkpoints/

# Copy datasets/ and MODEL_PIPELINE/checkpoints/ from the bucket (the nightly cron job,
# SHARED/src/sync_to_gcs.py, uploads them). Nothing is deleted, but a local file that
# differs from the bucket's copy is overwritten. DRY=1 only lists the copies.
download:
	gcloud storage rsync -r $(if $(DRY),--dry-run) $(BUCKET)/DATASET_CHECKPOINTS/datasets $(PROJ)/datasets
	gcloud storage rsync -r $(if $(DRY),--dry-run) $(BUCKET)/DATASET_CHECKPOINTS/checkpoints $(MODEL_DIR)/checkpoints

