.PHONY: auth ls ls-datasets ls-checkpoints download

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/
ls-datasets:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/datasets/
ls-checkpoints:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/checkpoints/runs/

download:
	gcloud storage rsync -r $(if $(DRY),--dry-run) $(BUCKET)/DATASET_CHECKPOINTS/datasets $(PROJ)/datasets
	gcloud storage rsync -r $(if $(DRY),--dry-run) $(BUCKET)/DATASET_CHECKPOINTS/checkpoints $(MODEL_DIR)/checkpoints

