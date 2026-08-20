BUCKET := gs://cmu-gpucloud-aiosman
PROJECT_DIR := diffusion_project
BACKUP := diffusion_backup.tar.gz

.PHONY: auth ls upload download backup

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/

upload:
	gcloud storage rsync -r ./ $(BUCKET)/

download:
	gcloud storage rsync -r $(BUCKET)/ ./download


