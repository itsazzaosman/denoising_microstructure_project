BUCKET := gs://cmu-gpucloud-aiosman


.PHONY: auth ls upload download backup

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/

upload:
	gcloud storage rsync -r ./training $(BUCKET)/training

download:
	gcloud storage rsync -r $(BUCKET)/training ./training


