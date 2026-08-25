BUCKET := gs://cmu-gpucloud-aiosman


.PHONY: auth ls upload download backup

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/

# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install 
upload:
	gcloud storage rsync -r ./training $(BUCKET)/training
	
# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install
download:
	gcloud storage rsync -r $(BUCKET)/training ./training
	