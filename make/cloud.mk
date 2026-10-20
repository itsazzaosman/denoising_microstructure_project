# ---- cloud storage: Google Cloud bucket and OneDrive ----------------------
# Included by ../Makefile, which sets BUCKET.

.PHONY: auth ls ls-training ls-datasets ls-checkpoints upload download

auth:
	gcloud auth login --no-launch-browser

ls:
	gcloud storage ls $(BUCKET)/
ls-training:
	gcloud storage ls $(BUCKET)/training/

ls-datasets:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/datasets/
ls-checkpoints:
	gcloud storage ls $(BUCKET)/DATASET_CHECKPOINTS/checkpoints/

# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install
upload:
	gcloud storage rsync -r ./DATASET_CHECKPOINTS/datasets $(BUCKET)/DATASET_CHECKPOINTS/datasets

# gcloud storage rsync -r ./emsoft_install $(BUCKET)/emsoft_install
download:
	gcloud storage rsync -r $(BUCKET)/DATASET_CHECKPOINTS/datasets ./DATASET_CHECKPOINTS/datasets

# To see what is in the onedrive folder, run this command. It is a bit slow
# rclone lsf "onedrive:Desktop/Research Project - Material Science + Diffusion Models/Pipline For Pictures and Euler Angles Only/"
# if you want to download the whole thing, run this command. It is very slow.
# rclone copyto "onedrive:Desktop/Research Project - Material Science + Diffusion Models/Pipline For Pictures and Euler Angles Only/misorientation_noise_dataset_5deg/mag5/clean_euler.tar" "/project/community/aiosman/MAPS_DENOISING_Orientation_Noise/datasets/clean_euler.tar" --progress
