# Data and keras-retinanet setup 
(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed and configured)

Run the following shell commands from the top level directory

	cd data
	kaggle competitions download -c rsna-pneumonia-detection-challenge
	unzip stage_2_test_images.zip -d stage_2_test_images
	unzip stage_2_train_images.zip -d stage_2_train_images
	cd ../keras-retinanet
	python setup.py build_ext --inplace
	cd ..
	python prepare_data.py

Note that `settings.json` is configured for creating training and validation sets from the Stage 1 training labels, which are what I used to train the models for the competition.  I did not perform any training on the Stage 2 training set, which included Stage 1 test images.

# Model training

There are two RetinaNets used in my solution, which are trained by the following scripts.

	./train50.sh
	./train101.sh

Snapshots after each epoch of training are saved in `snapshots`.  For the competition, I manually selected the best snapshots from several runs based on the precision score and maximum Youden index reported during training. 


# Bounding box prediction

	python predict.py

Output .csv is saved to `submissions` folder