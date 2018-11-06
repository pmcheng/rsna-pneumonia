## 3rd place solution for the 2018 RSNA Pneumonia Detection Challenge

**by Phillip Cheng, MD MS**

Hello!  Below is an outline of how I prepared the data, trained my models, and predicted bounding boxes for the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).  Further details are in the model documentation submitted to Kaggle.  For questions please contact me at <phillip.cheng@med.usc.edu>

# Directory structure

- `data`: contains raw and processed data from Kaggle
- `keras-retinanet`: modified version of [keras-retinanet](https://github.com/fizyr/keras-retinanet)
- `models`: selected resnet-50 and resnet-101 model snapshots used for the prediction submissions
	- Note: the .h5 model files are too large to be stored in the  Github repository, so you will need to manually download the .h5 files from the release and place them in this folder.
- `snapshots`: folder for saved snapshots after each epoch of training
- `submissions`: folder to hold output from `predict.py` 

# My hardware 
- Dell Precision T5500 with 3.07 GHz Intel Xeon X5675 CPU 
- 12 GB NVIDIA Titan Xp GPU
- 512 GB HDD
- 16 GB DDR3 RAM


# Software 
(python packages are detailed separately in `requirements.txt`):

- Ubuntu 18.04 LTS
- Python 3.6.6
- CUDA 9.0
- cuDNN 7.0.5
- NVIDIA driver v.390


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