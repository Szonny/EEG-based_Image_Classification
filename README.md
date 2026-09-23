# Image-classificaton-from-EEG
Image classification and reconstruction based on EEG data.
Project focusing on trying out different methods of modifying data hidden within EEG signal, using python with MNE and numpy. 
After data aqusituion in form of three hour examination of nine patients, exposed to different images during experiment, 
raw signal has been prepared and used in training three models in order to train neural networks in pattern recognition.

Result networks where expected to recognize object on the image from one of eleven possible object types.
Tested methods include CNN (Convolutional Neural Network) and models based on EEGnet, EEGInception.

Procject also contains an attempt to reconstruct images seen by patients, 
but because of insufficient data, attempt has been abandoned on early stage.

# Project Background
Semester project for biologically inspired AI subcject during bachelor studies on Silesian University of Technology.

# File Description
 |	File		 | Description 	 |
 | ------------- | ------------- |
 | raport/ 							| mandatory university end-project report, documenting achieved results. |
 | data_preparation.py  			| script that performs basic data processing.							 |
 | data_randomizer.py 	 			| script that shuffles and transposes processed data.					 |
 | dispatch_preparation.py 			| aggregates data preparation and shuffling.							 |
 | EEG_reconstruction.py   			| performs image recounstruction.										 |
 | Model_Training_EEGInception.py 	| trains an EEGInception based model.									 |
 | EEG_retrieval.py        			| data retrieval using image embedding.									 |
 | Model_Training_EEGNet.py			| trains an EEGnet based model.											 |
 | notebook.ipynb					| testfile for experiments with data preparation.						 |
 | ModelTrainingCNN.py     			| trains CNN model, made using Tensorflow.								 |
