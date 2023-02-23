# Dataset-BCI-competition-iv-2b
EEG Classification using CNN
Introduction
This is a python code for extracting EEG signals from dataset 2b from competition iv, then it converts the data to spectrogram images to classify them using a CNN classifier.

The code is designed to load and preprocess data, then pass it through a CNN classifier that was trained on the same dataset. The output of the classifier is then used to classify the data into two classes.

Requirements
Python 3.5 or later
gumpy
numpy
pandas
matplotlib
mne
Pillow
scipy



# Code Structure
The code is organized as follows:

The necessary libraries are imported.
Parameters for filtering data and CNN classification are set.
The raw data is loaded using gumpy library.
Data is preprocessed by applying a notch and a bandpass filter.
Train and test data is split using load_preprocess_data() function in the utilss.py module.
Data is converted to spectrogram images using stft_data() function.
The spectrogram images are then saved in a designated folder.
The output is then passed through a CNN classifier to predict the class labels.
The classification accuracy is printed on the console.

# Acknowledgements
This code is based on the gumpy repository and the dataset is from the BCI Competition IV. We acknowledge and appreciate their efforts to share their work with the research community.
