# Dataset-BCI-competition-iv-2b
EEG Classification using CNN
Introduction
This is a python code for extracting EEG signals from dataset 2b from competition iv, then it converts the data to spectrogram images to classify them using a CNN classifier.

The code is designed to load and preprocess data, then pass it through a CNN classifier that was trained on the same dataset. The output of the classifier is then used to classify the data into two classes.

# Project Dependencies
Python 2.7 or 3.7
The following python packages must be installed:
numpy
matplotlib
mne
pandas
scipy
gumpy



# Program Details
1. Load Raw Data
First, the program loads the raw data using the GrazB dataset location and subject ID.

2. Data Preprocessing
Then, the loaded data is preprocessed using the following parameters:

FS = 250
LOWCUT = 8
HIGHCUT = 30
ANTI_DRIFT = 0.5
CUTOFF = 50.0
Q = 30.0
W0 = CUTOFF / (FS / 2)
3. Data Augmentation
The data is then augmented by GAN model.

4. Short-Time Fourier Transform (STFT)
The STFT is used to transform the time-domain signal to the frequency domain signal.

5. Concatenation of Images
The function get_concat_v() is used to concatenate the images of MI_cl1, MI_cl2, and MI_cl1_cl2

6. CNN Model
The CNN model is used to classify the EEG images.

# Acknowledgements
This code is based on the gumpy repository and the dataset is from the BCI Competition IV. We acknowledge and appreciate their efforts to share their work with the research community.
