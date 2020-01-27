# Spoken Digit Recognition
This is an implementation of Spoken Digit Recognition by applying Convolutional Neural Networks using Spectrogram values of WAV audio files.
Here, to train, [Spoken_Digit_Dataset](https://www.kaggle.com/divyanshu99/spoken-digit-dataset/download) from kaggle is used. 

## Steps for loading the data
First Download the data from the given link : https://www.kaggle.com/divyanshu99/spoken-digit-dataset/download<br>
After downloading the data, to your workspace, extract the data and rename the folder as Spoken_Digit. Now, the data is ready to be processed.

## About the Dataset
Dataset consists of:
- 4 speakers
- 2,000 recordings (50 of each digit per speaker) at 8KHz frequency
- Digits from 0 to 9
- English pronunciations

## About the Model
In the model, spectrogram values for the WAV audio files are obatined which are then used to train the model after proper normalisation. Model consists of two layers of Convolution along with MaxPool, BatchNorm and two Dense Layers.

## Test Accuracy
Test data is assumed after making a 15% split from the total available dataset. Test accuracy of 92.33% was achieved after 20 iterations. Model with best performance was saved as Digit_Speech.bson

## References
<ul>
  <li>https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl</li>
</ul>
