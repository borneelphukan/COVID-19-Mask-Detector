# COVID-19-Mask-Detector

## Introduction
Ever since the dawn of the COVID-19 pandemic, public places all across the globe has introduced compulsory norms to wear facial masks in order to protect oneself and others from being a contagion of the deadly virus. However, people across the world not accustomed to the new normal are very uncomfortable adjusting to the compulsory mask norm and therefore are more inclined to not wearing masks and thereby endangering oneself and others. Also the difficulties of authorities are increasing day after day as large number of people are getting back to work but also violating the compulsory mask norm. Therefore this project aims to curb the issue of monitoring people not wearing mask especially in regions such as security boots, restaurant doors etc.

## Dataset
The dataset used for training the deep convolutional neural network model is an open-source custom generated dataset contributed to github. It contains two groups of image data:  
1. with_mask (686 images)
2. without_mask  (686 images)

However as a developer it is a kind request to please add more data into the set so that further variation to the dataset bring better training and functionalities to the project.

## Neural Network Architecture

![Architecture](https://github.com/borneelphukan/COVID-19-Mask-Detector/blob/master/images/network.png)


## Training & Accuracy

The training took approximately 10 minutes to complete. It required 20 epochs with each epoch having a batch size of 34 images and the learning rate of the Adam optimizer set to 1e-4. 
![Training Phase](https://github.com/borneelphukan/COVID-19-Mask-Detector/blob/master/images/Training.png)

The confusion matrix is given below:  
![Confusion Matrix](https://github.com/borneelphukan/COVID-19-Mask-Detector/blob/master/images/Classification.png)  

The training loss and the accuracy of the model were monitored throughout the training phase and is given below:  
![Training Loss & Accuracy Graph](https://github.com/borneelphukan/COVID-19-Mask-Detector/blob/master/plot.png)

## Screenshots

![With Mask](https://github.com/borneelphukan/COVID-19-Mask-Detector/blob/master/images/With%20Mask.png)  
With Mask
![Without Mask](https://github.com/borneelphukan/COVID-19-Mask-Detector/blob/master/images/Without%20Mask.png)

## References
1. Dataset Generation - https://github.com/prajnasb/observations
2. Mobilenet Reference - https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69
