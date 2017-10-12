# Image-Classification-CNN
A ResNet50 CNN that classifies 3D objects given 2D images

#Background
Project was done for the class CS 6501: 3D Reconstruction and Understanding at the University of Virginia.
Work and design of CNN was inspired by this paper: https://arxiv.org/abs/1505.00880

# Single View
This class fine tunes a ResNet50 convolutional neural network that has been pretrained on Imagenet. The input is a 2D image that is flipped with 50% probability to increase accuracy.

# Multiview
This class uses multiple 2D views of the 3D object as input. There is an intermediate pooling step before the classification is done.

# Hardware
The Deep Learning AMI EC2 instance from AWS was used to train/test the data: https://aws.amazon.com/marketplace/pp/B06VSPXKDX

# Results
The single view CNN had an accuracy of ~90%

The multiview CNN had an accuracy of ~87.5%
