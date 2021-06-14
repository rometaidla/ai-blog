---
toc: true
layout: post
description: Lane Following using Comma-AI dataset.
categories: [markdown]
title: Lane Following using Comma-AI dataset
---

# Part 1: Lane Following using Comma AI dataset

This is the first post in the series of posts on end-to-end model for autonomous driving. Future posts:
- Part 2: Using Vision Transformers with Comma AI dataset
- Part 3: Using PilotNet model in real life on Estonian gravel roads

## Introduction

- Imitation learning
- NHTSA Level

![EndToEnd]({{ site.baseurl }}/images/lanefollowing/end-to-end-learning.jpg "Credit: https://twitter.com/haltakov/status/1384192583597912065")

## Data

[Comma AI dataset](https://github.com/commaai/comma2k19) is used to train the model. This dataset has over 33 hours of
commute in California's highway. Dataset is divided into 95% training, 5% validation and 5% test set.

Video resolution is 1164x874. When extracting frames for training, image is downscaled to the resolution of 258x194 for
faster training process. From this downscaled image, smaller region of interest is cropped as most of the image does not
in include information useful for training, like trees and sky.

![RegionOfInterest]({{ site.baseurl }}/images/lanefollowing/roi.png "Region of interest used for training is marked with red box.")

Comma AI dataset contains a small sample of very difficult situations like crossroads, which are impossible to predict
correctly using just camera images as model has no clear information whether to turn left or right. Most of these cases
have high steering angle and make it very hard for model to converge (especially with using MSE loss). To avoid manually
going through hours of videos, all frames with steering angle bigger than 20 degrees are removed from dataset.
(TODO: include exact counts of frames removed)

### Model
Convolutional neural network have been most succesful architectures in computer vision and it is natural choice for lane
following. NVIDIA used CNN architecture in their DAVE-2 system called PilotNet [1].

![PilotNet]({{ site.baseurl }}/images/lanefollowing/pilotnet-architecture.png "PilotNet architecture defined in Nvidia paper")

I used Batch normalisation instead of first static normalisation layer as I found it made training more stable,
model trained quicker and had less variability in epochs validation losses. Also Leaky ReLU is used as activation
function for layers.

|Layer name|Output size|Number of parameters|
|---|---|---|
|Convolution2D 1|   |   |
|BatchNorm2D 1|   |   |
|LeakyRelu 1|   |   |
|Convolution2D 2|   |   |
|BatchNorm2D 2|   |   |
|LeakyRelu 2|   |   |

### Training

Model is trained until validation loss fails to improve for 10 epochs. Mean absolute error (MAE) is used  as loss function
as it proved to work better compared to mean square error (MSE) loss as it does not magnify errors with big steering angles.

Big effort was needed to speed up training speed as training on video files is very slow, this is magnified with the need
to access frames randomly during training. I tried to speed up the processing by using [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs),
but reading video frames was still bottleneck and not GPU. The best performance was achieved by extracting all frame from video
files into JPG files with reduced resolution using *ffmpeg* utility and training model using these.

TODO: hyperparameter tuning

#### Data balancing

Driving data is very unbalanced, most driving is done straight or with very small steering angle. This can be seen also
in Comma AI dataset:

![Unbalanced]({{ site.baseurl }}/images/lanefollowing/unbalanced-data.png "Comma AI dataset is very unbalances, most steering angles are near 0 degrees.")

This presents problem for training neural network as it will be biased to predict small degrees and under-predict bigger
steering angles. I tested with two balanced datasets

![Balanced]({{ site.baseurl }}/images/lanefollowing/balanced-data.png "Balanced dataset by oversampling images with less frequent steering angles so that we get uniform distribution.") | ![Fattails]({{ site.baseurl }}/images/lanefollowing/fattails-data.png "Balanced dataset by oversampling image with less frequent steering angles so that tails of distribution are fatter.")

## Results

Training loss improved mostly during first 5 iteration for every data balancing variation as this dataset is quite big 
and has similar driving data.

![Train loss]({{ site.baseurl }}/images/lanefollowing/trainloss.png "Training loss")


Similarly validation loss drops quickly and only has minor improvement afterwards. Unbalanced dataset achieves the lowest 
validation loss already on 7th epoch and fat-tailed balanced dataset on 11th epoch. Unbalanced and fat-tails data distributions
seem to train better and achieve lower validation loss compared to uniformly balanced dataset.

![Validation loss]({{ site.baseurl }}/images/lanefollowing/valloss.png "Validation loss")


Model trained with fat-tailed distribution got the lowest test loss of **1.107**, followed by model with unbalance
data **1.157**. Model trained with uniform distribution got the highest test loss of **1.223**, which is probably caused
by changing initial distribution too much and causing model to overestimate small steering angles. 

Oversampling data to fat-tailed distribution seems to be promising and training
model with even fatter tails could improve results even further. Losses were measured using only one training run and
to get more valid comparison, several runs should be made to see the variability in results.

![Test loss]({{ site.baseurl }}/images/lanefollowing/testloss.png "Test loss")

Model performance during the day (green is true steering angle and red is predicted steering angle):
<figure class="video_container">
  <iframe width="840" height="550" src="https://www.youtube.com/embed/iR3qkDQD_Pk?controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>

During the evening:
<figure class="video_container">
  <iframe width="840" height="550" src="https://www.youtube.com/embed/ZirSZe89fcU?controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>

During the night:
<figure class="video_container">
  <iframe width="840" height="550" src="https://www.youtube.com/embed/p1wjzHW8HCY?controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>

## Visualisation of network state
[Grad-CAM](https://arxiv.org/abs/1610.02391) paper introduces technique for producing "visual explanations" for decisions
from a large class of CNN-based models. There is great [pytorch implementation](https://github.com/jacobgil/pytorch-grad-cam)
implemention for class prediction, which I modified to work with regression problem to display activation maps.

![Gradcam layers]({{ site.baseurl }}/images/lanefollowing/gradcam_layers.png "Gradcam++ activation maps")

First layer seems to provides best information. Model seems to be mostly concentrating on road markings, ground under
other cars and sides of the road:
> <figure class="video_container">
  <iframe width="840" height="550" src="https://www.youtube.com/embed/hlQyDc7xGMc?controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>


## Conclusions

### Further improvements
- Use more complex model
- Use PilotNet with bigger input size
- Do data augmentation

### Learnings
- Most effort will go preparing data pipeline and not tuning model itself
- Gradient based visualisation can provide insights into how model works

## References

[1] Pilotnet https://arxiv.org/abs/1604.07316