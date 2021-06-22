---
toc: true
layout: post
description: Training end-to-end driving model using Nvidia PilotNet architecture and CommaAI dataset.
categories: [markdown]
title: End-to-end Driving using Comma AI dataset
---

# Part 1: End-to-end Driving using Comma AI dataset

This is the first post in the series of posts on the topic of end-to-end model for autonomous driving. Future posts:
- Part 2: Using PilotNet model in real car on Estonian gravel roads
- Part 3: Using Vision Transformers with Comma AI dataset

## Introduction

In this project I will use imitation learning to teach artificial neural network to drive the car end-to-end. Model will
predict only steering command, but controlling throttle can be achieved using similar techniques [7].

**End-to-End driving** is predicting output command directly from input sensor, in this case predicting steering command
from camera image.

![EndToEnd]({{ site.baseurl }}/images/lanefollowing/end-to-end-learning.jpg "Credit: https://twitter.com/haltakov/status/1384192583597912065")

**Imitation learning** is supervised learning, where model is trained to mimic the human behaviour. In this project model is trained
to predict how human is steering the car on highways.

## Data

[Comma AI dataset](https://github.com/commaai/comma2k19) [6] is used to train the model. This dataset has over **33 hours** of
commute on California's highway. I divided dataset into 95% training, 5% validation and 5% test set.

Video resolution is 1164x874. When extracting frames for training, image is downscaled to the resolution of **258x194** for
faster training process. From this downscaled image, smaller region of interest is cropped as most of the image does not
in include information useful for training, like trees and sky.

![RegionOfInterest]({{ site.baseurl }}/images/lanefollowing/roi.png "Region of interest used for training is marked with red box.")

Comma AI dataset contains a small sample of very difficult situations like crossroads, which are impossible to predict
correctly using just camera images as model has no clear information whether to turn left or right. Most of these cases
have high steering angle and make it very hard for model to converge (especially with using MSE loss). To avoid manually
going through hours of videos, all frames with steering angle bigger than 20 degrees are removed from dataset.

### Model
Convolutional neural network have been most succesful architectures in computer vision and it is natural choice for lane
following. NVIDIA used CNN architecture in their DAVE-2 system called PilotNet [1].

![PilotNet]({{ site.baseurl }}/images/lanefollowing/pilotnet-architecture.png "PilotNet architecture defined in Nvidia paper")

I used Batch normalisation instead of first static normalisation layer as I found it made training more stable,
model trained quicker and had less variability in epochs validation losses. Also Leaky ReLU is used as activation
function for layers.
### Training

Big effort was needed to speed up training speed as training on video files is very slow, this is magnified with the need
to access frames randomly during training. I tried to speed up the processing by using [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs) [5],
but reading video frames was still bottleneck and not GPU. The best performance was achieved by extracting all frame from video
files into JPG files with reduced resolution using *ffmpeg* utility and training model using these. When training using
extracted images, one epoch (around 30 hours of driving) takes around 30 minutes, instead of several hours when using video
based solutions.

Model is trained until validation loss fails to improve for 10 epochs. Mean absolute error (MAE) is used as the loss function
as it proved to work better compared to mean square error (MSE) loss as it does not magnify errors with big steering angles.

### Data balancing

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
  <iframe width="840" height="550" src="https://www.youtube.com/embed/0ZZtgRv3__c?controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>

![Predicted steering angles during the day]({{ site.baseurl }}/images/lanefollowing/test-steering-angles-day.png "Predicted steering angles during the day")

During the night:
<figure class="video_container">
  <iframe width="840" height="550" src="https://www.youtube.com/embed/VQ7BT5ZxkIc?start=0&controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>

![Predicted steering angles during the night]({{ site.baseurl }}/images/lanefollowing/test-steering-angles-night.png "Predicted steering angles during during the night")

## Visualisation of network state
[Grad-CAM](https://arxiv.org/abs/1610.02391) paper [2] introduces technique for producing "visual explanations" for decisions
from a large class of CNN-based models. It is technique for visualising importance of image feature to the final output
of the network. There is great [pytorch implementation](https://github.com/jacobgil/pytorch-grad-cam) of these gradient based methods. 

As this technique is for classification problem and predicting steering angle is regression problem, I modified the implementation 
to work with regression problem by using ideas from Jacob Gildenblat blog [4]. Target steering angles are divided into 3 ranges,
turning strong to the left (big positive steering angles), turning strongly the right (big negative steering angles) and driving straight
(small steering angles).This makes it classification problem again. When steering angle are big, image features contributing 
most to big steering angles are peaked. When steering angle is small, image features contributing mostly to small steering 
angle by taking inverse of steering angle as our target.

```python
def grad_cam_loss(self, x, angle):
    if angle > 2.0:
        return x
    elif angle < -2.0:
        return -x
    else:
        return torch.reciprocal(x.cpu()) * np.sign(angle.cpu())
```

By resulting activation maps for each convolutional layer are following:

![Gradcam layers]({{ site.baseurl }}/images/lanefollowing/gradcam_layers.png "Gradcam activation maps")

First layer seems to provide the best information. Model seems to be mostly concentrating on road markings, ground under
other cars and sides of the road:
<figure class="video_container">
  <iframe width="840" height="550" src="https://www.youtube.com/embed/hlQyDc7xGMc?controls=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</figure>

## Conclusions

Most effort will go preparing data pipeline and making it fast enough, not tuning model itself. Simple convolutional neural
network like PilotNet is quite good at learning simple lane following using camera images. Gradient based visualisations 
can provide insights into how model works and what parts of input image are important for the model.

There are several improvements that can be done to the model:
- Use more complex model (Resnet, Vision Transformers)
- Use PilotNet with bigger input image size
- Do data augmentation for concurring with the distribution shift problem

## References

[1] End to End Learning for Self-Driving Cars [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)

[2] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

[3] Class Activation Map methods implemented in Pytorch [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

[4] Visualizations for regressing wheel steering angles in self driving cars, Jacob Gildenblat [https://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations](https://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations)

[5] NVIDIA Data Loading Library [https://docs.nvidia.com/deeplearning/dali/user-guide/docs](https://docs.nvidia.com/deeplearning/dali/user-guide/docs)

[6] CommaAI 2k19 dataset [https://github.com/commaai/comma2k19](https://github.com/commaai/comma2k19)

[7] Towards End-to-End Deep Learning for Autonomous Racing: On Data Collection and a Unified Architecture for Steering and Throttle Prediction  [https://arxiv.org/abs/2105.01799] (https://arxiv.org/abs/2105.01799)