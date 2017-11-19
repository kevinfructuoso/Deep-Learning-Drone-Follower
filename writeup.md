# Project: Follow Me Project

[image0]: ./docs/misc/follow_me_performance_example.gif
[image1]: ./docs/misc/follow_me_pyqt_overlay.gif
[image3]: ./docs/misc/network_architecture.png

---
### Writeup

#### Network Architecture

The following image details the network architecture used to train this model. It includes:

* 2 Encoder layers to extract features
* 1 1x1 Convolutional layer to maintain spatial information
* 2 Decoder layers to upsample to the original image size
* 2 Skip connections between the Encoded and Decoded layers to retain information from the original image and make more precise segmentation decisions

![network architecture][image3]

#### Tuning Parameters

There are a handful of parameters that require fine-tuning in order to generate a well-trained model. The list and their descriptions are given below.

* learning_rate: The rate at wthich the network can learn (update the model weights) during training in order to minimize loss. 
> While larger learning rates should usually end up with better models and faster learning in theory, it can be observed that networks with larger learning rates can plateau earlier than those with smaller learning rates. This results in more loss during training. With this in mind, a small learning rate of 0.001 was chosen.

* batch_size: The number of training samples/images that get propagated through the network in a single pass.
> This parameter is recommended to be chosen such that it is approximately equal to the training set size divided by steps per epoch. With a chosen 100 steps per epoch and training set size (4,131), the batch size of 42 was chosen.  

* num_epochs: (number of epochs) The number of times the entire training dataset gets propagated through the network.
> Increasing the number of epochs for training improves the training accuracy without the requirement of more data. Generally, increasing this number will increase the accuracy up to a certain peak level. This parameter was set to 20 epochs.

* steps_per_epoch: The number of batches of training images that go through the network in 1 epoch.
> The general rule of thumb is to set this parameter to the training set size (4,131) divided by the batch size. This was chosen to be 100 steps while the batch size was adjusted accordingly.

* validation_steps: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.
> Along the same lines as the rule of thumb for the steps per epoch, this parameters should be set to the validation set size (1,184) divided by the batch size (42). This results in a validation step number of 29.

The final tuning parameters defined to generate a model that meets the passing requirement are given below.

```
learning_rate = 0.001
batch_size = 42
num_epochs = 20
steps_per_epoch = 100
validation_steps = 29
```

#### 1x1 Convolutions vs. Fully Connected Layers

A 1x1 convolutional layer preserves spatial information (pixel location) by ensuring that the output is 4D whereas a fully connected layer flattens out the input and loses the spatial information. The loss of spatial information would reduce efficacy of the decoding layers that follow. A fully connected layer is used for situations where it is only required to determine if the target is present and does not matter *where* the target it. If the position of the target is of importance, it is better to use 1x1 convolutions.

#### Encoder and Decoder Blocks

The encoder blocks in the network architecture extract features from each of the images while the decoder blocks upsamples the 1x1 convolutional layer to the same size as the original image, including spatial information. The result is that the final output image contains segmentations of each pixel from the original image.

#### Trained Model Limitations

The model provided in this project was trained specifically to detect the human target. The current model and data would not work well to detect other objects. However, the same network architecture can be used to train a network to recognize other objects. New training and validation data sets would need to be captured in order to train a new model to detect and follow a different object.

### Follow Me Deep Learning Model Results

The generated [model_weights.h5](https://github.com/kevinfructuoso/Deep-Learning-Drone-Follower/blob/master/data/weights/model_weights) file contains the resulting weights from the trained model. The performance and detection overlay of the training was tested and can be observed in the following .gif images.

The model is capable of reliably and repeatedly detecting the actual target from relatively close ddistances. As seen in the above images, it is very good at detecting the target from up close and continuing to follow the target.

![performance example][image0]
![pyqt overlay][image1]


The model training resulted in a ~41.5% accuracy metric. While not much higher than the requirement, the model performs fairly well at locating and following the target better than the accuracy measurement suggests.

### Improvements

There are a handful of areas in which this model can be improved. From the model scoring results, it is clear that there are two glaring weaknesses in this model.

1. Detecting other objects as the target
2. Detecting the target from very far away

These can be addressed by capturing and implementing a more rigorous data set to train the model for these specific instances. It should also be noted from the PyQt detection overlay graph that the model sometimes does not fully recognize other people and displays them with patchy results. Further model training should address this issue as well.