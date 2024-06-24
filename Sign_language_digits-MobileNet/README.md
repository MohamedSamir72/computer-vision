# MobileNet v1: Convolutional Neural Network for Sign Language Digits
## Overview
MobileNet V1, developed by Andrew G. Howard and colleagues at Google, is an efficient Convolutional Neural Network (CNN) architecture designed for mobile and embedded vision applications. Its lightweight design makes it well-suited for real-time image classification tasks on resource-constrained devices. This README provides a brief overview of using MobileNet V1 for recognizing sign language digits.

## Architecture
### MobileNet V1 uses depthwise separable convolutions to build lightweight deep neural networks. The architecture consists of the following layers:

## Features
#### Efficiency: Depthwise separable convolutions reduce the number of parameters and computational load, making MobileNet V1 efficient for mobile and embedded applications.
#### Width Multiplier (α): Allows the model to be scaled down uniformly at each layer, reducing the number of channels.
#### Resolution Multiplier (ρ): Reduces the input image size, allowing for a trade-off between latency and accuracy.

[MobileNet v1](https://arxiv.org/pdf/1704.04861) for more details
