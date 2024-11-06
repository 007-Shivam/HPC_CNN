# HPC on CNN

This project demonstrates a convolutional neural network (CNN) model for classifying handwritten digits in the MNIST dataset using high-performance computing (HPC) techniques. The model is trained with TensorFlow and leverages multiple GPUs to significantly improve training speed and efficiency.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [License](#license)

## Overview

The goal of this project is to build and train a CNN model for digit classification on the MNIST dataset, optimized using multi-GPU training with TensorFlow's `MirroredStrategy`. By distributing computations across multiple GPUs, we can achieve faster convergence and process larger datasets more efficiently.

## Features

- **Multi-GPU Training**: Uses `tf.distribute.MirroredStrategy` to distribute the training workload across available GPUs.
- **Efficient Data Pipeline**: Utilizes the `tf.data` API for efficient data loading with shuffling, batching, and prefetching.
- **Model Architecture**: A simple CNN architecture with two convolutional layers, a max-pooling layer, and two dense layers.
- **Automatic Dataset Normalization**: Scales pixel values of the MNIST dataset to the range [0, 1].
- **Performance Monitoring**: Tracks training and validation accuracy over time, with the ability to record training duration.

## Setup

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Matplotlib for plotting accuracy results

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/007-Shivam/HPC_CNN.git
    cd HPC_CNN
    ```

2. Install required packages:
    ```bash
    pip install tensorflow matplotlib
    ```

## License
This project is licensed under the MIT License.
