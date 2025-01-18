# MNIST Digit Classification using MLP (Multi-Layer Perceptron)

This project demonstrates the use of a **Multi-Layer Perceptron (MLP)** to classify handwritten digits from the MNIST dataset. The model is built using **TensorFlow** and **Keras** libraries, and the goal is to predict the correct digit (0-9) for each input image.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Evaluation and Prediction](#evaluation-and-prediction)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Project Overview

The **MNIST dataset** contains images of handwritten digits (0-9), each 28x28 pixels in size. The goal is to use these images to train an MLP-based neural network to classify the digits.

### Key Steps:

1. **Data Preprocessing:**
   - Load the MNIST dataset.
   - Normalize the pixel values (scale between 0 and 1).
   - Flatten the 28x28 image into a 784-dimensional vector.
   
2. **Model Architecture:**
   - Build an MLP model with an input layer, hidden layers, and an output layer.
   
3. **Training:**
   - Train the model using the training data.
   - Monitor training and validation loss and accuracy using callbacks.

4. **Evaluation and Prediction:**
   - Evaluate the model on the test dataset.
   - Predict the labels of test samples and compare with ground truth.

## Dataset

The **MNIST** dataset is available in Keras and contains 60,000 training images and 10,000 test images, each labeled with the corresponding digit (0-9). The images are grayscale (1 channel), 28x28 pixels in size.

- **Training Set:** 60,000 images.
- **Test Set:** 10,000 images.

## Model Architecture

The model consists of the following layers:

1. **Input Layer:** Accepts the flattened 28x28 image (784 features).
2. **Hidden Layer 1:** Dense layer with 128 units and ReLU activation function.
3. **Hidden Layer 2:** Dense layer with 64 units and ReLU activation function.
4. **Output Layer:** Dense layer with 10 units (one for each class) and softmax activation to output probabilities for each class.

### Optimizer:
- Adam Optimizer

### Loss Function:
- Categorical Crossentropy (since it's a multi-class classification problem)

### Metrics:
- Accuracy

## Requirements

Make sure you have the following libraries installed:

- **TensorFlow** (for building and training the neural network)
- **NumPy** (for handling data)
- **Matplotlib** (for visualizing results)
- **Keras** (for the model building)

You can install the dependencies using `pip`:

```bash
pip install tensorflow numpy matplotlib
