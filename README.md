# Neural Network for Regression Task

## Overview

This repository contains an implementation of a modular neural network for a regression task, with a focus on predicting the median house value using the California House Prices Dataset. The project is divided into two main parts:

1. **Neural Network Mini-Library**: A low-level neural network implementation built from scratch using NumPy.
2. **Regression Model**: A neural network architecture designed for predicting house prices, trained using either PyTorch or the custom neural network library developed in Part 1.

## Data

The dataset used for the regression task is the California House Prices Dataset, which contains information on various attributes of block groups in California, including:

- Longitude
- Latitude
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Median income
- Ocean proximity
- Median house value (target variable)

The dataset is provided in CSV format and can be loaded and preprocessed using Pandas.

## Features

### Part 1: Neural Network Mini-Library

The first part of the project involves building a modular neural network library using NumPy, including the following components:

- **Linear Layer**: Implements an affine transformation, with methods for forward pass, backward pass, and parameter updates.
- **Activation Functions**: Includes Sigmoid and ReLU activation layers with forward and backward methods.
- **Multi-Layer Network**: Combines multiple linear layers and activations into a neural network architecture.
- **Trainer**: A class that handles data shuffling, minibatch gradient descent, and training the network over multiple epochs.
- **Preprocessor**: A class that implements min-max scaling for input normalization.

### Part 2: Regression Model

In the second part, a neural network is created and trained to predict house prices using either PyTorch or the custom mini-library. Key components include:

- **Data Preprocessing**: Handling missing values, normalizing numerical data, and encoding categorical features (such as ocean proximity) using one-hot encoding.
- **Model Training**: The model is trained using gradient descent, with options for minibatch training and shuffling.
- **Evaluation**: Methods for model evaluation and prediction, using standard regression metrics.

### Hyperparameter Tuning

A hyperparameter search is performed to find the optimal configuration for the neural network, including parameters such as learning rate, number of layers, and the number of neurons per layer.

## Usage

1. Clone the repository and navigate to the project directory.
2. Ensure you have the necessary dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```
3. Preprocess the dataset and train the model:
    ```python
    from src.part2_house_value_regression import Regressor
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
    ```
4. Save the trained model:
    ```python
    from src.part2_house_value_regression import save_regressor
    save_regressor(regressor)
    ```
5. Use the trained model to make predictions:
    ```python
    predictions = regressor.predict(x_test)
    ```

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn (for preprocessing utilities)
- PyTorch (if using PyTorch for training)
