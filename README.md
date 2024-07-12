# PRODIGY_ML_04


# Hand Gesture Recognition using CNN

This repository contains code for hand gesture recognition using a Convolutional Neural Network (CNN). The dataset used is the LeapGestRecog dataset from Kaggle.

## Dataset

- **Source**: [LeapGestRecog Dataset on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- **Description**: This dataset contains images of 10 different hand gestures.

## Installation

To run this code, you need the following libraries installed:

- `pandas`
- `numpy`
- `opencv-python`
- `matplotlib`
- `keras`
- `tensorflow`

You can install these libraries using `pip`:

```bash
pip install pandas numpy opencv-python matplotlib keras tensorflow
```

## Usage

1. Download the LeapGestRecog dataset from Kaggle and place it in the same directory as the script.
2. Extract the dataset.
3. Run the script to preprocess the data, train the CNN model, and evaluate its performance.

## Key Steps

1. **Loading and Preprocessing Data**:
    - Load images from the dataset and resize them to 50x50 pixels.
    - Convert images to grayscale and normalize pixel values.

2. **Model Architecture**:
    - A CNN with two convolutional layers, followed by a max-pooling layer and a dropout layer.
    - The model is flattened and passed through two dense layers, with the final output layer having 10 units (one for each gesture).

3. **Model Training**:
    - The model is trained for 4 epochs with a batch size of 32.
    - Training and validation accuracies are plotted.

4. **Model Evaluation**:
    - The test accuracy is calculated and displayed.
    - Training and validation loss and accuracy are plotted.

## Results

- **Test Accuracy**: 99.95%
- The model achieves high accuracy in recognizing hand gestures.

