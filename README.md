# Handwritten Digit Classification with TensorFlow and Keras

This project demonstrates how to build, train, and evaluate a neural network model for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras. The MNIST dataset is a widely used benchmark in the field of machine learning, consisting of 28x28 grayscale images of handwritten digits (0-9).

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The goal of this project is to build a neural network model that can accurately classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras, and the results are visualized to demonstrate its performance.

### Dataset
The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits. Each image is a 28x28 grayscale image, and the task is to classify each image into one of the 10 digit classes (0-9).

---

## Key Features

- **Data Preprocessing**: Normalization of pixel values to the range [0, 1].
- **Neural Network Model**: A simple feedforward neural network with two hidden layers and a softmax output layer.
- **Training**: The model is trained using the Adam optimizer and evaluated on a validation set.
- **Evaluation**: The model's performance is tested on the MNIST test set, and predictions are visualized.
- **Visualization**: Sample images from the test set are displayed along with their predicted labels.

---

## Installation

To run this project, you need to have Python installed along with the following libraries:

- TensorFlow
- NumPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib
```
## Usage
Clone the repository:

```bash
git clone https://github.com/George-A-saed/hand-written-numbers-recognition.git
cd handwritten-digit-classification
```
Run the Jupyter notebook:

```bash
jupyter notebook handwritten_digit_classification.ipynb
```
Follow the steps in the notebook to load the dataset, build the model, train it, and evaluate its performance.

## Model Architecture
The neural network model consists of the following layers:

Input Layer: A Flatten layer to convert the 28x28 images into a 1D array of 784 pixels.

Hidden Layer 1: A Dense layer with 256 units and ReLU activation.

Hidden Layer 2: A Dense layer with 128 units and ReLU activation.

Output Layer: A Dense layer with 10 units (one for each digit) and softmax activation.

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

## Training and Evaluation
Training: The model is trained for 20 epochs with a batch size of 2000. A validation split of 20% is used to monitor the model's performance during training.

Evaluation: The model is evaluated on the MNIST test set, achieving high accuracy. Predictions are made on the test images, and the results are visualized.

## Results
The model achieves high accuracy on both the training and test sets, demonstrating its ability to classify handwritten digits effectively. Below are some sample predictions from the test set:

Sample Predictions 
![An Image of the predicton sample](https://github.com/MemaroX/hand-written-numbers-recognition/blob/main/image.png)

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Commit your changes and push to the branch.

Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
The MNIST dataset is provided by Yann LeCun.

This project was inspired by various tutorials and resources available in the TensorFlow and Keras documentation.

Feel free to explore the code, experiment with the model, and contribute to the project! If you have any questions or suggestions, please open an issue or contact the maintainers.

