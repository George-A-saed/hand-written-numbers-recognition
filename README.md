
# Handwritten Digit Classification with TensorFlow and Keras

This project demonstrates how to build, train, and evaluate a neural network model for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras. Additionally, it includes a Tkinter-based drawing interface where users can draw digits and get real-time predictions from the trained model.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Interactive Drawing Interface](#interactive-drawing-interface)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to build a neural network model that can accurately classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras, and the results are visualized to demonstrate its performance. Additionally, a Tkinter-based drawing interface is provided for users to draw digits and get real-time predictions.

### Dataset
The MNIST dataset contains:
- 60,000 training images
- 10,000 test images

Each image is a 28x28 grayscale image, and the task is to classify each image into one of the 10 digit classes (0-9).

---

## Key Features

- **Data Preprocessing**: Normalization of pixel values to the range [0, 1].
- **Neural Network Model**: A simple feedforward neural network with two hidden layers and a softmax output layer.
- **Training**: The model is trained using the Adam optimizer and evaluated on a validation set.
- **Evaluation**: The model's performance is tested on the MNIST test set, achieving high accuracy.
- **Interactive Drawing Interface**: A Tkinter-based interface allows users to draw digits and get real-time predictions.
- **Visualization**: Sample images from the test set are displayed along with their predicted labels.

---

## Installation

To run this project, you need to have Python installed along with the following libraries:

- TensorFlow
- NumPy
- Matplotlib
- Pillow
- Tkinter

You can install the required libraries using pip:

```
pip install tensorflow numpy matplotlib pillow
```
----------

## Usage

1.  Clone the repository:
    
    ```bash
    git clone https://github.com/your-username/handwritten-digit-classification.git
    cd handwritten-digit-classification
    ```
2.  Run the Jupyter notebook:
    
    ```bash
    jupyter notebook handwritten_digit_classification.ipynb
    ```
3.  Follow the steps in the notebook to:
    
    -   Load the dataset
        
    -   Build the model
        
    -   Train the model
        
    -   Evaluate its performance
        
4.  To use the interactive drawing interface:
    
    -   Run the Tkinter-based drawing interface code in the notebook.
        
    -   Draw a digit on the canvas and click "Predict" to see the model's prediction.
        



## Model Architecture

The neural network model consists of the following layers:

1.  **Input Layer**: A  `Flatten`  layer to convert the 28x28 images into a 1D array of 784 pixels.
    
2.  **Hidden Layer 1**: A  `Dense`  layer with 256 units and ReLU activation.
    
3.  **Hidden Layer 2**: A  `Dense`  layer with 128 units and ReLU activation.
    
4.  **Output Layer**: A  `Dense`  layer with 10 units (one for each digit) and softmax activation.
    

The model is compiled using:

-   **Optimizer**: Adam
    
-   **Loss Function**: Sparse Categorical Cross-Entropy
    
-   **Metric**: Accuracy
    

----------

## Training and Evaluation

-   **Training**: The model is trained for 5 epochs with a batch size of 64.
    
-   **Evaluation**: The model is evaluated on the MNIST test set, achieving high accuracy. Predictions are made on the test images, and the results are visualized.
    

----------

## Interactive Drawing Interface

The project includes a Tkinter-based drawing interface where users can:

-   Draw a digit on a 20x20 canvas.
    
-   Click "Predict" to get the model's prediction for the drawn digit.
    
-   Click "Clear" to reset the canvas and draw a new digit.
    

The drawn image is preprocessed and resized to 28x28 pixels before being fed into the model for prediction.
![Sample UI](https://github.com/MemaroX/hand-written-numbers-recognition/blob/main/image.png)

----------

## Results

The model achieves high accuracy on both the training and test sets, demonstrating its ability to classify handwritten digits effectively. Below are some sample predictions from the test set:

![Sample Predictions](https://github.com/MemaroX/hand-written-numbers-recognition/raw/main/image.png)  

----------

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
    
2.  Create a new branch for your feature or bugfix.
    
3.  Commit your changes and push to the branch.
    
4.  Submit a pull request.
    

----------

## License

This project is licensed under the MIT License. See the  [LICENSE](https://license/)  file for details.

----------

## Acknowledgments

-   The MNIST dataset is provided by  [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
    
-   This project was inspired by various tutorials and resources available in the TensorFlow and Keras documentation.
-  Programming with [Maher Gomaa Ismaeel](https://github.com/MemaroX) and [Robair Nashaat](https://github.com/BanditRN)

----------

Feel free to explore the code, experiment with the model, and contribute to the project! If you have any questions or suggestions, please open an issue or contact the maintainers.



