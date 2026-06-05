# Oil Spill Detection Using Machine Learning

A Convolutional Neural Network (CNN) that detects oil spills in satellite imagery to support environmental monitoring and disaster response. The model is trained on both RGB and Infrared (IR) imagery, enabling rapid, automated detection of oil spills, particularly in port environments where fast response (ideally within 30 minutes) is critical.

## Overview

Most oil spills are small to medium scale and occur in or near ports, where detection still relies heavily on manual visual inspection and is slow, especially at night. This project automates detection using a CNN trained on segmented RGB images and raw IR images, learning the color, texture, and thermal signatures associated with oil spills.

## Features

- CNN-based binary classification of oil spill vs. non-spill imagery
- Combined RGB + IR input for richer feature extraction
- Data augmentation to improve generalization
- Dropout regularization to reduce overfitting
- Training/validation accuracy and loss tracking with visualization
- Confusion matrix evaluation on test data

## Model Architecture

A `Sequential` Keras model with alternating feature-extraction and pooling layers, followed by fully connected layers for classification:

- **Conv2D** layers with ReLU activation for feature extraction
- **MaxPooling2D** layers to reduce spatial dimensions and computation
- **Dropout** layers for regularization
- **Flatten** layer to transition to fully connected layers
- **Dense** layers, with a **Softmax** output for class probabilities

**Training configuration:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metric: Categorical Accuracy
- Callback: Early Stopping (monitors loss to prevent overfitting)

## Method

### Training

1. RGB images are segmented to produce masks isolating oil-affected areas (ground truth labels).
2. Segmented RGB images and raw IR images are preprocessed (pixel values rescaled to 0–1).
3. `ImageDataGenerator` applies augmentation (rescaling, shearing, zooming, horizontal flip, width/height shifting).
4. `flow_from_directory` loads images and splits them into training and validation sets via `validation_split`.
5. The model is trained with backpropagation, validated each epoch, and stopped early if loss stops improving.

### Inference

The trained model is deployed on a low-power inference device optimized for GPU computation, enabling real-time segmentation and detection of oil spills in new imagery.

## Results

- Training and validation accuracy increase and stabilize after roughly 20 epochs, indicating good convergence and generalization.
- Training and validation loss decrease steadily, with the gap between them used to monitor overfitting.
- Confusion matrix analysis is used to assess true/false positives and negatives on the test set.

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn (test accuracy evaluation)
- NumPy, Matplotlib (data handling and visualization)

## Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn numpy matplotlib
```

### Dataset Structure

Organize your images so `flow_from_directory` can load them by class:

```
base_dir/
├── oil_spill/
│   ├── image1.jpg
│   └── ...
└── no_spill/
    ├── image1.jpg
    └── ...
```

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Place your dataset in `base_dir/` following the structure above.
3. Run training:
   ```bash
   python train.py
   ```
4. Run inference on new images:
   ```bash
   python predict.py --image path/to/image.jpg
   ```

## Authors

- **Markand Joshi** — Electronics & Communication Department, Nirma University
- **Rudra Malaviya** — Electronics & Communication Department, Nirma University

## License

This project is released under the MIT License. See the `LICENSE` file for details.
