# Quark-Gluon Classification

This repository contains a deep learning solution for classifying 125x125 matrices in three-channel images of quarks and gluons impinging on a calorimeter. The task involves building and evaluating two different deep learning models: a VGG model with 12 layers and a ResNet-152 model.

## Dataset

The dataset is available at: [Quark-Gluon Dataset](https://cernbox.cern.ch/s/oolDBdQegsITFcv)

### Dataset Description

- **Type**: 125x125 matrices in three-channel images.
- **Classes**: Two classes of particles - Quarks and Gluons.

## Models

### 1. VGG Model

- **Architecture**: VGG with 12 layers.
- **Modification**: Reduced weights in the fully connected (FC) layers to speed up training and optimized architecture for classification.

### 2. ResNet-152 Model

- **Architecture**: ResNet-152.
- **Details**: Utilizes a very deep residual network with 152 layers to achieve high classification performance.

## Training and Evaluation

### Data Splitting

- **Training Set**: 80% of the data.
- **Validation Set**: 20% of the data.

### Training

- Both models are trained on the training set.
- Evaluated on the validation set to ensure no overfitting.

### Evaluation

- The performance of each model is assessed based on classification accuracy.
- Model weights are saved and can be found in the `weights/` directory.

## Code

The code for training and evaluating the models is provided in the Jupyter notebook `Quark_Gluon_Classification.ipynb`. The notebook includes:

- Data preprocessing and augmentation.
- Model definitions (VGG-12 and ResNet-152).
- Training loops.
- Evaluation and result visualization.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL

Install the required libraries using:

```bash
pip install torch torchvision numpy matplotlib pillow
