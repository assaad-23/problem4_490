# Feed-Forward Neural Network for Iris Classification

This repository contains a simple feed-forward neural network implemented in PyTorch to classify the Iris dataset.

## Dataset
The [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a classic dataset for classification tasks. It has 150 samples, with four features per sample and three target classes (species of iris flowers).

### Features:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

### Target Classes:
- Iris setosa
- Iris versicolor
- Iris virginica

## Model Architecture
The neural network is a fully connected feed-forward network with:
- Input layer: 4 nodes (for the 4 features)
- One hidden layer: 10 nodes, using ReLU activation
- Output layer: 3 nodes (for each target class), with Softmax activation

The model uses **CrossEntropyLoss** for classification and the **Adam optimizer** for training.

## Code Structure
- `neural_network.py`: Contains the `NeuralNetwork` class and helper functions for training and evaluating the model.
- `train.ipynb`: Jupyter notebook with example usage of the model on the Iris dataset.

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/assaad-23/problem4_490
cd problem4_490
