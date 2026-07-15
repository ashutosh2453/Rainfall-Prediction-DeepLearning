# Models Directory

This folder contains the PyTorch model architectures for rainfall prediction.

## Available Architectures

### 1. CNN (Baseline) — `cnn.py`
* **Description**: A basic Convolutional Neural Network (CNN) architecture used as a baseline model.
* **Details**: Extracts spatial features from the 2D grid input maps using standard 2D convolutional layers.

### 2. CNN-BiLSTM — `cnn_bilstm.py`
* **Description**: A hybrid model combining Convolutional Neural Networks and Bidirectional Long Short-Term Memory (BiLSTM) networks.
* **Details**: CNN layers extract spatial features from the grid maps, which are then passed to the BiLSTM layers to capture temporal context in both forward and backward directions.

### 3. CNN-BiLSTM with Attention — `cnn_bilstm_attention.py`
* **Description**: An advanced model combining CNN and BiLSTM with a custom Attention mechanism.
* **Details**: CNN layers extract spatial features, BiLSTM layers process the temporal sequence, and the Attention layer dynamically weights the most critical time steps, improving forecasting accuracy for extreme rainfall events.

## Imports and Package Configuration

This folder is configured as a Python package. Make sure to import models using:
```python
from models.cnn import CNN
from models.cnn_bilstm import CNNBiLSTM
from models.cnn_bilstm_attention import CNNBiLSTMAttention
```
