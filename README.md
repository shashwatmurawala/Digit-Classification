# MNIST Digit Classification with CNN

A PyTorch implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. This project demonstrates deep learning fundamentals including CNN architecture, training loops, and experiment tracking with Comet.ml.

## 📋 Table of Contents

- [Overview](#overview)
- [What This Code Does](#what-this-code-does)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Execute](#how-to-execute)
- [Expected Output](#expected-output)
- [Code Structure Explanation](#code-structure-explanation)
- [Results](#results)

## 🎯 Overview

This project implements a CNN to recognize handwritten digits (0-9) from the MNIST dataset. The MNIST dataset contains 60,000 training images and 10,000 test images of handwritten digits, each 28×28 pixels in grayscale.

**Key Features:**
- Convolutional Neural Network with 2 conv layers
- Training with real-time loss visualization
- Experiment tracking via Comet.ml
- Prediction visualizations
- Achieves ~98-99% accuracy on test set

## 🔍 What This Code Does

The script performs the following steps:

1. **Data Loading**: Downloads and loads the MNIST dataset
2. **Model Creation**: Builds a CNN with convolutional and pooling layers
3. **Training**: Trains the model for 7 epochs with SGD optimizer
4. **Evaluation**: Tests the model on unseen test data
5. **Visualization**: Creates prediction plots and probability distributions
6. **Logging**: Tracks metrics and visualizations to Comet.ml

## 🏗️ Model Architecture

### CNN Architecture

```
Input: 28×28 grayscale image
    ↓
Conv2D (1→24 channels, 3×3 kernel) → ReLU → MaxPool (2×2)
    ↓
Conv2D (24→36 channels, 3×3 kernel) → ReLU → MaxPool (2×2)
    ↓
Flatten → Linear (900→128) → ReLU → Linear (128→10)
    ↓
Output: 10 class scores (digits 0-9)
```

**Layer Details:**
- **Conv Layer 1**: Input channels=1, Output channels=24, Kernel=3×3
  - After pooling: 24×13×13
- **Conv Layer 2**: Input channels=24, Output channels=36, Kernel=3×3
  - After pooling: 36×5×5
- **Fully Connected Layer 1**: 900 → 128 neurons
- **Fully Connected Layer 2**: 128 → 10 neurons (output classes)

### Alternative: Fully Connected Model

The code also includes a simpler `FullyConnectedModel` class for comparison:
- Flattens 28×28 image → 784 features
- Linear(784→128) → ReLU → Linear(128→10)
- Less accurate than CNN but faster to train

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries

```
torch>=1.9.0
torchvision>=0.10.0
torchsummary
comet-ml
matplotlib
numpy
tqdm
mitdeeplearning
```

## 🚀 Installation

### Step 1: Clone or Navigate to Repository

```bash
cd /Users/shashwatmurawala/Desktop/Repos/Digit-Classification
```

### Step 2: Install Dependencies

**Option A: Using pip**
```bash
pip3 install torch torchvision torchsummary comet-ml matplotlib numpy tqdm mitdeeplearning
```

**Option B: Using requirements file (if you create one)**
```bash
pip3 install -r requirements.txt
```

### Step 3: Set Up Comet.ml

1. Create a free account at [https://www.comet.ml](https://www.comet.ml)
2. Get your API key from your account settings
3. Update the `COMET_API_KEY` in `main.py` with your key (currently uses placeholder)

**Note**: The MNIST dataset will be automatically downloaded the first time you run the script (will be saved in `./data` folder).

## ▶️ How to Execute

### Basic Execution

```bash
# Navigate to project directory
cd /Users/shashwatmurawala/Desktop/Repos/Digit-Classification

# Run the training script
python3 main.py
```

### What Happens When You Run It:

1. **Device Setup**: Configures to use MPS (Metal Performance Shaders) for Mac M1/M2 GPUs
2. **Data Download**: Downloads MNIST if not already present
3. **Model Initialization**: Creates CNN and prints architecture
4. **Training**: Trains for 7 epochs with progress bars
5. **Evaluation**: Tests on 10,000 test images
6. **Visualization**: Generates and logs prediction plots to Comet.ml

### Execution Time

- **First run**: ~3-5 minutes (includes dataset download)
- **Subsequent runs**: ~2-3 minutes (on Mac M1/M2)

## 📊 Expected Output

### Console Output

```
======================================================================
TRAINING CNN MODEL
======================================================================

CNN(
  (conv1): Conv2d(1, 24, kernel_size=(3, 3), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(24, 36, kernel_size=(3, 3), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=900, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)

100%|████████████████████| 938/938 [00:45<00:00, 20.84it/s]
Epoch 1/7, Accuracy: 0.9123
100%|████████████████████| 938/938 [00:43<00:00, 21.45it/s]
Epoch 2/7, Accuracy: 0.9687
...
Epoch 7/7, Accuracy: 0.9891

======================================================================
EVALUATING CNN MODEL
======================================================================

Test Accuracy: 0.9856
Test Loss: 0.0456

Single Image Prediction: 7, Actual Label: 7

======================================================================
TRAINING COMPLETE - Check Comet.ml for visualizations!
======================================================================
```

### Generated Visualizations

The script creates and logs the following to Comet.ml:

1. **Loss Curve**: Real-time training loss over iterations
2. **Sample Prediction**: Single digit with predicted vs actual label
3. **Single Prediction Detail**: Image + probability distribution bar chart
4. **Multiple Predictions Grid**: 5×4 grid showing 20 predictions with probabilities

## 📖 Code Structure Explanation

### Main Components

#### 1. **Imports and Constants** (Lines 1-14)
- PyTorch libraries for neural networks
- Data loading and transformations
- Visualization and logging tools
- Comet.ml API key for experiment tracking

#### 2. **Model Classes** (Lines 16-77)

**`FullyConnectedModel`**: 
- Simple baseline model (not actively used but available)
- Flattens image and passes through 2 linear layers

**`CNN`**: 
- Main model with 2 convolutional blocks
- Each block: Conv → ReLU → MaxPool
- Followed by fully connected layers

#### 3. **Evaluation Function** (Lines 79-118)
- Tests model on validation/test data
- Computes loss and accuracy
- Runs in `torch.no_grad()` mode for efficiency

#### 4. **Main Execution** (Lines 120-308)

**Setup** (Lines 120-144):
- Device configuration (MPS for Mac M1/M2)
- Data transformations
- Dataset loading with DataLoaders

**Training Loop** (Lines 146-209):
- Initializes CNN and optimizer
- Comet.ml experiment tracking
- 7 epochs of training with:
  - Forward pass (predictions)
  - Loss calculation (CrossEntropyLoss)
  - Backward pass (gradient computation)
  - Weight updates (SGD optimizer)
  - Metrics logging

**Evaluation** (Lines 211-222):
- Runs model on test set
- Prints final accuracy and loss

**Visualization** (Lines 224-305):
- Single image prediction example
- Batch prediction collection
- Creates multiple visualization plots
- Logs everything to Comet.ml

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 64 | Number of images per training batch |
| `CNN_EPOCHS` | 7 | Number of complete passes through training data |
| `Learning Rate` | 0.01 | Step size for SGD optimizer |
| `Optimizer` | SGD | Stochastic Gradient Descent |
| `Loss Function` | CrossEntropyLoss | Standard for multi-class classification |

### Device Configuration

The code uses:
- **MPS** (Metal Performance Shaders): For Mac M1/M2 GPUs
- To use **CUDA** (NVIDIA GPUs): Change line 120 to `device = torch.device("cuda")`
- To use **CPU**: Change line 120 to `device = torch.device("cpu")`

## 📈 Results

### Expected Performance

- **Training Accuracy**: ~98-99% after 7 epochs
- **Test Accuracy**: ~98-99%
- **Training Time**: ~2-3 minutes on Mac M1/M2

### What Makes the CNN Better than Fully Connected?

CNNs work better for image data because:
1. **Spatial structure preservation**: Neighboring pixels are related
2. **Parameter sharing**: Same filter applied across entire image
3. **Translation invariance**: Can recognize digits anywhere in the image
4. **Hierarchical features**: Early layers detect edges, later layers detect shapes

## 🔧 Customization

### Change Number of Epochs

In `main.py`, line 162:
```python
CNN_EPOCHS = 7  # Change to desired number
```

### Modify Learning Rate

In `main.py`, line 163:
```python
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2)  # Change lr value
```

### Use Different Optimizer

Replace SGD with Adam for potentially faster convergence:
```python
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
```

### Switch to Fully Connected Model

Uncomment the FC model training section and comment out the CNN section (not recommended, CNN performs better).

## 📝 Notes

- The MNIST dataset will be downloaded to `./data/MNIST/` on first run (~50MB)
- Comet.ml experiments can be viewed at [https://www.comet.ml](https://www.comet.ml)
- Loss plots update every 2 seconds during training
- All plots are automatically logged to your Comet.ml dashboard

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'mitdeeplearning'`
- **Solution**: Install via `pip3 install mitdeeplearning`

**Issue**: `RuntimeError: MPS backend not available`
- **Solution**: Change device to CPU: `device = torch.device("cpu")`

**Issue**: Dataset download fails
- **Solution**: Check internet connection, or manually download MNIST and place in `./data/`

**Issue**: Comet.ml authentication fails
- **Solution**: Update `COMET_API_KEY` with your valid API key

## 📚 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CNN Explained](https://cs231n.github.io/convolutional-networks/)
- [Comet.ml Docs](https://www.comet.ml/docs/)

---

**Happy Learning! 🎓**
