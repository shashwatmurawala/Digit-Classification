# Import PyTorch and other relevant libraries
import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mitdeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Constants
COMET_API_KEY = ""

# ============================================================================
# MODEL CLASSES
# ============================================================================

class FullyConnectedModel(nn.Module):
    """Simple fully connected neural network for MNIST digit classification."""
    
    def __init__(self):
        super(FullyConnectedModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """Convolutional Neural Network for MNIST digit classification."""
    
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel, 24 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        # First max pooling layer: 2x2 pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Second convolutional layer: 24 input channels, 36 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3)
        # Second max pooling layer: 2x2 pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(36 * 5 * 5, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate(model, dataloader, criterion, device):
    """Evaluate model performance on the test dataset."""
    model.eval()
    test_loss = 0
    correct_pred = 0
    total_pred = 0
    
    # Disable gradient calculations when in inference mode
    with torch.no_grad():
        for images, labels in dataloader:
            # Ensure evaluation happens on the GPU/MPS
            images, labels = images.to(device), labels.to(device)

            # Feed the images into the model and obtain the predictions
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate test loss
            test_loss += loss.item() * images.size(0)

            # Identify the digit with the highest probability prediction
            predicted = torch.argmax(outputs, dim=1)

            # Tally the number of correct predictions
            correct_pred += (predicted == labels).sum().item()

            # Tally the total number of predictions
            total_pred += predicted.size(0)

    # Compute average loss and accuracy
    test_loss /= total_pred
    test_acc = correct_pred / total_pred
    return test_loss, test_acc


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Setup device (use MPS for Mac M1/M2, CUDA for NVIDIA GPUs, or CPU)
device = torch.device("mps")

# Data preprocessing - convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders for batching
BATCH_SIZE = 64
trainset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================================================================
# CNN MODEL TRAINING
# ========================================================================

print("\n" + "="*70)
print("TRAINING CNN MODEL")
print("="*70 + "\n")

# Initialize CNN model and move to device
cnn_model = CNN().to(device)

# Print the model architecture
print(cnn_model)
print()

# Define hyperparameters
CNN_EPOCHS = 7
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2)
cnn_loss_function = nn.CrossEntropyLoss()

# Initialize Comet.ml experiment for tracking
comet_ml.init(project_name="6.s191lab2_part1_CNN")
comet_model = comet_ml.Experiment(api_key=COMET_API_KEY)

# Initialize loss tracking and plotting utilities
loss_history = mdl.util.LossHistory(smoothing_factor=0.95)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')

# Clear any existing tqdm instances
if hasattr(tqdm, '_instances'): 
    tqdm._instances.clear()

# Training loop
cnn_model.train()

for epoch in range(CNN_EPOCHS):
    correct_pred = 0
    total_pred = 0

    # Iterate through batches with progress bar
    for idx, (images, labels) in enumerate(tqdm(trainset_loader)):
        images, labels = images.to(device), labels.to(device)

        # Forward pass: feed images into model and get predictions
        logits = cnn_model(images)

        # Compute the categorical cross entropy loss
        loss = cnn_loss_function(logits, labels)

        # Log loss to Comet.ml and loss history
        loss_value = loss.item()
        comet_model.log_metric("loss", loss_value, step=idx + epoch * len(trainset_loader))
        loss_history.append(loss_value)
        plotter.plot(loss_history.get())

        # Backpropagation: reset gradients, compute gradients, update weights
        cnn_optimizer.zero_grad()
        loss.backward()
        cnn_optimizer.step()

        # Calculate accuracy metrics
        predicted = torch.argmax(logits, dim=1)
        correct_pred += (predicted == labels).sum().item()
        total_pred += labels.size(0)

    # Display epoch metrics
    epoch_accuracy = correct_pred / total_pred
    print(f"Epoch {epoch + 1}/{CNN_EPOCHS}, Accuracy: {epoch_accuracy:.4f}")


# ========================================================================
# MODEL EVALUATION
# ========================================================================

print("\n" + "="*70)
print("EVALUATING CNN MODEL")
print("="*70 + "\n")

# Evaluate model on test set
test_loss, test_acc = evaluate(cnn_model, testset_loader, cnn_loss_function, device)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}\n')

# ========================================================================
# SINGLE IMAGE PREDICTION EXAMPLE
# ========================================================================

# Get first test image and make prediction
test_image, test_label = test_dataset[0]
test_image_tensor = test_image.to(device).unsqueeze(0)

# Put model in evaluation mode
cnn_model.eval()
predictions_test_image = cnn_model(test_image_tensor)

# Get predicted digit (highest probability)
predictions_value = predictions_test_image.cpu().detach().numpy()
prediction = np.argmax(predictions_value)
print(f"Single Image Prediction: {prediction}, Actual Label: {test_label}")

# Visualize single prediction
plt.figure()
plt.imshow(test_image_tensor[0, 0, :, :].cpu(), cmap=plt.cm.binary)
plt.title(f"Prediction: {prediction}, Label: {test_label}")
fig = plt.gcf()
comet_model.log_figure(figure_name="MNIST_sample_images", figure=fig)

# ========================================================================
# BATCH PREDICTIONS AND VISUALIZATION
# ========================================================================

# Collect all predictions for visualization
all_predictions = []
all_labels = []
all_images = []

# Process test set in batches
with torch.no_grad():
    for images, labels in testset_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images)

        # Apply softmax to get probabilities from logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        all_predictions.append(probabilities)
        all_labels.append(labels)
        all_images.append(images)

# Concatenate all batches
all_predictions = torch.cat(all_predictions)  # Shape: (total_samples, num_classes)
all_labels = torch.cat(all_labels)            # Shape: (total_samples,)
all_images = torch.cat(all_images)            # Shape: (total_samples, 1, 28, 28)

# Convert to NumPy for plotting
predictions = all_predictions.cpu().numpy()
test_labels = all_labels.cpu().numpy()
test_images = all_images.cpu().numpy()

# Visualize single prediction with probability distribution
image_index = 79
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
mdl.lab2.plot_image_prediction(image_index, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
mdl.lab2.plot_value_prediction(image_index, predictions, test_labels)
comet_model.log_figure(figure_name="Single_Prediction", figure=plt.gcf())

# Visualize multiple predictions in a grid
num_rows = 5
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    mdl.lab2.plot_image_prediction(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    mdl.lab2.plot_value_prediction(i, predictions, test_labels)
comet_model.log_figure(figure_name="Multiple_Predictions", figure=plt.gcf())

# End Comet experiment
comet_model.end()

print("\n" + "="*70)
print("TRAINING COMPLETE - Check Comet.ml for visualizations!")
print("="*70)