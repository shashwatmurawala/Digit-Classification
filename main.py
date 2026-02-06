# Import PyTorch and other relevant libraries
from cProfile import label
import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import mitdeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Constants
COMET_API_KEY = ""

# ============================================================================
# MODEL CLASSES
# ============================================================================

class FullyConnectedModel(nn.Module):
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
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        # First max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(24, 36, kernel_size=3)
        # Second max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(36 * 5 * 5, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional and pooling layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second convolutional and pooling layers
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
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train(model, dataloader, criterion, optimizer, device, epochs):
    """Train the model for a specified number of epochs."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_pred = 0
        total_pred = 0

        for images, labels in dataloader:
            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Clear gradients before performing backward pass
            optimizer.zero_grad()
            # Calculate loss based on model predictions
            loss = criterion(outputs, labels)
            # Backpropagate and update model parameters
            loss.backward()
            optimizer.step()

            # Multiply loss by total nos. of samples in batch
            total_loss += loss.item() * images.size(0)

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct_pred += (predicted == labels).sum().item()
            total_pred += labels.size(0)

        # Compute metrics
        total_epoch_loss = total_loss / total_pred
        epoch_accuracy = correct_pred / total_pred
        print(f"Epoch {epoch + 1}, Loss: {total_epoch_loss}, Accuracy: {epoch_accuracy:.4f}")


def evaluate(model, dataloader, criterion, device):
    """Evaluate model performance on the test dataset."""
    model.eval()
    test_loss = 0
    correct_pred = 0
    total_pred = 0
    
    # Disable gradient calculations when in inference mode
    with torch.no_grad():
        for images, labels in dataloader:
            # Ensure evaluation happens on the GPU
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

# Setup device
device = torch.device("mps")

# Initialize Comet.ml experiment
comet_model_1 = comet_ml.Experiment(api_key=COMET_API_KEY, project_name="digit-classification")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
BATCH_SIZE = 64
trainset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================================================================
# FULLY CONNECTED MODEL TRAINING
# ========================================================================

print("\n" + "="*70)
print("TRAINING FULLY CONNECTED MODEL")
print("="*70 + "\n")

# Initialize model
fc_model = FullyConnectedModel().to(device)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
fc_optimizer = optim.Adam(fc_model.parameters(), lr=0.001)

# Train the model
FC_EPOCHS = 5
train(fc_model, trainset_loader, loss_function, fc_optimizer, device, FC_EPOCHS)

# End Comet experiment for FC model
comet_model_1.end()

# Evaluate the fully connected model
test_loss, test_acc = evaluate(fc_model, testset_loader, loss_function, device)
print(f'\nFC Model Test Accuracy: {test_acc:.4f}')

# ========================================================================
# CNN MODEL TRAINING
# ========================================================================

print("\n" + "="*70)
print("TRAINING CNN MODEL")
print("="*70 + "\n")

# Initialize CNN model
cnn_model = CNN().to(device)

# Test the model with a sample image
image, label = train_dataset[0]
image = image.to(device).unsqueeze(0)  # Add batch dimension → Shape: (1, 1, 28, 28)
output = cnn_model(image)

# Print the model architecture
print(cnn_model)

# Define hyperparameters for CNN
CNN_EPOCHS = 7
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2)
cnn_loss_function = nn.CrossEntropyLoss()

# Train CNN model
train(cnn_model, trainset_loader, cnn_loss_function, cnn_optimizer, device, CNN_EPOCHS)
cnn_test_loss, cnn_test_acc = evaluate(cnn_model, testset_loader, cnn_loss_function, device)
print(f'\nCNN Model Test Accuracy: {cnn_test_acc:.4f}')

loss_history = mdl.util.LossHistory(smoothing_factor=0.95) # to record the evolution of the loss
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')

# Initialize new comet experiment
comet_ml.init(project_name="6.s191lab2_part1_CNN")
comet_model_2 = comet_ml.Experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

# Training loop!
cnn_model.train()

for epoch in range(epochs):
    total_loss = 0
    correct_pred = 0
    total_pred = 0

    # First grab a batch of training data which our data loader returns as a tensor
    for idx, (images, labels) in enumerate(tqdm(trainset_loader)):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        # TODO: feed the images into the model and obtain the predictions
        logits = cnn_model(images)

        # TODO: compute the categorical cross entropy loss using the predicted logits
        loss = cnn_loss_function(images, labels)

        # Get the loss and log it to comet and the loss_history record
        loss_value = loss.item()
        comet_model_2.log_metric("loss", loss_value, step=idx)
        loss_history.append(loss_value) # append the loss to the loss_history record
        plotter.plot(loss_history.get())

        # Backpropagation/backward pass
        '''TODO: Compute gradients for all model parameters and propagate backwads
            to update model parameters. remember to reset your optimizer!'''
        # TODO: reset optimizer
        # TODO: compute gradients
        # TODO: update model parameters

        cnn_optimizer.zero_grad()
        loss.backward()
        cnn_optimizer.step()


        # Get the prediction and tally metrics
        predicted = torch.argmax(logits, dim=1)
        correct_pred += (predicted == labels).sum().item()
        total_pred += labels.size(0)

    # Compute metrics
    total_epoch_loss = total_loss / total_pred
    epoch_accuracy = correct_pred / total_pred
    print(f"Epoch {epoch + 1}, Loss: {total_epoch_loss}, Accuracy: {epoch_accuracy:.4f}")

comet_model_2.log_figure(figure=plt)