# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the Iris dataset
data = load_iris()
X, y = data['data'], data['target']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

# Define a more advanced neural network model
class AdvancedNeuralNetwork(nn.Module):
    def __init__(self):
        super(AdvancedNeuralNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(4, 16)        # First hidden layer
        self.bn1 = nn.BatchNorm1d(16)      # Batch normalization after first layer
        self.fc2 = nn.Linear(16, 32)       # Second hidden layer
        self.bn2 = nn.BatchNorm1d(32)      # Batch normalization after second layer
        self.fc3 = nn.Linear(32, 16)       # Third hidden layer
        self.fc4 = nn.Linear(16, 3)        # Output layer
        
        # Define dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Pass through layers with activations, batch norm, and dropout
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer (no activation, handled by CrossEntropyLoss)
        return x

# Initialize the model, loss function, and optimizer
model = AdvancedNeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce learning rate by a factor of 0.1 every 10 epochs

# Training function
def train(model, loader, criterion, optimizer, scheduler, epochs=30):
    for epoch in range(epochs):
        model.train()  # Set to training mode
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # Zero gradients
            outputs = model(batch_x)  # Forward pass
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
        scheduler.step()  # Update learning rate
        
        # Print loss at the end of each epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Train the model
train(model, train_loader, criterion, optimizer, scheduler)

# Evaluation function
def evaluate(model, loader):
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient tracking
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model on the test set
evaluate(model, test_loader)
