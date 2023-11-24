import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = 10
hidden_size = 20
output_size = 5
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example usage
input_data = torch.randn(32, input_size)  # Example input data
output = model(input_data)               # Forward pass
target = torch.empty(32, dtype=torch.long).random_(output_size)  # Example target
loss = criterion(output, target)         # Calculate the loss
loss.backward()                          # Backpropagation and gradient update
optimizer.step()                         # Optimizer step
