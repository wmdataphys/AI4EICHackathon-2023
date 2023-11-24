import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create a simple MLP class
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

# Define the input size, hidden layer size and output size
input_size = 10
hidden_size = 20
output_size = 5

# Create an instance of the MLP model
model = MLP(input_size, hidden_size, output_size)

# Define some dummy input data
input_data = torch.randn(1, input_size)

# Obtain the output from the model
output = model(input_data)

# Print the output
print(output)
