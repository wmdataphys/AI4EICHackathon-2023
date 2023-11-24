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

# Create an instance of the MLP model
input_size = 10
hidden_size = 20
output_size = 5
model = MLP(input_size, hidden_size, output_size)

# Define a sample input
sample_input = torch.randn(1, input_size)  # Random input with batch size 1

# Forward pass
output = model(sample_input)
print("Model output:", output)
