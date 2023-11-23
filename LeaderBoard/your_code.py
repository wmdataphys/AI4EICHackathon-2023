import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the MLP
input_dim = 10
hidden_dim = 20
output_dim = 1
model = MLP(input_dim, hidden_dim, output_dim)

# Define the loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create some random input data
input_data = torch.from_numpy(np.random.rand(1, input_dim)).float()

# Forward pass
output = model(input_data)

# Compute loss
target = torch.from_numpy(np.random.rand(1, output_dim)).float()
loss = criterion(output, target)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()