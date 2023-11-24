import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the MLP
input_size = 10
hidden_size = 20
output_size = 5
model = MLP(input_size, hidden_size, output_size)

# Define some dummy input data
input_data = torch.randn(1, input_size)

# Make a prediction using the model
output = model(input_data)
print(output)
