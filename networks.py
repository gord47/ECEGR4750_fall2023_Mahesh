import torch
import torch.nn as nn

# adapted from LearningModel from prior lab
class FullyConnectedNetwork(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int):
        super(FullyConnectedNetwork, self).__init__()
        assert input_dim > 0, "Input dimension must be a positive integer"
        assert hidden_dim > 1, "Hidden dimensions must be an integer greater than 1"
        self.linear1 = nn.Linear(in_features = input_dim, out_features = hidden_dim)
        self.linear2 = nn.Linear(in_features = hidden_dim, out_features = round(hidden_dim//2))
        self.linear3 = nn.Linear(in_features = round(hidden_dim//2), out_features = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)       
        return x




    
class CNNRegressor(nn.Module):
    def __init__(self, output_dim: int):
        super(CNNRegressor, self).__init__()
        assert output_dim > 0, "Output dimension must be a positive integer"
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 16,
            kernel_size = (5, 5), 
            stride = (1, 1),
            padding = (0, 0)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (0,0)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 64, 
            kernel_size = (3, 3), 
            stride = (2, 2), 
            padding = (0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = (5,5),
            stride = (2,2),
            padding = (0,0)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            in_features=6400,
            out_features=output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        
        x = self.maxpool1(x)
        
        x = self.relu(self.conv2(x))
        
        x = self.maxpool2(x)

        # reshape for linear layer
        # note that the output of maxpool 2 is (*,64,1,1) so we just need to take the first column and row. 
        # If the output size is not 1,1, we have to flatten x before going into linear using torch.flatten
        x = self.flatten(x)

        x = self.linear1(x)     
        
        return x