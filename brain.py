import torch
from torch import nn
from typing import List


class Brain(nn.Module):
    
    def __init__(self, 
                 input_dim: int=7,
                 hidden_dim: int=10,
                 output_dim: int=2
                 ) -> None:
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)  # Input layer
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)  # Output layer
        self.activation = nn.Sigmoid()  # Activation function
    
    # input: left, right, center, vision, currently eating (binary)
    #        %energy remaining
    # output: direction to move (theta), speed  
    def forward(self, 
                vision: List[float],
                eating: float,
                energy_remaining: float):
        inputs = torch.tensor([*vision, eating, energy_remaining], dtype=torch.float32)  
        hidden = self.fc1(inputs)
        activated_hidden = self.activation(hidden)
        output = self.fc2(activated_hidden)

        # theta needs to be bounded between -180 and 180
        # output[0] is passed through tanh to get values between -1 and 1, then scaled to be between -180 and 180
        theta = torch.tanh(output[0]) * 180 

        # speed needs to be bounded between 0 and 1
        # output[1] is passed through sigmoid to get values between 0 and 1
        speed = torch.sigmoid(output[1])

        return theta, speed 