import torch
from torch import nn
from Constants import * 


class Brain(nn.Module):
    
    def __init__(self, 
                 input_dim: int=NUM_SIGHT_LINES*2-1,
                 hidden_dim: int=2,
                 output_dim: int=2,
                 parent_weights = None
                 ) -> None:

        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define the layers
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)  # Input layer
        # self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)  # Output layer
        self.activation = nn.ReLU()  # Activation function

        if parent_weights:
            self.load_state_dict(parent_weights)
            self.mutate()

    
    # input: left, right, center, vision, currently eating (binary)
    #        %energy remaining
    # output: direction to move (theta), speed  
    def forward(self, 
                inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)  
        output = self.fc1(inputs)
        #output = self.activation(hidden)
        # output = self.fc2(activated_hidden)

        # theta needs to be bounded between -180 and 180
        # output[0] is passed through tanh to get values between -1 and 1, it is then scaled by turn rate inside creature
        theta = torch.tanh(output[0])

        # speed needs to be bounded between 0 and 1
        # output[1] is passed through sigmoid to get values between 0 and 1
        speed = torch.sigmoid(output[1])

        return theta, speed 
    
    def mutate(self, mutation_rate=.05, mutation_scale=0.01):
        with torch.no_grad():  # We don't want these operations to be tracked by autograd
            for param in self.parameters():
                mutation_tensor = torch.randn_like(param)  # Tensor of random numbers with the same shape as param
                mutation_mask = torch.rand_like(param) < mutation_rate  # Tensor of booleans indicating which weights to mutate
                param.add_(mutation_mask * mutation_tensor * mutation_scale)  # Add scaled random changes to the selected weights