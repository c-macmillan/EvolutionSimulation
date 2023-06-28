import torch
import random
from torch import nn
from Constants import * 


class Brain(nn.Module):
    
    def __init__(self,
                 masks,
                 input_dim: int=NUM_SIGHT_LINES*2-1,
                 hidden_dim: int=5,
                 output_dim: int=2,
                 parent_weights = None,
                 num_layers = 1
                 ) -> None:

        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.masks = masks
        self.num_layers = num_layers

        # initially just have one hidden layer 
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)  # Input layer
        self.hidden = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers))
        self.activation = nn.ReLU()  # Activation function
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        if parent_weights:
            self.load_state_dict(parent_weights)
            self.mutate()

    
    # input: left, right, center, vision, currently eating (binary)
    #        %energy remaining
    # output: direction to move (theta), speed  
    def forward(self, 
                inputs):
        
        inputs = torch.tensor(inputs, dtype=torch.float32)
        x = self.activation(self.input_layer(inputs))
        for layer, mask in zip(self.hidden, self.masks):
            weights, biases = list(layer.parameters())
            new_weights = torch.mul(weights, mask)
            new_biases = torch.mul(biases, mask)
            x = new_weights@x + new_biases
            x = self.activation(x)

        output = self.output_layer(x)
        

        # theta needs to be bounded between -180 and 180
        # output[0] is passed through tanh to get values between -1 and 1, it is then scaled by turn rate inside creature
        theta = torch.tanh(output[0])

        # speed needs to be bounded between 0 and 1
        # output[1] is passed through sigmoid to get values between 0 and 1
        speed = torch.sigmoid(output[1])

        return theta, speed 
    
    def mutate(self, mutation_rate=0.5, mutation_scale=0.01):
        with torch.no_grad():  
            if random.uniform(0, 1) < mutation_rate:
                change_mask = random.randint(0, len(self.masks)-1)
                change_row = random.randint(0, self.hidden_dim-1)
                change_elem = self.masks[change_mask][change_row]
                self.masks[change_mask][change_row] = 0 if change_elem else 1
                
            if random.uniform(0, 1) < mutation_rate:
                # randomly shift weights
                for param in self.parameters():
                    mutation_tensor = torch.randn_like(param)  # Tensor of random numbers with the same shape as param
                    mutation_mask = torch.rand_like(param) < mutation_rate  # Tensor of booleans indicating which weights to mutate
                    param.add_(mutation_mask * mutation_tensor * mutation_scale)  # Add scaled random changes to the selected weights