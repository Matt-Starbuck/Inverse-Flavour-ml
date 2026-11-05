import torch
from torch import nn
import math

# This script defines the deterministic model for the BSM couplings.
# The input is the scale of NP (1dim) and the output is the 54 elements of the 
# 3x3 complex lambda matrices. 


class DetModel(nn.Module):
    def __init__(self, input_dim: int =1,
                hidden_dim: int = 108,
                output_dim: int=54,
                lambda_min: float = 3000.0,
                lambda_max: float = 100000.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.Lambda_min = math.log(lambda_min)
        self.Lambda_max = math.log(lambda_max)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    def forward(self, x):
        # Renormalises the input to [-1,+1] over the whole range of Lambda to be
        # considered.
        x = 2*(x- self.Lambda_min) / (self.Lambda_max-self.Lambda_min) - 1
        x = torch.unsqueeze(x, dim=1)  # Ensure input is of shape (batch_size, 1)
        return self.model(x)
    

# Define a function to reshape output of neural network into three 3x3 complex matrices
# for the loss function input

def reshape_output(output: torch.Tensor, batch_size: int):
    # Reshape to (batch_size, 3,2,3,3)
    reshape_output = torch.reshape(output, (batch_size, 3, 2, 3, 3))
    # Split into three matrices
    lam_q_bare, lam_u_bare, lam_d_bare = reshape_output[:, 0, ...], reshape_output[:, 1, ...], reshape_output[:, 2, ...]    
    # convert each into a complex matrix
    lam_q = lam_q_bare[..., 0, :, :] + 1j * lam_q_bare[..., 1, :, :]
    lam_u = lam_u_bare[...,0,:,:] + 1j * lam_u_bare[..., 1, :, :]
    lam_d = lam_d_bare[...,0,:,:] + 1j * lam_d_bare[..., 1, :, :]
    return lam_q, lam_u, lam_d