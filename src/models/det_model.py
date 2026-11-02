import torch
from torch import nn
import math

# This script defines the deterministic model for the BSM couplings.
# The input is the scale of NP (1dim) and the output is the 54 elements of the 
# 3x3 complex lambda matrices. 


class DetModel(nn.Module):
    def __init__(self, input_dim: int =1, hidden_dim: int = 108, output_dim: int=54):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    def forward(self, x):
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
######### Basic tests: to be incorporated into a proper test file later.

model_0 = DetModel(input_dim=1, hidden_dim=108, output_dim=54)
print(model_0.state_dict())

x_test = torch.Tensor([[0.1]])
output_test = model_0(torch.Tensor(x_test))
better_format = torch.reshape(output_test,( 3, 2, 3, 3))
lam_q_test, lam_u_test, lam_d_test = better_format[0,...], better_format[1,...], better_format[2,...]


print("This is the current output of the model:", output_test)
print("This is the shape of the outout:", output_test.shape)
print("This is the reshaped output", better_format)
print("This is the shape", better_format.shape)
lam_q_test = lam_q_test[0,...] + 1j * lam_q_test[1,...]
print("This is the test lambda_q matrix:", lam_q_test)

