import math
import torch
from torch import nn
from src.inverse_flavour.inputs import b1, b2, b3, loop_factor
from torchdiffeq import odeint_adjoint as odeint


class LearnedBSMRGE(nn.Module):
  def __init__(self, b1, b2, b3, log_lambda_batch, device='cpu'):
    super().__init__()

    # Initialize the learnable parameters directly as complex tensors.
    self.lambda_q = nn.Parameter(torch.randn(3,3, dtype=torch.complex64, device=device) * 1e-4)
    self.lambda_u = nn.Parameter(torch.randn(3,3, dtype=torch.complex64, device=device) * 1e-4)
    self.lambda_d = nn.Parameter(torch.randn(3,3, dtype=torch.complex64, device=device) * 1e-4)

    # Initialise the MFV parameters as before.
    self.t_end = math.log(1e19)
    self.t_lambda = log_lambda_batch
    self.delta_t = (self.t_end - self.t_lambda).view(-1,1).to(dtype=torch.complex64, device=device)

  def forward(self, t, state):
        # Extract the batch size.
        batch_size = state.shape[0]

        # Unpack the flattened state [batch_size, 21].
        g1 = state[:, 0].to(torch.complex64)
        g2 = state[:, 1].to(torch.complex64)
        g3 = state[:, 2].to(torch.complex64)
        yu = state[:, 3:12].reshape(batch_size, 3, 3)
        yd = state[:, 12:21].reshape(batch_size, 3, 3)

        dij = torch.eye(3, dtype=torch.complex64, device='cpu')

        # Construct the tensor structures in the Yukawa beta functions
        spur_yu = yu @ yu.mH
        spur_yd = yd @ yd.mH
        tr = 3 * torch.einsum('bii->b', spur_yu + spur_yd)
        Trace_val = tr.view(batch_size, 1, 1) * dij
        Gu_val = ((17/12 * g1**2 + 9/4 * g2**2 + 8 * g3**2)
                  .view(batch_size, 1, 1) * dij)
        Gd_val = ((5/12 * g1**2 + 9/4 * g2**2 + 8 * g3**2)
                  .view(batch_size, 1, 1) * dij)

        # Extract the BSM coefficients for the up sector
        C1u, C2u, C3u = (self.C[:, 0].view(batch_size, 1, 1),
                         self.C[:, 1].view(batch_size, 1, 1),
                         self.C[:, 2].view(batch_size, 1, 1))
        C1d, C2d, C3d = (self.C[:, 3].view(batch_size, 1, 1),
                         self.C[:, 4].view(batch_size, 1, 1),
                         self.C[:, 5].view(batch_size, 1, 1))

        # Compute the MFV beta functions for up sector
        dyu_bracket = ((Trace_val - Gu_val + C1u * dij)
                       + (C2u) * spur_yu
                       + (C3u) * spur_yd)
        dyd_bracket = ((Trace_val - Gd_val + C1d * dij)
                       + (C2d) * spur_yu
                       + (C3d) * spur_yd)
        dyu_MFV = yu @ dyu_bracket
        dyd_MFV = yd @ dyd_bracket

        # Multiply the full beta function with the loop factor
        dyu = loop_factor * (dyu_MFV + self.delta_u)
        dyd = loop_factor * (dyd_MFV + self.delta_d)

        # Write the gauge beta functions and concatenate into a tensor
        dg1 = loop_factor * (self.b1 * g1**3)
        dg2 = loop_factor * (self.b2 * g2**3)
        dg3 = loop_factor * (self.b3 * g3**3)
        dg = torch.stack([dg1, dg2, dg3], dim=1)

        # Flatten and concatenate to match the original state shape [batch_size, 21]
        dyu_flat = dyu.reshape(batch_size, 9)
        dyd_flat = dyd.reshape(batch_size, 9)

        # Concatenate all beta functions together into a vector
        dstate = torch.cat([dg, dyu_flat, dyd_flat], dim=1)

        return dstate * self.delta_t