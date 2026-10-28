import math
import torch
from torch import nn
from src.inverse_flavour.inputs import b1, b2, b3, loop_factor


class BSMRGE(nn.Module):
  def __init__(self, b1:float, b2:float, b3:float, log_lambda_batch, lambda_q: torch.Tensor, lambda_u: torch.Tensor, lambda_d: torch.Tensor, device='cpu'):
    super().__init__()

    self.b1 = b1
    self.b2 = b2
    self.b3 = b3

    batch_size = log_lambda_batch.shape[0]

    # Initialize the learnable parameters directly as complex tensors.
    self.lambda_q = lambda_q # shape [batch_size, 3,3]
    self.lambda_u = lambda_u # shape [batch_size, 3,3]
    self.lambda_d = lambda_d # shape [batch_size, 3,3]

    # Initialise the MFV parameters as before.
    self.t_end = math.log(1e19)
    self.t_lambda = log_lambda_batch
    self.register_buffer('delta_t', (self.t_end - self.t_lambda).view(-1, 1).to(torch.complex64))

  def forward(self, t, state):
        "This puts collects the SM and BSM beta functions, and evolves the yukawas and gauge couplings from them"
        # Extract the batch size.
        batch_size = state.shape[0]
        device = state.device

        # Unpack the flattened state [batch_size, 21].
        g1 = state[:, 0].to(torch.complex64)
        g2 = state[:, 1].to(torch.complex64)
        g3 = state[:, 2].to(torch.complex64)
        yu = state[:, 3:12].reshape(batch_size, 3, 3)
        yd = state[:, 12:21].reshape(batch_size, 3, 3)

        dij = torch.eye(3, dtype=torch.complex64, device=device)

        # Compute the Yukawa beta function tensor structures within the SM
        mat_trace = 3 * yu @ yu.mH + 3 * yd @ yd.mH # Making a batch safe trace
        trace_val = torch.diagonal(mat_trace, dim1=-2, dim2=-1).sum(-1).view(batch_size, 1,1)
        Trace = trace_val * dij

        g1_sq = (g1**2).view(batch_size, 1, 1)
        g2_sq = (g2**2).view(batch_size, 1, 1)
        g3_sq = (g3**2).view(batch_size, 1, 1)

        Gu = (17/12 * g1_sq + 9/4 * g2_sq + 8 * g3_sq) * dij
        Gd = (5/12 * g1_sq + 9/4 * g2_sq + 8 * g3_sq) * dij
        Delta_ud = 3/2 * (yu @ yu.mH - yd @ yd.mH)

        # Compute the Yukawa coupling beta functions
        dyu_SM = ((Trace - Gu) + Delta_ud)@ yu
        dyd_SM = ((Trace - Gd) - Delta_ud)@ yd

        # Multiply the full beta function with the loop factor
        dyu = loop_factor * (dyu_SM + self.lambda_q @ self.lambda_u.mH)
        dyd = loop_factor * (dyd_SM + self.lambda_q @ self.lambda_d.mH)

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