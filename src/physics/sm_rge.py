import torch
from torch import nn
from src.inverse_flavour.inputs import b1, b2, b3, loop_factor
from torchdiffeq import odeint_adjoint as odeint

# This scipts contains classses to both evolve the SM gauge and Yukawa couplings,
# and to extract the state of these couplings at the relevant scale Lambda.
# This can be thought of as the IR evolution part of a BSM scenerio,
# and provides the intial conditions for the BSM evolution at a particular Lambda scale.

# This first class maps the vector state in the SM to the beta functions for the SM couplings
class SMRGE(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, t, state):

    # Set device to state device
    device = state.device
    type  = state.dtype

    # Unpack the 1D state vector into couplings and Yukawa matrices.
    g1 = state[0].real
    g2 = state[1].real
    g3 = state[2].real
    yu = state[3:12].reshape(3,3)
    yd = state[12:21].reshape(3,3)

    # Compute gauge coupling beta functions
    dg1 = b1 * g1**3
    dg2 = b2 * g2**3
    dg3 = b3 * g3**3

    dij = torch.eye(3, dtype=type, device=device)

    # Compute the Yukawa beta function tensor structures
    Trace = torch.trace(3 * yu @ yu.mH + 3 * yd @ yd.mH) * dij
    Gu = (17/12 * g1**2 + 9/4 * g2**2 + 8 * g3**2) * dij
    Gd = (5/12 * g1**2 + 9/4 * g2**2 + 8 * g3**2) * dij
    Delta_ud = 3/2 * (yu @ yu.mH - yd @ yd.mH)

    # Compute the Yukawa coupling beta functions
    dyu = yu @ ((Trace - Gu) + Delta_ud)
    dyd = yd @ ((Trace - Gd) - Delta_ud)

    # Concatenate into a single 1D state derivative vector
    dg = torch.stack([dg1, dg2, dg3]).to(type)
    dstate = torch.cat([dg, dyu.flatten(), dyd.flatten()])

    return dstate * loop_factor
  
def sm_at_Lambda(Lambda_batch, SM_dense_traj, t_eval_SM):
  "Takes in a dense trajctory of the SM states upto scale t_eval_SM, along with a batch of Lambda scales, and returns the SM states as close to the Lambda scales as possible "
  idx = torch.bucketize(Lambda_batch, t_eval_SM) -1
  idx = torch.clamp(idx, 0, len(t_eval_SM)-2)

  t0 = t_eval_SM[idx]
  t1 = t_eval_SM[idx+1]
  weight = (Lambda_batch - t0) / (t1 - t0)

  state0 = SM_dense_traj[idx]
  state1 = SM_dense_traj[idx+1]

  return state0 + weight.unsqueeze(-1) * (state1 - state0)



