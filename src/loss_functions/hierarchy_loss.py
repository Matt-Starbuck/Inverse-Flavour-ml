import torch
import math
from src.physics.bsm_rge import BSMRGE, evolve_bsm_batch

# This script takes in a batch of BSM initial states, RG evolves them to Mpl and returns
# ratios of singular values.

def hierarchy_loss(bsm_init_states,
              log_lambda_batch: torch.Tensor,
              lambda_q: torch.Tensor,
              lambda_u: torch.Tensor, 
              lambda_d: torch.Tensor,  
              n_steps: int =50):
    
    # Evolve the state up to planck scale
    mpl_state = evolve_bsm_batch(bsm_init_states,log_lambda_batch,
                                  lambda_q, lambda_u, lambda_d,  n_steps)
    
    # extract the up and down yukawas and reshape them into matrices
    yu_mpl, yd_ml = mpl_state[..., 3:12], mpl_state[...,12:21]

    yu_mpl = torch.reshape(yu_mpl, (...,3,3))
    yd_mpl = torch.reshape(yd_mpl, (...,3,3))

    yu_vals = torch.linalg.svdvals(yu_mpl, full_matrices=True)
    yd_vals = torch.linalg.svdvals(yd_mpl, full_matrices=True)

    yu3_yu2 = yu_vals[...,2] / yu_vals[...,1]
    yu2_yu1 = yu_vals[...,1] / yu_vals[...,0]
    
    yd3_yd2 = yd_vals[...,2] / yd_vals[...,1]
    yd2_yd1 = yd_vals[...,1] / yd_vals[...,0]

    return yu3_yu2, yu2_yu1, yd3_yd2, yd2_yd1



