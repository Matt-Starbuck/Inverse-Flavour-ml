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
    yu_mpl, yd_mpl = mpl_state[...,1], mpl_state[...,2]

    yu_vals = torch.linalg.svdvals(yu_mpl)
    yd_vals = torch.linalg.svdvals(yd_mpl)

    yu3_yu2 = yu_vals[...,0] / yu_vals[...,1]
    yu2_yu1 = yu_vals[...,1] / yu_vals[...,2]
    
    yd3_yd2 = yd_vals[...,0] / yd_vals[...,1]
    yd2_yd1 = yd_vals[...,1] / yd_vals[...,2]

    return yu3_yu2, yu2_yu1, yd3_yd2, yd2_yd1



