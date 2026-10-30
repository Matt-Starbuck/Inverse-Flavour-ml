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
    _, yu_mpl, yd_mpl = mpl_state

    yu_vals = torch.linalg.svdvals(yu_mpl)
    yd_vals = torch.linalg.svdvals(yd_mpl)

    # Want to aim this towards zero, and therefore take the log of it.
    yu3_yu2 = torch.log(yu_vals[...,0] / yu_vals[...,1]) **2
    yu2_yu1 = torch.log(yu_vals[...,1] / yu_vals[...,2]) **2
    
    yd3_yd2 = torch.log(yd_vals[...,0] / yd_vals[...,1]) **2
    yd2_yd1 = torch.log(yd_vals[...,1] / yd_vals[...,2]) **2

    losses = {
            "up_32": yu3_yu2,
            "up_21": yu2_yu1,
            "down_32": yd3_yd2,
            "down_21": yd2_yd1
    }

    return losses



