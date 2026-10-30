import torch
import math
from src.loss_functions.eft_bounds import *
from src.loss_functions.hierarchy_loss import *

def combined_loss(bsm_init_states,
              log_lambda_batch: torch.Tensor,
              lambda_q: torch.Tensor,
              lambda_u: torch.Tensor, 
              lambda_d: torch.Tensor,  
              n_steps: int =50,
              theta: float = 0.5):
    
    # compute dictionary of yukawa singular values (yu3/yu2, yu2/yu1, yd3/yd2, yd2/yd1)
    h_infos = hierarchy_loss(bsm_init_states,log_lambda_batch,lambda_q,
                                        lambda_u, lambda_d, n_steps)
    
    # compute dictionary of pheno constraints
    p_infos = compute_eft_violation_ratios(lambda_q,lambda_u, lambda_d, 
                                               log_lambda_batch)
    
    # Sum dictionary values to form the hierarhcy and phenomenological loss functions
    h_loss = h_infos["up_32"] + h_infos["up_21"] + h_infos["down_32"] + h_infos["down_21"]

    p_loss =  (p_infos["D0"] + p_infos["K0"] + p_infos["Bd"] + p_infos["Bs"]
                + p_infos["uu"] + p_infos["dd"] + p_infos["qq"]
    )

    # Combine into total loss function with theta weighting

    tot_loss = theta * h_loss + (1- theta) * p_loss

    return tot_loss