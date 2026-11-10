import torch
import math
from src.loss_functions.eft_bounds import *
from src.loss_functions.hierarchy_loss import *
import torch.nn.functional as F

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
    h_loss = sum(r for r in h_infos.values())

    # Uses softplus function to not over penalise Wilson coefficients far within the bounds.
    p_loss = sum(F.softplus(r-0.5, beta=10.0) for r in p_infos.values()) 

    # punish the loss for choosing large coupling values
    pert_loss = (torch.exp(torch.max(torch.abs(lambda_q)) / (2 * torch.pi)) 
                 + torch.exp(torch.max(torch.abs(lambda_u)) / (2 * torch.pi)) 
                 + torch.exp(torch.max(torch.abs(lambda_d)) / (2 * torch.pi)) -3)
                          

    # Combine into total loss function with theta weighting

    tot_loss = theta * h_loss + (1- theta) * p_loss + pert_loss

    mean_tot_loss = torch.mean(tot_loss)
    mean_h_loss = torch.mean(h_loss)
    mean_p_loss = torch.mean(p_loss)

    return mean_tot_loss, mean_h_loss, mean_p_loss