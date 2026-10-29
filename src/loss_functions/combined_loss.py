import torch
import math
from src.loss_functions.eft_bounds import *
from src.loss_functions.hierarchy_loss import *

def combined_loss(bsm_init_states,
              log_lambda_batch: torch.Tensor,
              lambda_q: torch.Tensor,
              lambda_u: torch.Tensor, 
              lambda_d: torch.Tensor,  
              n_steps: int =50):
    