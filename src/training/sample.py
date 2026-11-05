import math
import torch

# This script contains functions to generate a Lambda_batch, corresponding to different BSM scales. 

def Lambda_sample(sample_size: int,
                  lambda_min: float =3000.0,
                  lambda_max: float =100000.0, 
                  mode: str ='log_grid',
                  device: str ='cpu'
                  ):
    " Creates a sample of Lambda values in the range[Lambda_min, Lambda_max]"
    log_min = math.log(lambda_min)
    log_max = math.log(lambda_max)
    if mode == 'log_grid':
        return torch.linspace(log_min, log_max, steps=sample_size, device=device)
    elif mode == 'log_uniform':
        seed = torch.rand(sample_size, device=device)
        return log_min + (log_max - log_min) * seed

