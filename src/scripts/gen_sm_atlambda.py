from src.inverse_flavour.inputs import init_state_sm
from src.physics.sm_rge import SMRGE
from src.physics.sm_rge import sm_at_Lambda
import torchdiffeq
from torchdiffeq import odeint
import torch
import math

if __name__ == "__main__":
    
    # Set the deivce to GPU if available, or to cpu otherwise.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Initialise the SM state at the scale mu=3TeV and put it on the device.
    sm_init_state = init_state_sm(device=device)

    # Create a dense trajectory of the SM evolution from mu=3TeV to a high scale.
    Lambda_max = 100000.0
    t_start = torch.tensor(math.log(3000.0), dtype=torch.float64, device=device)
    t_max_lambda = torch.tensor(math.log(1e5), dtype=torch.float64, device=device)
    t_eval_SM = torch.linspace(t_start, t_max_lambda, 1000, dtype=torch.float32, device=device)
    sm_model = SMRGE().to(device)
    sm_dense_traj = odeint(sm_model, sm_init_state, t_eval_SM, method='rk4')
    
    # Initialise a set of BSM scales (Lambda) for the training data.
    batch_size= 100
    Lambda_min = 3000.0
    sample_mode = 'log_grid'
    log_lambda_batch = torch.linspace(math.log(Lambda_min), math.log(Lambda_max), steps=batch_size, device=device)

    # For each Lambda, have the initial SM state at this scale to prepare for the BSM evolution.
    bsm_init_states = sm_at_Lambda(log_lambda_batch, sm_dense_traj.squeeze(1), t_eval_SM)

