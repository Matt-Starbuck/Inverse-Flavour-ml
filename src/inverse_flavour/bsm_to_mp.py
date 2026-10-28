import torch
from torchdiffeq import odeint
from src.physics.bsm_rge import BSMRGE

def bsm_to_mp(bsm_init_states,
              b1: float,
              b2: float,
              b3: float,
              log_lambda_batch: torch.Tensor,
              lambda_q: torch.Tensor,
              lambda_u: torch.Tensor, 
              lambda_d: torch.Tensor,  
              n_steps: int =50) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    "Evolves a batch of BSM initial states up to Mpl for a given lambda configuration"
    
    # Extract the deivce and batch size from the BSM initial states.
    device = bsm_init_states.device()
    batch_size = bsm_init_states.shape()[0]

    # Create an instance of the BSMRGE class and the registered time span s
    rge_func = BSMRGE(b1, b2, b3, log_lambda_batch, lambda_q, lambda_u, lambda_d, device=device).to(device)
    s_span = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)

    # Integrate over normalized time s in [0, 1] using fixed-step RK4
    traj = odeint(rge_func, bsm_init_states, s_span, method='rk4', options={'step_size': 1.0 / n_steps})
    final_state = traj[-1]  # shape [batch_size, 21]

    # return the gauge coupling and Yukawas at Mpl
    g_mpl = final_state[:, :3].real
    yu_mpl = final_state[:, 3:12].reshape(batch_size, 3, 3)
    yd_mpl = final_state[:, 12:21].reshape(batch_size, 3, 3)

    return g_mpl, yu_mpl, yd_mpl


    
    