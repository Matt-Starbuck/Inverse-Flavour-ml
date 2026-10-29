from pathlib import Path
import math
import torch
from torchdiffeq import odeint
from src.inverse_flavour.inputs import init_state_sm
from src.physics.sm_rge import SMRGE, sm_at_Lambda

class BSMInitialState:
    """ Computes the dense SM RGE trajectory once and interpolates initial states for any batched lambda set.
        Contains method to split this interpolated batch into a training and testing set.
    """
    def __init__(self, lambda_min: float = 3000.0, lambda_max: float = 1e5, steps: int = 1000, device: str = 'cpu', cache_path: str | None = None):
        self.device = device
        self.t_eval_SM = torch.linspace(math.log(lambda_min), math.log(lambda_max), steps=steps, dtype=torch.float32, device=device)

        if cache_path and Path(cache_path).exists():
            self.sm_dense_traj = torch.load(cache_path, map_location=device)
        else:
            sm_init = init_state_sm(device=device)
            sm_model = SMRGE().to(device)
            with torch.no_grad():
                self.sm_dense_traj = odeint(sm_model, sm_init, self.t_eval_SM, method='rk4')
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.sm_dense_traj, cache_path)

    def get_init_states_train_test(self, log_lambda_batch: torch.Tensor, split: float=0.8) -> torch.Tensor:
        """ Takes in lambda batch and returns the intial BSM states for each of the Lambdas
        """
        full_sm_at_Lambda = sm_at_Lambda(log_lambda_batch, self.sm_dense_traj, self.t_eval_SM)
        sm_at_Lambda_train = full_sm_at_Lambda[:split,...]
        sm_at_Lambda_test = full_sm_at_Lambda[split:,...]
        return sm_at_Lambda_train, sm_at_Lambda_test
    