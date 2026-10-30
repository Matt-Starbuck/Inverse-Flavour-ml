## Script to test the workflow from generating a batch of Lambda scales to computing
## the combined loss functions.

import math
import torch
from torchdiffeq import odeint
from src.inverse_flavour.inputs import init_state_sm, CKM_device
from src.physics.sm_rge import SMRGE, sm_at_Lambda
from src.physics.bsm_rge import evolve_bsm_batch
from src.loss_functions.eft_bounds import compute_eft_violation_ratios
from src.loss_functions.hierarchy_loss import hierarchy_loss
from src.loss_functions.combined_loss import combined_loss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running BSM Evolution & Loss Factory test on device: {device}\n")

    # Generate a small test batch of SM initial states at various Lambda scales
    batch_size = 8
    rg_steps = 1000
    # Choose evalutation time for the dense sm state trajectories.
    t_eval_SM = (torch.linspace(math.log(3000.0), math.log(1e5),
                                 steps=rg_steps, dtype=torch.float32, device=device)
    )
    # Turn off gradient tracking since this test does not require it
    with torch.no_grad():
        sm_dense_traj = odeint(SMRGE().to(device), init_state_sm(device), t_eval_SM, method='rk4')
    assert sm_dense_traj.shape == (rg_steps, 21)
    log_lambda_batch = torch.linspace(math.log(3000.0), math.log(1e5), steps=batch_size, dtype=torch.float32, device=device)
    assert log_lambda_batch.shape == (batch_size,)
    bsm_init_states = sm_at_Lambda(log_lambda_batch, sm_dense_traj, t_eval_SM)

    # Create mock learnable BSM couplings and theta
    torch.manual_seed(42)
    lambda_q = (0.1 * torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)).requires_grad_(True)
    lambda_u = (0.1 * torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)).requires_grad_(True)
    lambda_d = (0.1 * torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)).requires_grad_(True)
    theta = 0.5

    # Compute the pheno loss at the low scale
    pheno_dict = compute_eft_violation_ratios(lambda_q, lambda_u, lambda_d, log_lambda_batch
    )
    # Santiy check
    assert pheno_dict["K0"][0] > 1
    assert pheno_dict["D0"][0] > 1
    assert pheno_dict["Bd"][0] > 1
    assert pheno_dict["Bs"][0] > 1
    print("[Check 1]: Neutral meson bound saturated at 3 TeV")

    # Check the BSM batch evolution to M_Pl is working correctly
    g_mpl, yu_mpl, yd_mpl = evolve_bsm_batch(bsm_init_states, log_lambda_batch, lambda_q, lambda_u, lambda_d)
    print(f"[Check 2:] Evolved shapes at M_Pl -> g: {g_mpl.shape}, yu: {yu_mpl.shape}, yd: {yd_mpl.shape}")
    assert yu_mpl.shape == (batch_size, 3, 3) and yd_mpl.shape == (batch_size, 3, 3)
    print
    # Compute the hierarchy loss dictionary at Mpl
    hierarchy_dict = hierarchy_loss(bsm_init_states, log_lambda_batch,
                                    lambda_q, lambda_u, lambda_d
             )
    assert hierarchy_dict["up_32"][0] >0
    assert hierarchy_dict["up_21"][0] >0
    assert hierarchy_dict["down_32"][0] >0
    assert hierarchy_dict["down_21"][0] > 0
    print("[Check 3]: Expectations of hierarhcy orderings confirmed")
    
    # Test of combined loss functions
    theta= 0.5
    total_loss = combined_loss(bsm_init_states, log_lambda_batch,lambda_q,
              lambda_u, lambda_d, theta=0.5 )
    # Checking that the functions has combined correctly
    sum_pheno_loss=  (pheno_dict["D0"] + pheno_dict["K0"] + pheno_dict["Bd"] + pheno_dict["Bs"]
                + pheno_dict["uu"] + pheno_dict["dd"] + pheno_dict["qq"]
    )
    
    sum_hierarchy_loss = hierarchy_dict["up_32"] + hierarchy_dict["up_21"] + hierarchy_dict["down_32"] + hierarchy_dict["down_21"]
    assert total_loss[0] == theta * sum_hierarchy_loss[0] + (1-theta) * sum_pheno_loss[0]
    print("[Check 4]: Combined loss asembled correctly")
    # 5. Backpropagate gradients to lambda_q, lambda_u, lambda_d
    mean_loss_batch = total_loss.mean()
    mean_loss_batch.backward()
    for name, param in [("lambda_q", lambda_q), ("lambda_u", lambda_u), ("lambda_d", lambda_d)]:
        assert param.grad is not None, f"Gradient for {name} is None!"
        assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}!"
        print(f"[Check 3] Gradient norm for {name}: {param.grad.norm().item():.4e}")

    print("\nAll BSM Evolution & Autograd checks passed successfully!")
    print("\nAll checks passed, worklow is fully functioning!")

if __name__ == "__main__":
    main()