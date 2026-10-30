import math
import torch
from torchdiffeq import odeint
from src.inverse_flavour.inputs import init_state_sm, CKM_device
from src.physics.sm_rge import SMRGE, sm_at_Lambda
from src.physics.bsm_rge import evolve_bsm_batch
from src.loss_functions.eft_bounds import compute_eft_violation_ratios

def am_gm_hierarchy_loss(Y: torch.Tensor) -> torch.Tensor:
    """Basis-invariant hierarchy metric: 0 iff all singular values are equal."""
    H = Y @ Y.mH
    tr_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1).real
    det_H = torch.linalg.det(H).real
    return (3.0 * torch.log(tr_H / 3.0) - torch.log(det_H.clamp(min=1e-30))).mean()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running BSM Evolution & Loss Factory test on device: {device}\n")

    # 1. Generate a mini-batch of SM initial states at various Lambda scales
    batch_size = 16
    t_eval_SM = torch.linspace(math.log(3000.0), math.log(1e5), steps=1000, dtype=torch.float32, device=device)
    with torch.no_grad():
        sm_dense_traj = odeint(SMRGE().to(device), init_state_sm(device), t_eval_SM, method='rk4')

    log_lambda_batch = torch.linspace(math.log(3000.0), math.log(1e5), steps=batch_size, dtype=torch.float32, device=device)
    bsm_init_states = sm_at_Lambda(log_lambda_batch, sm_dense_traj, t_eval_SM)

    # 2. Create mock learnable BSM couplings (simulating Neural Network outputs with requires_grad=True)
    torch.manual_seed(42)
    lambda_q = (0.01 * torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)).requires_grad_(True)
    lambda_u = (0.01 * torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)).requires_grad_(True)
    lambda_d = (0.01 * torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)).requires_grad_(True)

    # 3. Evolve batch to M_Pl
    g_mpl, yu_mpl, yd_mpl = evolve_bsm_batch(bsm_init_states, log_lambda_batch, lambda_q, lambda_u, lambda_d)
    print(f"[Check 1] Evolved shapes at M_Pl -> g: {g_mpl.shape}, yu: {yu_mpl.shape}, yd: {yd_mpl.shape}")
    assert yu_mpl.shape == (batch_size, 3, 3) and yd_mpl.shape == (batch_size, 3, 3)

    # 4. Compute Hierarchy Loss + EFT Bounds Loss and test Backward Autograd
    loss_hier = am_gm_hierarchy_loss(yu_mpl) + am_gm_hierarchy_loss(yd_mpl)
    eft_ratios = compute_eft_violation_ratios(lambda_q, lambda_u, lambda_d, log_lambda_batch, CKM_device(device))
    loss_eft = sum(torch.relu(r - 1.0).pow(2).mean() for r in eft_ratios.values())

    total_loss = loss_hier + loss_eft
    print(f"[Check 2] Forward Loss -> Hierarchy: {loss_hier.item():.4f}, EFT penalty: {loss_eft.item():.4f}")

    # 5. Backpropagate gradients to lambda_q, lambda_u, lambda_d
    total_loss.backward()
    for name, param in [("lambda_q", lambda_q), ("lambda_u", lambda_u), ("lambda_d", lambda_d)]:
        assert param.grad is not None, f"Gradient for {name} is None!"
        assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}!"
        print(f"[Check 3] Gradient norm for {name}: {param.grad.norm().item():.4e}")

    print("\nAll BSM Evolution & Autograd checks passed successfully!")

if __name__ == "__main__":
    main()