import math
import torch
from torchdiffeq import odeint
from src.inverse_flavour.inputs import init_state_sm
from src.physics.sm_rge import SMRGE, sm_at_Lambda

# This test script checks that the evolution of the SM up to the BSM scales is working correctly.
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running SM Initial State Factory test on device: {device}\n")

    # Initialise SM state at mu = 3 TeV
    Lambda_min = 3000.0 # Must be equal to 3TeV
    Lambda_max = 100000.0 # Can be anything creater than 3TeV
    sm_init_state = init_state_sm(device=device)
    print(f"[Check 1] sm_init_state shape: {sm_init_state.shape}, dtype: {sm_init_state.dtype}")
    assert sm_init_state.shape == (21,), "Expected 1D state vector of length 21"

    # Evolve dense SM trajectory from 3 TeV to 100 TeV
    t_eval_SM = torch.linspace(
        math.log(Lambda_min), math.log(Lambda_max), steps=1000, dtype=torch.float32, device=device
    )
    sm_model = SMRGE().to(device)

    with torch.no_grad():
        sm_dense_traj = odeint(sm_model, sm_init_state, t_eval_SM, method='rk4')

    print(f"[Check 2] sm_dense_traj shape: {sm_dense_traj.shape}, dtype: {sm_dense_traj.dtype}")
    assert sm_dense_traj.shape == (1000, 21), "Expected trajectory shape (1000, 21)"
    assert not torch.isnan(sm_dense_traj).any(), "NaNs detected in SM trajectory"

    # Physics sanity check on gauge & top Yukawa running (3 TeV -> 100 TeV)
    g_3tev = sm_dense_traj[0, :3].real
    g_100tev = sm_dense_traj[-1, :3].real
    yt_3tev = sm_dense_traj[0, 3:12].reshape(3, 3)[2, 2].abs().item()
    yt_100tev = sm_dense_traj[-1, 3:12].reshape(3, 3)[2, 2].abs().item()

    print("\n[Check 3] Physics Running Summary (3 TeV -> 100 TeV):")
    print(f"  g1: {g_3tev[0]:.4f} -> {g_100tev[0]:.4f} (expected to increase, b1 > 0)")
    print(f"  g2: {g_3tev[1]:.4f} -> {g_100tev[1]:.4f} (expected to decrease, b2 < 0)")
    print(f"  g3: {g_3tev[2]:.4f} -> {g_100tev[2]:.4f} (expected to decrease, b3 < 0)")
    print(f"  yt: {yt_3tev:.4f} -> {yt_100tev:.4f} (expected to decrease via QCD)")

    assert g_100tev[0] > g_3tev[0], "Physics Error: g1 should increase with energy!"
    assert g_100tev[1] < g_3tev[1], "Physics Error: g2 should decrease with energy!"
    assert g_100tev[2] < g_3tev[2], "Physics Error: g3 should decrease with energy!"
    assert yt_100tev < yt_3tev, "Physics Error: top Yukawa should decrease with energy!"

    # Test sm_at_Lambda interpolation for a batch of scales
    batch_size = 100
    log_lambda_batch = torch.linspace(
        math.log(Lambda_min), math.log(Lambda_max), steps=batch_size, dtype=torch.float32, device=device
    )
    bsm_init_states = sm_at_Lambda(log_lambda_batch, sm_dense_traj, t_eval_SM)
    print(f"\n[Check 4] bsm_init_states shape: {bsm_init_states.shape}, dtype: {bsm_init_states.dtype}")
    assert bsm_init_states.shape == (batch_size, 21)

    # Verify endpoints match the dense trajectory exactly
    err_start = (bsm_init_states[0] - sm_dense_traj[0]).abs().max().item()
    err_end = (bsm_init_states[-1] - sm_dense_traj[-1]).abs().max().item()
    print(f"  Endpoint interpolation error: start={err_start:.2e}, end={err_end:.2e}")
    assert err_start < 1e-6 and err_end < 1e-6, "Endpoint interpolation mismatch!"

    print("\nAll SM Initial State Factory checks passed successfully!")

if __name__ == "__main__":
    main()