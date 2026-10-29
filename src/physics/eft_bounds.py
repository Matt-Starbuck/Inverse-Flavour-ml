import math 
import torch 
from src.inverse_flavour.inputs import CKM_device

# This script contains LEFT Wilson coefficient constraints, and functions that 
# compute whether a set of lambdas satisfy these constraints. 
# Bounds are taken from https://arxiv.org/abs/2402.09503.

# For simplicity, LR structures are neglected and left for a more precise implementation.

# Delta F=2
C_bs_1_max = 2.2676e-11
C_bd_1_max = 8.2645e-13
Im_C_sd_1_max = 1.4793e-15
Im_C_cu_1_max = 1.0000e-14

# LHC diject 
C_uu_run2_max = 4.4e-8  # GeV^-2
C_dd_run2_max = 1.1e-7  # GeV^-2
C_qq_run2_max = 3.7e-8  # GeV^-2

# The flavour basis for qL used throughout is the up-mass basis, and so the Wilson coefficients for down type coefficients will need to be rotated by the CKM.

def compute_eft_violation_ratios(
    lambda_q: torch.Tensor,
    lambda_u: torch.Tensor,
    lambda_d: torch.Tensor,
    log_lambda_batch: torch.Tensor,
    V_CKM: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes the dimensionless ratio |C_pred| / C_max for each EFT constraint across the batch.
    A ratio <= 1.0 means the constraint is satisfied; > 1.0 means it is violated.
    Fully differentiable for use in the training loss function.
    """
    # Scale factor 1 / Lambda^2 of shape [batch_size]
    inv_lambda_sq = torch.exp(-2.0 * log_lambda_batch)

    # Ensure V_CKM is on the same device/dtype as lambda_q
    V_CKM = CKM_device(device=lambda_q.device).to(lambda_q.dtype)

    lambda_q_d = V_CKM.mH @ lambda_q

    # Construct the flavour combinations M = lambda^\dagger @ lambda of shape [batch_size, 3, 3]
    M_u = lambda_u @ lambda_u.mH
    M_d = lambda_d @ lambda_d.mH
    M_q_u = lambda_q @ lambda_q.mH
    M_q_d = lambda_q_d @ lambda_q_d.mH

    # Delta F = 2 Wilson coefficients: C_{ij} = (M_{ij})^2 / Lambda^2 (complex!)
    C_cu_pred = ((M_u[:, 0, 1] ** 2) + (M_q_u[:, 0, 1] ** 2)) * inv_lambda_sq
    C_sd_pred = ((M_d[:, 0, 1] ** 2) + (M_q_d[:, 0, 1] ** 2)) * inv_lambda_sq
    C_bd_pred = ((M_d[:, 0, 2] ** 2) + (M_q_d[:, 0, 2] ** 2)) * inv_lambda_sq
    C_bs_pred = ((M_d[:, 1, 2] ** 2) + (M_q_d[:, 1, 2] ** 2)) * inv_lambda_sq

    # LHC dijet ΔF = 0 four-fermion operators: C_{ii} = (M_{ii})^2 / Lambda^2
    C_uu_pred = (M_u[:, 0, 0].real ** 2) * inv_lambda_sq
    C_dd_pred = (M_d[:, 0, 0].real ** 2) * inv_lambda_sq
    C_qq_pred = (M_q_u[:, 0, 0].real ** 2) * inv_lambda_sq

    # Add a tiny epsilon (1e-30) inside abs() on complex numbers to prevent NaN gradients at 0
    ratios = {
        "D0": torch.abs(torch.imag(C_cu_pred)) / Im_C_cu_1_max,
        "K0": torch.abs(torch.imag(C_sd_pred)) / Im_C_sd_1_max,
        "Bd": torch.abs(C_bd_pred + 1e-30) / C_bd_1_max,
        "Bs": torch.abs(C_bs_pred + 1e-30) / C_bs_1_max,
        "uu": torch.abs(C_uu_pred) / C_uu_run2_max,
        "dd": torch.abs(C_dd_pred) / C_dd_run2_max,
        "qq": torch.abs(C_qq_pred) / C_qq_run2_max,
    }
    return ratios