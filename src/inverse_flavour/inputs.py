import torch
import math

## Input quantities

# Quark mass inputs at mu=3TeV.
m_up = 0.993289e-3
m_down = 2.16132e-3
m_strange = 4.29966e-2
m_charm = 0.518318
m_bottom = 2.17964
m_top = 140.90

## Converting to low scale Yukawa matrices

# Computing yukawa eigenvalues.
v = 246
y_up = math.sqrt(2) * m_up / v
y_down = math.sqrt(2) * m_down / v
y_strange = math.sqrt(2) * m_strange / v
y_charm = math.sqrt(2) * m_charm / v
y_bottom = math.sqrt(2) * m_bottom / v
y_top = math.sqrt(2) * m_top / v

# Inputs for the gauge couplings at the scale mu=3TeV in the MSbar scheme (same as masses).
g1_3tev = 0.36
g2_3tev = 0.63
g3_3tev = 1

# Gauge beta function coefficients.
b1 = 41/6
b2 = -19/6
b3 = -7
loop_factor = 1.0 / (16 * torch.pi **2)

def Id_device(device='cpu'):
    "Constructs the identity matrix on the device."
    return torch.eye(3, dtype=torch.complex64, device=device)

def yukawas_to_device(device='cpu'):
    "Constructs quark yukawa matrices in the mass basis of up quarks, and puts them onto the device."
    # CKM Matirx in PDG phase convention.
    VCKM_mean = (torch.tensor([
        [0.9743355 +0.0j,              0.22506836+0.0j, 0.00156066-3.48431162e-03j],
        [-0.2249448 -1.38439423e-04j,  0.97351783-3.19790606e-05j,0.04077819+0.0j],
        [0.00765866-3.39208938e-03j, -0.04008318-7.83561716e-04j, 0.99916093+0.0j]
        ], dtype=torch.complex64)
    ).to(device)

    # Constructing the up Yukawa matrices.
    y_u_vals = torch.tensor([y_up, y_charm, y_top], dtype=torch.complex64).to(device)
    y_u_3tev = torch.diag(y_u_vals)

    # Constructing the down Yukawa matrices.
    y_d_vals = torch.tensor([y_down, y_strange, y_bottom], dtype= torch.complex64).to(device)
    y_d_3tev = VCKM_mean @ torch.diag(y_d_vals)

    return y_u_3tev.flatten(), y_d_3tev.flatten()

def CKM_device(device='cpu'):
    "returns the CKM matrix on the current device "
    VCKM_mean = (torch.tensor([
        [0.9743355 +0.0j,              0.22506836+0.0j, 0.00156066-3.48431162e-03j],
        [-0.2249448 -1.38439423e-04j,  0.97351783-3.19790606e-05j,0.04077819+0.0j],
        [0.00765866-3.39208938e-03j, -0.04008318-7.83561716e-04j, 0.99916093+0.0j]
        ], dtype=torch.complex64)
    ).to(device)
    return VCKM_mean

def init_state_sm(device='cpu'):
    "Initialises (g1,g2,g3,y_u,y_d) flattened at mu=3TeV. This is the intial input for the ODE solver."
    y_u_3tev, y_d_3tev = yukawas_to_device(device)
    g_3tev = torch.tensor([g1_3tev, g2_3tev, g3_3tev], dtype = torch.complex64, device=device)
    return torch.cat([g_3tev, y_u_3tev, y_d_3tev], dim=0)
    