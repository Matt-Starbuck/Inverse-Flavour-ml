import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import math
from src.inverse_flavour.inputs import CKM_device

from src.models.det_model import DetModel, reshape_output
from src.training.sample import Lambda_sample
from src.loss_functions.combined_loss import combined_loss
from src.loss_functions.hierarchy_loss import hierarchy_loss
from src.loss_functions.eft_bounds import compute_eft_violation_ratios
from src.inverse_flavour.get_initial_bsm import BSMInitialState
from src.physics.bsm_rge import BSMRGE
from src.inverse_flavour.inputs import b1, b2, b3

# This script loads a trained model and investigates the properties of the predicted BSM matrices (lambda_q, lambda_u, lambda_d) visaully
# as a function of Lambda BSM scales.

# Set which plots to produce
h_plot = True # plot the hierarchy of Yukawa singular values at the Planck scales as a function of Lambda BSM scale
p_plot = True # plot the phenomenological constraints on the BSM couplings as a function of Lambda BSM scale
rg_flow_plot = True # plot the scale variation of the yukawa singular value ratios for Lambda=3TeV
bsm_matrices_plot = True # plot the BSM coupling matrices at 3TeV and 100TeV

SAVE_PATH = "plots/" # Set the path to save the plots to
PATH = "src/training/det_model.pt" # Set the path to the saved model from training.py file
# Plot configuartions
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.1,
    'grid.linestyle': '--'
})

#######################################################
#Plotting functions
#######################################################

# Hierarchy loss
def plot_hierarchy(Lambda_batch: torch.Tensor, h_infos: dict, PATH: str):
    """ Plots the hierarchy loss functions as a function of Lambda BSM scale.
    """
    Lambda_batch_np = np.exp(torch.Tensor.numpy(Lambda_batch))
    # Invert the log^2 singular value ratios
    up_32_np = np.exp(np.sqrt(h_infos["up_32"].detach().numpy()))
    up_21_np = np.exp(np.sqrt(h_infos["up_21"].detach().numpy()))
    down_32_np = np.exp(np.sqrt(h_infos["down_32"].detach().numpy()))
    down_21_np = np.exp(np.sqrt(h_infos["down_21"].detach().numpy()))

    plt.figure(figsize=(8, 6))
    plt.plot(Lambda_batch_np, up_32_np, label=r'$y^{3}_{u} / y^{2}_{u}$', color='#1f77b4')
    plt.plot(Lambda_batch_np, up_21_np, label=r'$y^{2}_{u} / y^{1}_{u}$', color='#ff7f0e')
    plt.plot(Lambda_batch_np, down_32_np, label=r'$y^{3}_{d} / y^{2}_{d}$', color='#2ca02c')
    plt.plot(Lambda_batch_np, down_21_np, label=r'$y^{2}_{d} / y^{1}_{d}$', color='#d62728')
    plt.xlabel(r'BSM Scale $\Lambda$ (GeV)')
    plt.ylabel(r'Planck Scale Yukawa ratios ($y_{i}/y_{j}$)')
    plt.title(r'Yukawa Hierarchy Compression at $M_{\text{Pl}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.grid()
    plt.savefig(PATH + "hierarchy_loss_vs_lambda.pdf", format='pdf', bbox_inches='tight')
    plt.close()

# Pheno loss
def plot_pheno(Lambda_batch:torch.Tensor, p_infos:dict, PATH:str):
    """ Plots the phenomenological loss functions as a function of Lambda BSM scale.
    """
    Lambda_batch_np = np.exp(torch.Tensor.numpy(Lambda_batch))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    labels = [r'$D^0$', r'$K^0$', r'$B_d$', r'$B_s$', r'$uu \rightarrow uu$', r'$dd \rightarrow dd$', r'$qq \rightarrow qq$']
    keys = ["Im_D0_l", "Im_K0_l", "Bd_l", "Bs_l", "uu", "dd", "qq"]

    plt.figure(figsize=(8, 6))
    
    for key, color, label in zip(keys, colors, labels):
        data_np = p_infos[key].detach().cpu().numpy()
        plt.plot(Lambda_batch_np, data_np, label=label, color=color)

    plt.axhline(1.0, color='black', linestyle='--', linewidth=2.5, label='95% CL Bound')
    plt.xlabel(r'BSM Scale $\Lambda$ (GeV)')
    plt.ylabel(r'Wilson Coefficient Bounds ($|C_{\text{pred}}| / C_{\text{max}}$)')
    plt.title(r'Phenomenological Constraints on BSM Couplings')
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend(frameon=True, bbox_to_anchor=(1, 0.5), loc='center left', edgecolor='black')
    plt.tight_layout()
    plt.savefig(PATH + "pheno_loss_vs_lambda.pdf", format = 'pdf', bbox_inches='tight')
    plt.close()

# Evolution of Yukawa singular values
def plot_rg_evolution(bsm_init_state: torch.Tensor, log_lambda_val: torch.Tensor,
                      lambda_q: torch.Tensor, lambda_u: torch.Tensor, lambda_d: torch.Tensor,
                      PATH: str, n_steps: int=100):
    "Plots the evolution of the Yukawa singular values for BSM couplings outputted by the model"
    device = bsm_init_state.device
    rge_fun = BSMRGE(b1, b2, b3, log_lambda_val, lambda_q, lambda_u, lambda_d, device=device)
    evol_span = torch.linspace(0.0, 1.0, steps=n_steps, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        traj = odeint(rge_fun, bsm_init_state, evol_span, method='rk4')
    
    t_start = log_lambda_val.item()
    t_end = math.log(1e19)
    mu_span = np.exp(np.linspace(t_start, t_end, n_steps))

    yu_traj = traj[:, 0, 3:12].reshape(n_steps, 3, 3)
    yd_traj = traj[:, 0, 12:21].reshape(n_steps, 3, 3)

    yu_vals = torch.linalg.svdvals(yu_traj).cpu().numpy()
    yd_vals = torch.linalg.svdvals(yd_traj).cpu().numpy()

    fig, axes = plt.subplots(1,2, figsize=(14,6), sharey=True)

    # Plot Up-Type Evolution
    axes[0].plot(mu_span, yu_vals[:, 0], color='#d62728', label=r'$y_t$')
    axes[0].plot(mu_span, yu_vals[:, 1], color='#ff7f0e', label=r'$y_c$')
    axes[0].plot(mu_span, yu_vals[:, 2], color='#1f77b4', label=r'$y_u$')
    axes[0].set_title(r'Up-Type Yukawa Evolution')
    axes[0].set_xlabel(r'RG Scale $\mu$ (GeV)')
    axes[0].set_ylabel(r'Singular Values $\sigma_i(Y)$')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend(frameon=True, edgecolor='black')

    # Plot Down-Type Evolution
    axes[1].plot(mu_span, yd_vals[:, 0], color='#d62728', label=r'$y_b$')
    axes[1].plot(mu_span, yd_vals[:, 1], color='#ff7f0e', label=r'$y_s$')
    axes[1].plot(mu_span, yd_vals[:, 2], color='#1f77b4', label=r'$y_d$')
    axes[1].set_title(r'Down-Type Yukawa Evolution')
    axes[1].set_xlabel(r'RG Scale $\mu$ (GeV)')
    axes[1].set_xscale('log')
    axes[1].legend(frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig(PATH + "rg_evolution.pdf", format='pdf', bbox_inches='tight')
    plt.close()

# BSM coupling structures
def plot_bsm_couplings(lambda_q: torch.Tensor, lambda_u: torch.Tensor, lambda_d: torch.Tensor, PATH:str):
    """ Plots to visualise the learned coupling matrices. """

    # Load the CKM to check basis alignment
    VCKM = CKM_device(device=device)

    # Put the couplings to numpy for plotting.
    lambda_q_np = lambda_q.detach().cpu().numpy()
    lambda_u_np = lambda_u.detach().cpu().numpy()
    lambda_d_np = lambda_d.detach().cpu().numpy()

    ## Up mass basis print outs
    # Plot the matrices as heatmaps at 3TeV
    print("This is lambda_q at 3 TeV", lambda_q_np[0,...])
    print("This is lambda_u at 3 TeV", lambda_u_np[0,...])
    print("This is lambda_d at 3 TeV", lambda_d_np[0,...])

    # Plot the matrices as heatmaps at Lambda_max
    print(f"This is lambda_q at 100 TeV", lambda_q_np[-1,...])
    print(f"This is lambda_u at 100 TeV", lambda_u_np[-1,...])
    print(f"This is lambda_d at 100 TeV", lambda_d_np[-1,...])

    ## Down mass basis print outs
    print("This is lambda_q at 3 TeV in down-mass basis", VCKM.mH.resolve_conj().numpy() @ lambda_q_np[0,...])
    print(f"This is lambda_q at 100 TeV in down-mass basis", VCKM.mH.resolve_conj().numpy() @ lambda_q_np[-1,...])

    print("These are the lambda structures that enter the constraints")
    print("lambda_q @ lambda_q^dagger at 3 TeV", lambda_q_np[0,...] @ lambda_q_np[0,...].conj().T)
    print(f"lambda_q @ lambda_q^dagger at 100 TeV", lambda_q_np[-1,...] @ lambda_q_np[-1,...].conj().T)

if __name__ == "__main__":
    
    # Set the device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load the trained model and set to eval mode
    model_loaded = DetModel()
    model_loaded.load_state_dict(torch.load(PATH, weights_only=True))
    model_loaded.eval()

    # Generate a grid of Lambda values for plotting
    sample_size = 1000
    Lambda_batch = Lambda_sample(sample_size=sample_size, lambda_min=3000.0, lambda_max=100000.0, mode='log_grid')
    state_obj = BSMInitialState(lambda_min=3000.0, lambda_max=100000.0, steps=100, device=device)
    init_states = state_obj.get_init_states_train_test(Lambda_batch, split=1.0)
    lambda_flat = model_loaded(Lambda_batch)
    lambda_q, lambda_u, lambda_d = reshape_output(lambda_flat, batch_size=Lambda_batch.shape[0])

    # Extract the hierarchy and phenomenological dictionary information.
    h_infos = hierarchy_loss(init_states, torch.squeeze(Lambda_batch), lambda_q, lambda_u, lambda_d, n_steps=50)
    p_infos = compute_eft_violation_ratios(lambda_q, lambda_u, lambda_d, torch.squeeze(Lambda_batch))

    # If switched on, plot the Yukawa hierarchies as a function of Lambda
    if h_plot == True:
        plot_hierarchy(Lambda_batch, h_infos, SAVE_PATH)
    
    # Same for phenomenological constraints plot
    if p_plot == True:
        plot_pheno(Lambda_batch, p_infos, SAVE_PATH)

    if rg_flow_plot == True:
        bsm_state_3tev = init_states[0:1, ...]
        log_lambda_3tev = torch.squeeze(Lambda_batch)[0:1]
        lam_q_3tev = lambda_q[0:1, ...]
        lam_u_3tev = lambda_u[0:1, ...]
        lam_d_3tev = lambda_d[0:1, ...]
        plot_rg_evolution(bsm_state_3tev, log_lambda_3tev, lam_q_3tev, lam_u_3tev, lam_d_3tev, SAVE_PATH)
    
    if bsm_matrices_plot == True:
        plot_bsm_couplings(lambda_q, lambda_u, lambda_d, SAVE_PATH)