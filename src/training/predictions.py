import torch
from src.models.det_model import DetModel, reshape_output
from src.training.sample import Lambda_sample
from src.loss_functions.combined_loss import combined_loss
from src.loss_functions.hierarchy_loss import hierarchy_loss
from src.loss_functions.eft_bounds import compute_eft_violation_ratios
from src.inverse_flavour.get_initial_bsm import BSMInitialState
import matplotlib.pyplot as plt
import numpy as np

# This script loads a trained model and investigates the properties of the predicted BSM matrices (lambda_q, lambda_u, lambda_d) visaully
# as a function of Lambda BSM scales.

#######################################################
#Plotting functions
#######################################################

SAVE_PATH = "src/training/plots/"

# Hierarchy loss
def plot_hierarchy(Lambda_batch:torch.Tensor, h_infos:dict, PATH:str):
    """ Plots the hierarchy loss functions as a function of Lambda BSM scale.
    """
    Lambda_batch_np = np.exp(torch.Tensor.numpy(Lambda_batch))
    up_32_np = np.exp(np.sqrt(h_infos["up_32"].detach().numpy()))
    up_21_np = np.exp(np.sqrt(h_infos["up_21"].detach().numpy()))
    down_32_np = np.exp(np.sqrt(h_infos["down_32"].detach().numpy()))
    down_21_np = np.exp(np.sqrt(h_infos["down_21"].detach().numpy()))

    plt.figure(figsize=(8, 6))
    plt.plot(Lambda_batch_np, up_32_np, label='up_32', color='blue')
    plt.plot(Lambda_batch_np, up_21_np, label='up_21', color='orange')
    plt.plot(Lambda_batch_np, down_32_np, label='down_32', color='green')
    plt.plot(Lambda_batch_np, down_21_np, label='down_21', color='red')
    plt.xlabel('Lambda (GeV)')
    plt.ylabel('yi/yj')
    plt.title('Hierarchy Loss Functions vs Lambda')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_PATH + "hierarchy_loss_vs_lambda.png", dpi=300)
    plt.show()

# Pheno loss

def plot_pheno(Lambda_batch:torch.Tensor, p_infos:dict, PATH:str):
    """ Plots the phenomenological loss functions as a function of Lambda BSM scale.
    """
    Lambda_batch_np = np.exp(torch.Tensor.numpy(Lambda_batch))
    D0_np = p_infos["D0"].detach().numpy()
    K0_np = p_infos["K0"].detach().numpy()
    Bd_np = p_infos["Bd"].detach().numpy()
    Bs_np = p_infos["Bs"].detach().numpy()
    uu_np = p_infos["uu"].detach().numpy()
    dd_np = p_infos["dd"].detach().numpy()
    qq_np = p_infos["qq"].detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(Lambda_batch_np, D0_np, label='D0', color='blue')
    plt.plot(Lambda_batch_np, K0_np, label='K0', color='orange')
    plt.plot(Lambda_batch_np, Bd_np, label='Bd', color='green')
    plt.plot(Lambda_batch_np, Bs_np, label='Bs', color='red')
    plt.plot(Lambda_batch_np, uu_np, label='uu', color='purple')
    plt.plot(Lambda_batch_np, dd_np, label='dd', color='brown')
    plt.plot(Lambda_batch_np, qq_np, label='qq', color='pink')
    plt.xlabel('Lambda (GeV)')
    plt.ylabel('EFT Violation Ratios')
    plt.title('Phenomenological Loss Functions vs Lambda')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_PATH + "pheno_vs_lambda.png", dpi=300)
    plt.show()

# Set the device to GPU if available, otherwise use CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set the path to the saved model from training.py file
PATH = "src/training/det_model.pt"

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

#print(model_loaded.state_dict())

h_infos = hierarchy_loss(init_states, torch.squeeze(Lambda_batch), lambda_q, lambda_u, lambda_d, n_steps=50)
p_infos = compute_eft_violation_ratios(lambda_q, lambda_u, lambda_d, torch.squeeze(Lambda_batch))
plot_hierarchy(Lambda_batch, h_infos, SAVE_PATH)
plot_pheno(Lambda_batch, p_infos, SAVE_PATH)

