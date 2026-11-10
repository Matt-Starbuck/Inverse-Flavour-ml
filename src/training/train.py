import torch
from torch import nn
from src.models.det_model import DetModel, reshape_output
from src.loss_functions.combined_loss import combined_loss
from src.inverse_flavour.get_initial_bsm import BSMInitialState
from src.training.sample import Lambda_sample
import math

# To save the model to PATH switch save to "ON"
save = "ON"
PATH = "src/training/det_model.pt"

torch.manual_seed(42)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set configurations for the training run
epochs = 5000 # Number of epochs used to train the model
sample_size = 100 # Number of BSM scale samples used in training and testing
batch_size = 8 # Batch size for training and testing
loss_fn = combined_loss
lambda_min = 3000.0 # Minimum BSM scale to consider in GeV (must be 3TeV since this is where the inputs are defined)
lambda_max = 100000.0 # Maximum BSM scale to consider in GeV (Can be anything greater than 3TeV)

# Initiate the model, initial parameter sizes, and optimizer
model_0 = DetModel(lambda_min=lambda_min, lambda_max=lambda_max).to(device)
# Start the parameters small! Otherwise they will run the yukawas to a Landau pole.
with torch.no_grad():
    for p in model_0.parameters():
        nn.init.normal_(p, mean=0.0, std=0.01)
optimizer = torch.optim.Adam(model_0.parameters(), lr=5e-5)

# Quick print test of the sampled Lambda values
Lambda_batch = Lambda_sample(sample_size=sample_size, lambda_min=lambda_min, lambda_max=lambda_max, mode='log_grid')
perm = torch.randperm(Lambda_batch.shape[0], device=device)
Lambda_batch = Lambda_batch[perm]
idx_split = 0.8
state_obj = BSMInitialState(lambda_min=lambda_min, lambda_max=lambda_max, steps=100, device=device)
init_states_train, init_states_test = state_obj.get_init_states_train_test(Lambda_batch, split=idx_split)
Lambda_train, Lambda_test = Lambda_batch[:int(idx_split * Lambda_batch.shape[0]),...], Lambda_batch[int(idx_split * Lambda_batch.shape[0]):,...]

# Checking test-train split sizes
print( f"This should have 0 dim:", {math.ceil((idx_split) * Lambda_batch.shape[0])}, init_states_train.shape)
print( f"This should have 0 dim:", {math.ceil((1-idx_split) * Lambda_batch.shape[0])}, init_states_test.shape)
print( f"This should have 0 dim:", {math.ceil(idx_split * Lambda_batch.shape[0])}, Lambda_train.shape)
print( f"This should have 0 dim:", {math.ceil((1-idx_split) * Lambda_batch.shape[0])}, Lambda_test.shape)

for epoch in range(epochs):
    # Training
    model_0.train()
    optimizer.zero_grad()
    # Forward pass
    lambda_flat = model_0(Lambda_train)
    lambda_q, lambda_u, lambda_d = reshape_output(lambda_flat, batch_size=Lambda_train.shape[0])
    loss, h_infos, p_infos = combined_loss(init_states_train, torch.squeeze(Lambda_train), lambda_q, lambda_u, lambda_d, n_steps=50, theta=0.5)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Testing
    model_0.eval()
    with torch.inference_mode():
        lambda_flat_test = model_0(Lambda_test)
        lambda_q_test, lambda_u_test, lambda_d_test = reshape_output(lambda_flat_test, batch_size=Lambda_test.shape[0])
        loss_test, h_infos_test, p_infos_test = combined_loss(init_states_test, torch.squeeze(Lambda_test), lambda_q_test, lambda_u_test, lambda_d_test, n_steps=50, theta=0.5)
        if epoch % 10 == 0:

            max_q, max_u, max_d = torch.max(torch.abs(lambda_q)), torch.max(torch.abs(lambda_u)), torch.max(torch.abs(lambda_d))
            max_q_t, max_u_t, max_d_t = torch.max(torch.abs(lambda_q_test)), torch.max(torch.abs(lambda_u_test)), torch.max(torch.abs(lambda_d_test))
            # Display training and testing metrics
            print(f"\nEpoch [{epoch+1:>{len(str(epochs))}}/{epochs}]")
            print(f"  ├─ [Train] Tot: {loss.item():>7.4f} | Hier: {h_infos:>7.4f} | Pheno: {p_infos:>7.4f} || Max |λq|: {max_q:>6.4f} | |λu|: {max_u:>6.4f} | |λd|: {max_d:>6.4f}")
            print(f"  └─ [Test ] Tot: {loss_test.item():>7.4f} | Hier: {h_infos_test:>7.4f} | Pheno: {p_infos_test:>7.4f} || Max |λq|: {max_q_t:>6.4f} | |λu|: {max_u_t:>6.4f} | |λd|: {max_d_t:>6.4f}")

# Save the model if save is on
if save == "ON":
    torch.save(model_0.state_dict(), PATH)
    print(f"Model saved successfully to {PATH}")


