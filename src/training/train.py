import torch
from torch import nn
from src.models.det_model import DetModel, reshape_output
from src.loss_functions.combined_loss import combined_loss
from src.inverse_flavour.get_initial_bsm import BSMInitialState
from src.training.sample import Lambda_sample
import math

torch.manual_seed(42)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_0 = DetModel().to(device)

# Start the parameters small! Otherwise they will run the yukawas to a Landau pole.
with torch.no_grad():
    for p in model_0.parameters():
        nn.init.normal_(p, mean=0.0, std=0.01)

optimizer = torch.optim.Adam(model_0.parameters(), lr=1e-4)
epochs = 2000
sample_size = 100
batch_size = 8
loss_fn = combined_loss

# Quick print test of the sampled Lambda values
Lambda_batch = Lambda_sample(sample_size=sample_size, lambda_min=3000.0, lambda_max=100000.0, mode='log_grid')
idx_split = 0.8
state_obj = BSMInitialState(lambda_min=3000.0, lambda_max=100000.0, steps=100, device=device)
init_states_train, init_states_test = state_obj.get_init_states_train_test(Lambda_batch, split=idx_split)
Lambda_train, Lambda_test = Lambda_batch[:int(idx_split * Lambda_batch.shape[0]),...], Lambda_batch[int(idx_split * Lambda_batch.shape[0]):,...]

# Checking test-train split sizes
print( f"This should have 0 dim:", {math.ceil((idx_split) * Lambda_batch.shape[0])}, init_states_train.shape)
print( f"This should have 0 dim:", {math.ceil((1-idx_split) * Lambda_batch.shape[0])}, init_states_test.shape)
print( f"This should have 0 dim:", {math.ceil(idx_split * Lambda_batch.shape[0])}, Lambda_train.shape)
print( f"This should have 0 dim:", {math.ceil((1-idx_split) * Lambda_batch.shape[0])}, Lambda_test.shape)

for epoch in range(epochs):
    model_0.train()
    optimizer.zero_grad()
    # Forward pass
    lambda_flat = model_0(Lambda_train)
    lambda_q, lambda_u, lambda_d = reshape_output(lambda_flat, batch_size=Lambda_train.shape[0])
    loss, h_infos, p_infos = combined_loss(init_states_train, torch.squeeze(Lambda_train), lambda_q, lambda_u, lambda_d, n_steps=50, theta=0.5)
    if epoch %10 ==0:
        print(f"Epoch [{epoch+1}/{epochs}]|Total Loss: {loss.item():.4f}|Hierarchy Loss: {h_infos:.4f}|Phenomenology Loss: {p_infos:.4f}|Max lq coupling: {torch.max((torch.abs(lambda_q))):.4f}|Max lu coupling: {torch.max((torch.abs(lambda_u))):.4f} | Max ld coupling: {torch.max((torch.abs(lambda_d))):.4f}")
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

