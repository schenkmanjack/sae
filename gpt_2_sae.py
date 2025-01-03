import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import TensorDataset, DataLoader
from utils import prepare_text_data
from models import SparseAutoencoder




# Load the GPT-2 model and tokenizer
dataloader, texts, llm_model = prepare_text_data(model_name='gpt2', num_samples=4, batch_size=16)


mlp_activations = {}
def get_activation(name):
    def hook(model, input, output):
        mlp_activations[name] = output.detach()
    return hook

all_activations = []
target_layer = 5
llm_model.transformer.h[target_layer].mlp.register_forward_hook(get_activation(f'mlp_{target_layer}'))

for batch in dataloader:
    inputs = {
        'input_ids': batch[0],
        'attention_mask': batch[1]
    }
    with torch.no_grad():
        outputs = llm_model(**inputs)
    activation = mlp_activations[f'mlp_{target_layer}']
    activation = activation[:, -1, :]
    all_activations.append(activation)
activations_tensor = torch.cat(all_activations, dim=0)
# add knowledge of index to have a touple of (activation, index)    
activations_data = [(index, activation) for index, activation in enumerate(activations_tensor)]

overcompleteness = 2
model_path = 'snapshots/sparse_autoencoder_l1_1e_6.pth'
config = dict(
    input_size = activations_tensor.size(1),
    overcompleteness = overcompleteness,
    hidden_size = activations_tensor.size(1) * overcompleteness,
    lr = 1e-4,
    batch_size = 128,
    sparsity_weight = 1e-6,
    model_path=model_path,
    epochs = 1000000
)

input_size = config['input_size']
hidden_size = config['hidden_size']
lr = config['lr']
batch_size = config['batch_size']
sparsity_weight = config['sparsity_weight']
epochs = config['epochs']


wandb.init(project="GPT2-sae", config=config)
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sparse_autoencoder = SparseAutoencoder(input_size, hidden_size, sparsity_weight)
sparse_autoencoder = sparse_autoencoder.to(device)
optimizer = optim.Adam(sparse_autoencoder.parameters(), lr=lr)
#  dataloader for activations_tensor
dataloader = DataLoader(activations_data, batch_size=batch_size, shuffle=True)
min_loss = torch.tensor(np.inf)
for epoch in range(epochs):
    for index, x in dataloader:
        x = x.to(device)
        reconstruction, hidden_activation = sparse_autoencoder(x)
        loss = sparse_autoencoder.loss(reconstruction, x, hidden_activation)

        optimizer.zero_grad()
        loss.backward()
        if loss < min_loss:
            min_loss = loss
            torch.save(sparse_autoencoder.state_dict(), model_path)
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    # log sparsity as percent nonzero
    sparsity = (hidden_activation != 0).sum() / hidden_activation.numel()
    wandb.log({"Pct Nonzero": sparsity.item(), "Epoch": epoch + 1})
    wandb.log({"Epoch Loss": loss.item(), "Epoch": epoch + 1})
    

