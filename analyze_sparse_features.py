import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
from torch.utils.data import TensorDataset, DataLoader
from utils import prepare_text_data
from models import SparseAutoencoder




# Load the GPT-2 model and tokenizer
NUM_SAMPLES = 4000
dataloader, texts, llm_model = prepare_text_data(model_name='gpt2', num_samples=NUM_SAMPLES, batch_size=256)


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
config = dict(
    input_size = activations_tensor.size(1),
    overcompleteness = overcompleteness,
    hidden_size = activations_tensor.size(1) * overcompleteness,
    lr = 1e-4,
    batch_size = 128,
    sparsity_weight = 1e-6,
    epochs = 1000000
)

input_size = config['input_size']
hidden_size = config['hidden_size']
lr = config['lr']
batch_size = config['batch_size']
sparsity_weight = config['sparsity_weight']
epochs = config['epochs']
model_path = 'snapshots/sparse_autoencoder_l1_1e_6.pth'



sparse_autoencoder = SparseAutoencoder(input_size, hidden_size, sparsity_weight)
# load the model
sparse_autoencoder.load_state_dict(torch.load(model_path))
#  dataloader for activations_tensor
dataloader = DataLoader(activations_data, batch_size=batch_size, shuffle=False)
history = torch.zeros(NUM_SAMPLES, hidden_size)
for index, x in dataloader:
    reconstruction, hidden_activation = sparse_autoencoder(x)
    history[index, :] = hidden_activation

# Save the activations
history_path = 'snapshots/sparse_codes_l1_1e_6.pth'
torch.save(history, history_path)
print(f"Sparse codes saved to {history_path}")

    

