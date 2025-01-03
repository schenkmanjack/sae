import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
from torch.utils.data import TensorDataset, DataLoader
import json
import matplotlib.pyplot as plt
from utils import prepare_text_data
from models import SparseAutoencoder


# load sparse activations history
history_path = 'snapshots/sparse_codes_l1_1e_6.pth'
history = torch.load(history_path)
# Load the GPT-2 model and tokenizer
NUM_SAMPLES = 4000
dataloader, texts, llm_model = prepare_text_data(model_name='gpt2', num_samples=NUM_SAMPLES, batch_size=256)


# plot histogram of activations where history is a 2D tensor of shape (num_samples, hidden_size)
history_cpu = history.detach().cpu().numpy()
activations = history_cpu.flatten()
plt.hist(activations, bins=100)
plt.savefig('results/histogram_l1_1e6.png')

# get the top k text samples for each activation
k = 10
top_k_samples = []
key_examples = {}
for i in range(history.size(1)):
    activation = history[:, i]
    top_k_indices = torch.topk(activation, k).indices
    key_examples[str(i)] = [texts[idx] for idx in top_k_indices.tolist()]
    # top_k_samples.append(top_k_indices)

# save key_examples to json file
with open('results/key_examples_l1_1e6.json', 'w') as f:
    json.dump(key_examples, f)

# 

