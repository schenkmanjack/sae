import torch
import matplotlib.pyplot as plt
import json
from .prepare_data import prepare_text_data

import os
import torch

def extract_sparse_features(sparse_autoencoder, dataloader, hidden_size, results_dir, device=None):
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists

    batch_idx = 0
    activation_files = []  # Track saved chunk files
    for index, x in dataloader:
        if device is not None:
            x = x.to(device)
        inputs = {
        'input_ids': x[0],
        'attention_mask': x[1]
        }
        with torch.no_grad():
            outputs = self.llm_model(**inputs)
        activation = self.mlp_activations[f'mlp_{self.target_layer}']
        activation = activation[:, -1, :]
        x = activation


        # Get the hidden activations from the sparse autoencoder
        _, hidden_activation = sparse_autoencoder(x)

        # Save the current batch's activations to a file
        batch_file = f"{results_dir}/sparse_features_batch_{batch_idx}.pth"
        torch.save(hidden_activation.detach().cpu(), batch_file)
        activation_files.append(batch_file)
        batch_idx += 1

    # Save metadata about activation chunks
    metadata_file = f"{results_dir}/activation_chunks.txt"
    with open(metadata_file, "w") as f:
        f.writelines("\n".join(activation_files))
    
    print(f"Activations saved in {len(activation_files)} chunks to {results_dir}/.")


import json
import torch
import numpy as np
import matplotlib.pyplot as plt


import json
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_sparse_features(results_dir, texts, k=10, epoch=0):
    # Load activation file paths
    with open(f"{results_dir}/activation_chunks.txt", "r") as f:
        activation_files = f.read().splitlines()

    # First pass: Determine global min and max for bin edges
    global_min = float("inf")
    global_max = float("-inf")

    for batch_file in activation_files:
        batch_activations = torch.load(batch_file).cpu().numpy().flatten()
        global_min = min(global_min, batch_activations.min())
        global_max = max(global_max, batch_activations.max())

    # Define consistent bin edges
    num_bins = 100
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)
    histogram_bins = np.zeros(num_bins)

    # Initialize buffers for top-k values and indices
    global_topk_values = {}
    global_topk_indices = {}

    # Process batches for histogram and top-k
    for batch_idx, batch_file in enumerate(activation_files):
        batch_activations = torch.load(batch_file)
        batch_size = batch_activations.size(0)

        # Update histogram bins
        batch_flattened = batch_activations.cpu().numpy().flatten()
        counts, _ = np.histogram(batch_flattened, bins=bin_edges)
        histogram_bins += counts

        # Update top-k for each feature
        for feature_idx in range(batch_activations.size(1)):  # Iterate over features
            feature_activations = batch_activations[:, feature_idx]

            # Get current batch's top-k
            batch_topk_values, batch_topk_indices = torch.topk(feature_activations, k)
            batch_topk_indices += batch_idx * batch_size  # Adjust indices for global dataset

            if feature_idx not in global_topk_values:
                # Initialize global buffers
                global_topk_values[feature_idx] = batch_topk_values
                global_topk_indices[feature_idx] = batch_topk_indices
            else:
                # Combine current batch's top-k with global top-k
                combined_values = torch.cat((global_topk_values[feature_idx], batch_topk_values))
                combined_indices = torch.cat((global_topk_indices[feature_idx], batch_topk_indices))

                # Recompute the top-k across combined data
                new_topk_values, new_topk_indices = torch.topk(combined_values, k)
                global_topk_values[feature_idx] = new_topk_values
                global_topk_indices[feature_idx] = combined_indices[new_topk_indices]

    # Plot histogram
    plt.bar(bin_edges[:-1], histogram_bins, width=np.diff(bin_edges), align="edge")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sparse Activations")
    plt.savefig(f"{results_dir}/histogram_sae.png")
    print(f"Histogram saved to {results_dir}/histogram_sae.png")

    # Map top-k indices to text examples
    key_examples = {}
    for feature_idx in global_topk_indices:
        key_examples[str(feature_idx)] = [texts[idx.item()] for idx in global_topk_indices[feature_idx]]

    # Save key examples to JSON
    with open(f"{results_dir}/features_{epoch}.json", "w") as f:
        json.dump(key_examples, f)
    print(f"Key examples saved to {results_dir}/features_{epoch}.json")



# def extract_sparse_features(sparse_autoencoder, dataloader, num_samples, hidden_size, results_dir, device=None):
#     history = torch.zeros(num_samples, hidden_size)
#     if device is not None:
#         history = history.to(device)
#     for index, x in dataloader:
#         if device is not None:
#             x = x.to(device)
#         reconstruction, hidden_activation = sparse_autoencoder(x)
#         history[index, :] = hidden_activation
#     # Save the activations
#     torch.save(history, f"{results_dir}/sparse_features.pth")
#     print(f"Sparse codes saved to {results_dir}/sparse_features.pth")

# def analyze_sparse_features(results_dir, texts,  k=10, epoch=0):
#     # load sparse activations history
#     history = torch.load(f"{results_dir}/sparse_features.pth")
#     # plot histogram of activations where history is a 2D tensor of shape (num_samples, hidden_size)
#     history_cpu = history.detach().cpu().numpy()
#     activations = history_cpu.flatten()
#     plt.hist(activations, bins=100)
#     plt.savefig(f'{results_dir}/histogram_sae.png')
#     # get the top k text samples for each activation
#     top_k_samples = []
#     key_examples = {}
#     for i in range(history.size(1)):
#         activation = history[:, i]
#         top_k_indices = torch.topk(activation, k).indices
#         key_examples[str(i)] = [texts[idx] for idx in top_k_indices.tolist()]
#         # top_k_samples.append(top_k_indices)

#     # save key_examples to json file
#     with open(f"{results_dir}/features_{epoch}.json", 'w') as f:
#         json.dump(key_examples, f)
#     print(f"Key examples saved to {results_dir}/features.json")
