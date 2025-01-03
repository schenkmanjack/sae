import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import DataParallel
# from utils import extract_sparse_features, analyze_sparse_features


import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os


class LogSparseFeatures:
    def post_epoch(self, loss, x, hidden_activation):
        sparse_feature_config = self.config.get("sparse_feature_config", {})
        log_epoch_freq = sparse_feature_config.get("log_epoch_freq", 100)
        epoch = self.epoch
        if epoch % log_epoch_freq == 0:
            self.analyze_sparse_features()
        super().post_epoch(loss, x, hidden_activation)

    def analyze_sparse_features(self):
        os.makedirs(self.results_dir, exist_ok=True)  # Ensure the results directory exists

        # Initialize variables for histogram and top-k
        global_min = float("inf")
        global_max = float("-inf")
        num_bins = 100
        k = 10
        bin_edges = None
        histogram_bins = None

        global_topk_values = {}
        global_topk_indices = {}

        # Process each batch
        for batch_idx, batch in enumerate(self.text_dataloader_static):
            x, attention_mask = batch
            x = x.to(self.device)
            attention_mask = attention_mask.to(self.device)
            inputs = {
                'input_ids': x,
                'attention_mask': attention_mask
            }

            with torch.no_grad():
                llm_model = DataParallel(self.llm_model)
                outputs = llm_model(**inputs)
            
            activation = self.mlp_activations[f'mlp_{self.target_layer}']
            activation = activation[:, -1, :]

            # Get the hidden activations from the sparse autoencoder
            sparse_autoencoder = DataParallel(self.sparse_autoencoder)
            _, hidden_activation = sparse_autoencoder(activation)

            # Update global min/max for histogram binning
            batch_min = hidden_activation.min().item()
            batch_max = hidden_activation.max().item()
            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)

            # Initialize histogram bins on the first pass
            if bin_edges is None:
                bin_edges = np.linspace(global_min, global_max, num_bins + 1)
                histogram_bins = np.zeros(num_bins)

            # Compute histogram for this batch
            batch_flattened = hidden_activation.detach().cpu().numpy().flatten()
            counts, _ = np.histogram(batch_flattened, bins=bin_edges)
            histogram_bins += counts

            # Update top-k for each feature
            for feature_idx in range(hidden_activation.size(1)):
                feature_activations = hidden_activation[:, feature_idx]

                # Get current batch's top-k
                batch_topk_values, batch_topk_indices = torch.topk(feature_activations, k)
                batch_topk_indices += batch_idx * hidden_activation.size(0)

                if feature_idx not in global_topk_values:
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
        plt.savefig(f"{self.results_dir}/histogram_sae.png")
        print(f"Histogram saved to {self.results_dir}/histogram_sae.png")

        # Map top-k indices to text examples
        key_examples = {}
        for feature_idx in global_topk_indices:
            key_examples[str(feature_idx)] = [self.texts[idx.item()] for idx in global_topk_indices[feature_idx]]

        # Save key examples to JSON
        with open(f"{self.results_dir}/features_{self.epoch}.json", "w") as f:
            json.dump(key_examples, f)
        print(f"Key examples saved to {self.results_dir}/features_{self.epoch}.json")

# class LogSparseFeatures:
#     def post_epoch(self, loss,  x, hidden_activation):
#         sparse_feature_path = self.config.get("sparse_feature_path", "sparse_features.pt")
#         sparse_feature_config = self.config.get("sparse_feature_config", {})
#         log_epoch_freq = sparse_feature_config.get("log_epoch_freq", 100)
#         epoch = self.epoch
#         if epoch % log_epoch_freq == 0:
#             # extract sparse features
#             self.extract_sparse_features()
#             self.analyze_sparse_features()
#             # extract_sparse_features(sparse_autoencoder=self.sparse_autoencoder, 
#             # dataloader=self.activations_dataloader, num_samples=self.num_samples, 
#             # hidden_size=self.hidden_size, 
#             # results_dir=self.results_dir,
#             # device=self.device)

#             # # analyze sparse features
#             # analyze_sparse_features(results_dir=self.results_dir, texts=self.texts, k=10, epoch=self.epoch)
#         super().post_epoch(loss, x, hidden_activation)

    
#     def extract_sparse_features(self):
#         os.makedirs(self.results_dir, exist_ok=True)  # Ensure the results directory exists

#         batch_idx = 0
#         activation_files = []  # Track saved chunk files
#         for index, x in self.text_dataloader_static:
#             x = x.to(self.device)
#             inputs = {
#             'input_ids': x[0],
#             'attention_mask': x[1]
#             }
#             with torch.no_grad():
#                 outputs = self.llm_model(**inputs)
#             activation = self.mlp_activations[f'mlp_{self.target_layer}']
#             activation = activation[:, -1, :]
#             x = activation
#             # Get the hidden activations from the sparse autoencoder
#             _, hidden_activation = self.sparse_autoencoder(x)
#             # Save the current batch's activations to a file
#             batch_file = f"{self.results_dir}/sparse_features_batch_{batch_idx}.pth"
#             torch.save(hidden_activation.detach().cpu(), batch_file)
#             activation_files.append(batch_file)
#             batch_idx += 1

#         # Save metadata about activation chunks
#         metadata_file = f"{self.results_dir}/activation_chunks.txt"
#         with open(metadata_file, "w") as f:
#             f.writelines("\n".join(activation_files))
        
#         print(f"Activations saved in {len(activation_files)} chunks to {self.results_dir}/.")
    
#     def analyze_sparse_features(self):
#         # Load activation file paths
#         with open(f"{self.results_dir}/activation_chunks.txt", "r") as f:
#             activation_files = f.read().splitlines()

#         # First pass: Determine global min and max for bin edges
#         global_min = float("inf")
#         global_max = float("-inf")

#         for batch_file in activation_files:
#             batch_activations = torch.load(batch_file).cpu().numpy().flatten()
#             global_min = min(global_min, batch_activations.min())
#             global_max = max(global_max, batch_activations.max())

#         # Define consistent bin edges
#         num_bins = 100
#         k = 10
#         bin_edges = np.linspace(global_min, global_max, num_bins + 1)
#         histogram_bins = np.zeros(num_bins)

#         # Initialize buffers for top-k values and indices
#         global_topk_values = {}
#         global_topk_indices = {}

#         # Process batches for histogram and top-k
#         for batch_idx, batch_file in enumerate(activation_files):
#             batch_activations = torch.load(batch_file)
#             batch_size = batch_activations.size(0)

#             # Update histogram bins
#             batch_flattened = batch_activations.cpu().numpy().flatten()
#             counts, _ = np.histogram(batch_flattened, bins=bin_edges)
#             histogram_bins += counts

#             # Update top-k for each feature
#             for feature_idx in range(batch_activations.size(1)):  # Iterate over features
#                 feature_activations = batch_activations[:, feature_idx]

#                 # Get current batch's top-k
#                 batch_topk_values, batch_topk_indices = torch.topk(feature_activations, k)
#                 batch_topk_indices += batch_idx * batch_size  # Adjust indices for global dataset

#                 if feature_idx not in global_topk_values:
#                     # Initialize global buffers
#                     global_topk_values[feature_idx] = batch_topk_values
#                     global_topk_indices[feature_idx] = batch_topk_indices
#                 else:
#                     # Combine current batch's top-k with global top-k
#                     combined_values = torch.cat((global_topk_values[feature_idx], batch_topk_values))
#                     combined_indices = torch.cat((global_topk_indices[feature_idx], batch_topk_indices))

#                     # Recompute the top-k across combined data
#                     new_topk_values, new_topk_indices = torch.topk(combined_values, k)
#                     global_topk_values[feature_idx] = new_topk_values
#                     global_topk_indices[feature_idx] = combined_indices[new_topk_indices]

#         # Plot histogram
#         plt.bar(bin_edges[:-1], histogram_bins, width=np.diff(bin_edges), align="edge")
#         plt.xlabel("Activation Value")
#         plt.ylabel("Frequency")
#         plt.title("Histogram of Sparse Activations")
#         plt.savefig(f"{self.results_dir}/histogram_sae.png")
#         print(f"Histogram saved to {self.results_dir}/histogram_sae.png")

#         # Map top-k indices to text examples
#         key_examples = {}
#         for feature_idx in global_topk_indices:
#             key_examples[str(feature_idx)] = [texts[idx.item()] for idx in global_topk_indices[feature_idx]]

#         # Save key examples to JSON
#         with open(f"{results_dir}/features_{epoch}.json", "w") as f:
#             json.dump(key_examples, f)
#         print(f"Key examples saved to {results_dir}/features_{epoch}.json")

        