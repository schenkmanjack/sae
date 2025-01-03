import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_weight=1e-5):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_weight = sparsity_weight
    def forward(self, x):
        hidden = self.encoder(x)
        hidden_activation = torch.relu(hidden)
        reconstruction = self.decoder(hidden_activation)
        return reconstruction, hidden_activation
    def loss(self, reconstruction, x, hidden_activation):
        reconstruction = reconstruction.to(x.device)
        hidden_activation = hidden_activation.to(x.device)
        decoder_weight = self.decoder.weight.to(x.device)
        mse_loss = nn.MSELoss()(reconstruction, x)
        # sparsity_loss = self.sparsity_weight * torch.norm(hidden_activation, 1)
        # Compute the L1 norm per sample, then average over the batch.
        sparsity_loss = self.sparsity_weight * torch.mean(torch.norm(hidden_activation, 1, dim=1)) * decoder_weight.norm(p=2, dim=0).mean()

        return mse_loss + sparsity_loss

