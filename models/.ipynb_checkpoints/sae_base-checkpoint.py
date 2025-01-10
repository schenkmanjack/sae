import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, config):
        super(SparseAutoencoder, self).__init__()
        input_size = config.get("input_size")
        hidden_size = config.get("hidden_size")
        sparsity_weight = config.get("sparsity_weight", 1e-5)
        self.use_auxiliary_loss = config.get("auxiliary_loss", False)
        self.auxiliary_loss_weight = config.get("auxiliary_loss_weight", 0.1)
        self.config = config
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_weight = sparsity_weight
    def forward(self, x):
        hidden = self.encoder(x)
        hidden_activation = torch.relu(hidden)
        reconstruction = self.decoder(hidden_activation)
        return reconstruction, hidden_activation
    def loss(self, reconstruction, x, hidden_activation, dead_neurons=None):
        reconstruction = reconstruction.to(x.device)
        hidden_activation = hidden_activation.to(x.device)
        decoder_weight = self.decoder.weight.to(x.device)
        mse_loss = nn.MSELoss()(reconstruction, x)
        # sparsity_loss = self.sparsity_weight * torch.norm(hidden_activation, 1)
        # Compute the L1 norm per sample, then average over the batch.
        sparsity_loss = self.sparsity_weight * torch.mean(torch.norm(hidden_activation, 1, dim=1)) * decoder_weight.norm(p=2, dim=0).mean()
        # auxiliary_loss = torch.zeros_like(sparsity_loss).to(sparsity_loss.device)
        # if self.use_auxiliary_loss:
        #     auxiliary_loss = compute_auxiliary_loss(x=x, hidden_activation=hidden_activation, dead_neurons=dead_neurons)
        #     auxiliary_loss = auxiliary_loss * self.auxiliary_loss_weight

        return mse_loss + sparsity_loss # + auxiliary_loss

    def compute_auxiliary_loss(self, x, hidden_activation, dead_neurons):
        """
        Compute an additional loss term by performing a forward pass 
        using only the neurons identified as dead.
        """
        batch_size, hidden_size = hidden_activation.shape

        # Create a Mask from the `dead_neurons` Binary Vector
        mask = dead_neurons.float().unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, hidden_size)
        # Pass x throuhg encoder
        hidden = self.encoder(x)
        # Apply activation
        hidden_activation = torch.exp(hidden)
        # Mask non-dead neurons
        hidden_activation = hidden_activation * mask
        # Compute New Forward Pass with Masked Activation
        masked_reconstruction = self.decoder(hidden_activation)  # Decode new activations
        # Compute Extra MSE Loss
        extra_mse_loss = nn.MSELoss()(masked_reconstruction, x)  

        return extra_mse_loss  # Add to total loss

