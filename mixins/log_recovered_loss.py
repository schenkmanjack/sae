import torch
import wandb
from torch.nn import DataParallel

class LogRecoveredLoss:
    def post_epoch(self, loss, x, hidden_activation):
        epoch = self.epoch
        log_loss_config = self.config.get("log_recovered_loss_config", dict())
        log_epoch_freq = log_loss_config.get("log_epoch_freq", 40)
        if epoch % log_epoch_freq == 0:
            if self.log_wandb:
                ### Compute GPT-2 Loss Using SAE Reconstructions ###
                modified_activations = self.reconstruction  # Replace activations with SAE output
                # Manually inject activations into GPT-2
                with torch.no_grad():
                    # Replace original activations with SAE-reconstructed activations
                    self.mlp_activations[f'mlp_{self.target_layer}'] = modified_activations
                    llm_model = DataParallel(self.llm_model)
                    modified_outputs = llm_model(**self.inputs)  # GPT-2 forward pass with modified activations
                    reconstructed_loss = modified_outputs.loss  # GPT-2 loss with SAE activations
                ### Compute Recovered Loss ###
                recovered_loss = self.llm_loss / reconstructed_loss  # Ratio of original loss to modified loss
                # log WandB
                wandb.log({"Recovered Loss": recovered_loss.item(), "Epoch": epoch + 1})
        super().post_epoch(loss, x, hidden_activation)