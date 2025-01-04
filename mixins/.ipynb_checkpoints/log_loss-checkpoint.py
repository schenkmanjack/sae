import torch
import wandb

class LogLoss:
    def post_epoch(self, loss, x, hidden_activation):
        epoch = self.epoch
        log_loss_config = self.config.get("log_loss_config", dict())
        log_epoch_freq = log_loss_config.get("log_epoch_freq", 10)
        if epoch % log_epoch_freq == 0:
            # log sparsity as percent nonzero
            sparsity = (hidden_activation != 0).sum() / hidden_activation.numel()
            if self.log_wandb:
                wandb.log({"Pct Nonzero": sparsity.item(), "Epoch": epoch + 1})
                wandb.log({"Epoch Loss": loss.item(), "Epoch": epoch + 1})
        super().post_epoch(loss, x, hidden_activation)