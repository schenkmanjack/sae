import torch
import wandb

class LogLoss:
    def post_batch(self, loss, x, hidden_activation):
        epoch = self.epoch
        if epoch % 400 == 0:
            # log sparsity as percent nonzero
            sparsity = (hidden_activation != 0).sum() / hidden_activation.numel()
            if self.log_wandb:
                wandb.log({"Pct Nonzero": sparsity.item(), "Epoch": epoch + 1})
                wandb.log({"Epoch Loss": loss.item(), "Epoch": epoch + 1})
        super().post_batch(loss, x, hidden_activation)