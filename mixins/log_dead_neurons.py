import torch
import wandb

class LogDeadNeurons:
    def __init__(self, config):
        super().__init__(config)
        self.dead_neurons = torch.zeros((self.batch_size, self.hidden_size), device=self.device)
    
    def post_batch(self, loss,  x, hidden_activation):
        # get percent of neurons dead in batch, neuron is dead if always off
        dead_neurons_batch = (hidden_activation.sum(dim=0) == 0)
        # get dead neurons of batch and dead neurons of all batches
        self.dead_neurons *= dead_neurons_batch
        super().post_batch(loss, x, hidden_activation)
    
    def post_epoch(self, loss,  x, hidden_activation):
        epoch = self.epoch
        if self.log_wandb:
            pct_dead_neurons = self.dead_neurons.sum().detach().cpu().item() / hidden_activation.size(1)
            wandb.log({"Pct Dead Neurons": pct_dead_neurons, "Epoch": epoch + 1})
            self.dead_neurons = torch.zeros((self.batch_size, self.hidden_size), device=self.device)

        super().post_epoch(loss, x, hidden_activation)