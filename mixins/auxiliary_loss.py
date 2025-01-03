import torch

class AuxiliaryLoss:
    def post_optim_step(self, loss, epoch, x, hidden_activation):
        super().post_optim_step(loss, epoch, x, hidden_activation)
        