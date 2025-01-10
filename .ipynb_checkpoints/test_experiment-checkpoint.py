import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import TensorDataset, DataLoader
from utils import prepare_text_data
from models import SparseAutoencoder
from experiments import SAEBaseExperiment
from mixins import LogLoss, LogSparseFeatures, LogDeadNeurons, LogRecoveredLoss



overcompleteness = 10
input_size = 768
sae_save_path = 'snapshots/sae.pth'
config = dict(
    overcompleteness = overcompleteness,
    log_wandb=True,
    lr = 1e-4,
    batch_size = 128,
    model_config=dict(
        input_size = input_size,
        hidden_size = int(input_size * overcompleteness),
        sparsity_weight = 1e-4,
        use_auxiliary_loss = True,
        auxiliary_loss_weight = 0.1
    ),
    num_samples=800000,
    sae_save_path=sae_save_path,
    epochs = 1000000,
    results_dir = 'results_1',
    wandb_config = dict(
        project_name="gpt-2-sae"
    ),
    sparse_feature_config = dict(
        log_epoch_freq=100,

    ),
)

class ExperimentClass(LogRecoveredLoss, LogLoss, LogSparseFeatures, LogDeadNeurons, SAEBaseExperiment):
    pass

experiment = ExperimentClass(config)
experiment.train()



