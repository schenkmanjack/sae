import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import DataParallel
from utils import prepare_text_data
from models import SparseAutoencoder


class SAEBaseExperiment:
    def __init__(self, config):
        self.config = config
        self.lr = config['lr']
        self.num_samples = config.get("num_samples", 20)
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.results_dir = config.get("results_dir", "results")
        self.model_path = config.get("model_path", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # setup wandb
        self.log_wandb = self.config.get("log_wandb", False)
        if self.log_wandb:
            wandb_config = self.config.get("wandb_config", dict())
            self.wandb = wandb.init(project=wandb_config.get("project_name", "gpt-2-sae"), config=config)
        # create SAE
        self.model_config = self.config.get("model_config", dict())
        self.input_size = self.model_config['input_size']
        self.hidden_size = self.model_config['hidden_size']
        self.sparsity_weight = self.model_config['sparsity_weight']
        self.sparse_autoencoder = SparseAutoencoder(config=self.model_config)
        load_sae = self.config.get("load_sae", False)
        if load_sae:
            self.sparse_autoencoder.load_state_dict(torch.load(self.model_path))
        self.sparse_autoencoder = self.sparse_autoencoder.to(self.device)
        self.sae_save_path = self.config.get("sae_save_path", "sae.pt")
        self.optimizer = optim.Adam(self.sparse_autoencoder.parameters(), lr=self.lr)
        self.min_loss = torch.tensor(np.inf)
        # Load the GPT-2 model and tokenizer
        self.model_name = self.config.get("model_name", "gpt2")
        self.text_dataloader, self.text_dataloader_static, self.texts, self.tokenizer, self.llm_model = prepare_text_data(model_name=self.model_name, 
        num_samples=self.num_samples, batch_size=self.batch_size)
        self.llm_model = self.llm_model.to(self.device)
         # initialize activations storage
        self.mlp_activations = {}
        # place hook
        self.target_layer = 5

        def get_activation(name):
            def hook(model, input, output):
                self.mlp_activations[name] = output.detach()  # Modify instance variable
            return hook

        self.llm_model.transformer.h[self.target_layer].mlp.register_forward_hook(get_activation(f'mlp_{self.target_layer}'))

    def generate_activations(self):
        self.mlp_activations = {}
        self.all_activations = []

        for batch in self.text_dataloader:
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1]
            }
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
            activation = self.mlp_activations[f'mlp_{self.target_layer}']
            activation = activation[:, -1, :]
            self.all_activations.append(activation)
        activations_tensor = torch.cat(self.all_activations, dim=0)
        # add knowledge of index to have a touple of (activation, index)    
        activations_data = [(index, activation) for index, activation in enumerate(activations_tensor)]
        self.activations_data = activations_data


    def train(self):
        #  dataloader for activations_tensor
        # self.generate_activations()
        # dataloader = DataLoader(self.activations_data, batch_size=self.batch_size, shuffle=True)
        # self.activations_dataloader = dataloader
        # self.activations_dataloader_static = DataLoader(self.activations_data, batch_size=1024, shuffle=False)
        dataloader = self.text_dataloader
        self.min_loss = torch.tensor(np.inf)
        for epoch in range(self.epochs):
            self.epoch = epoch
            for batch_id, batch in enumerate(dataloader):
                x, attention_mask = batch
                x = x.to(self.device)
                attention_mask = attention_mask.to(self.device)
                self.batch_id = batch_id
                inputs = {
                'input_ids': x,
                'attention_mask': attention_mask,
                'labels': x.clone()
                }
                self.inputs = inputs
                with torch.no_grad():
                    llm_model = DataParallel(self.llm_model)
                    outputs = llm_model(**inputs)
                    self.llm_loss = outputs.loss
                activation = self.mlp_activations[f'mlp_{self.target_layer}']
                #activation = activation[:, -1, :]
                activation = activation.reshape(-1, activation.shape[-1])  # Flatten sequence & batch for SAE
                x = activation
                sparse_autoencoder = DataParallel(self.sparse_autoencoder)
                reconstruction, hidden_activation = sparse_autoencoder(x)
                self.reconstruction = reconstruction
                loss = self.sparse_autoencoder.loss(reconstruction, x, hidden_activation)

                self.optimizer.zero_grad()
                loss.backward()
                # if loss < min_loss:
                #     min_loss = loss
                #     torch.save(sparse_autoencoder.state_dict(), self.sae_save_path)
                self.optimizer.step()
                self.post_optim_step(loss, x, hidden_activation)
                self.post_batch(loss, x, hidden_activation)

            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')
            self.post_epoch(loss, x, hidden_activation)
    
    def post_optim_step(self, loss,  x, hidden_activation):
        pass
    
    def post_batch(self, loss,  x, hidden_activation):
        pass

    def post_epoch(self, loss,  x, hidden_activation):
        # save model
        loss = loss.to(self.min_loss.device)
        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(self.sparse_autoencoder.state_dict(), self.results_dir + '/sae.pth')

            

