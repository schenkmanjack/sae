import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def mask_patches(img, patch_size=8, mask_ratio=0.5):
    """
    Randomly masks patches of an image.
    Works for grayscale and RGB images with variable input sizes.

    Args:
        img (torch.Tensor): Input tensor of shape [B, C, H, W].
        patch_size (int): Size of the square patches.
        mask_ratio (float): Ratio of patches to mask (0 to 1).

    Returns:
        masked_img (torch.Tensor): Masked image of shape [B, C, H, W].
        mask_grid (torch.Tensor): Binary mask grid of shape [B, 1, num_h, num_w].
    """
    B, C, H, W = img.size()  # Batch size, Channels, Height, Width
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    num_masked = int(mask_ratio * num_patches)

    # Extract patches
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    B, C, num_h, num_w, patch_h, patch_w = patches.shape  # Patch dimensions
    assert num_h == num_patches_h and num_w == num_patches_w, "Mismatch in patch dimensions"

    # Flatten patches for masking
    patches = patches.reshape(B, C, num_h * num_w, patch_h, patch_w)

    # Create a random mask
    mask = torch.zeros(num_h * num_w, device=img.device)  # Flat mask
    mask[:num_masked] = 1
    mask = mask[torch.randperm(num_h * num_w)]  # Shuffle the mask
    mask = mask.reshape(1, 1, num_h * num_w, 1, 1).expand(B, C, num_h * num_w, patch_h, patch_w)

    # Apply the mask
    masked_patches = patches.clone()
    masked_patches[mask.bool()] = 0  # Zero out the masked patches
    masked_patches = masked_patches.reshape(B, C, num_h, num_w, patch_h, patch_w)

    # Reshape back to original image dimensions
    masked_img = masked_patches.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

    # Reshape the mask to match the patch grid dimensions
    mask_grid = mask[:, :, :, 0, 0].reshape(B, 1, num_h, num_w)
    return masked_img, mask_grid.bool()






# MAE Model
class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=128, hidden_dim=512):
        super(MaskedAutoencoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        num_patches = (img_size // patch_size) ** 2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(),
            nn.Linear(num_patches * embed_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, num_patches * embed_dim),
            nn.ReLU(),
            nn.Unflatten(1, (embed_dim, img_size // patch_size, img_size // patch_size)),
            nn.ConvTranspose2d(embed_dim, 1, kernel_size=patch_size, stride=patch_size),
        )
    

    def forward(self, x, mask):
        # Ensure the mask is Boolean and upsample to match input dimensions
        mask = mask.bool()
        mask = F.interpolate(mask.float(), size=x.shape[2:], mode='nearest').bool()  # Upsample
        masked_img = x * ~mask  # Apply the inverted mask
        encoded = self.encoder(masked_img)
        reconstructed = self.decoder(encoded)
        return reconstructed






# Load MNIST data
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 64
IMG_SIZE=32
PATCH_SIZE=4
EMBED_DIM=256
HIDDEN_DIM=1024

# transform = transforms.Compose([transforms.ToTensor()])
# mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# Load CIFAR-10 data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskedAutoencoder(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Initialize wandb
wandb.init(
    project="mnist_mae",  # Set your project name
    name="mnist_masked_autoencoder",  # Set your experiment/run name
)

# Training loop
epochs = EPOCHS
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        masked_data, mask = mask_patches(data, mask_ratio=0.5)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(masked_data, mask)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            # log loss
            wandb.log({"loss": loss.item()})
            # log image of output and data to wandb
            output_img = output[0].detach().cpu().numpy().squeeze()
            data_img = data[0].detach().cpu().numpy().squeeze()
            wandb.log({"output": [wandb.Image(output_img, caption="output")],
                       "data": [wandb.Image(data_img, caption="data")]})  

print("Training complete!")
