import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
import wandb
from transformer_model import VisionTransformer


# Masking Function
def mask_patches(x, mask_ratio=0.5):
    """
    Randomly mask patches of an input tensor.

    Args:
        x: Tensor of shape [B, num_patches, patch_dim].
        mask_ratio: Ratio of patches to mask.

    Returns:
        masked_x: Tensor with masked patches set to zero.
        mask: Boolean mask indicating masked patches.
    """
    B, num_patches, patch_dim = x.shape
    num_masked = int(mask_ratio * num_patches)

    # Create a random mask
    mask = torch.ones(num_patches, device=x.device).bool()
    mask[:num_masked] = False
    mask = mask[torch.randperm(num_patches)].expand(B, -1)  # Randomize mask per batch

    # Mask patches
    masked_x = x.clone()
    masked_x[~mask] = 0
    return masked_x, mask


# Visualization and Logging Function
def log_images_to_wandb(data, masked_patches, mask, patches, outputs, patch_size, step):
    """
    Log original, masked, and predicted images to wandb.
    """
    B, C, H, W = data.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Convert patches back to images
    def patches_to_image(patches, mask=None):
        patches = patches.view(B, num_patches_h, num_patches_w, patch_size, patch_size)
        image = rearrange(patches, "b h w ph pw -> b 1 (h ph) (w pw)")
        if mask is not None:
            mask = mask.view(B, num_patches_h, num_patches_w)
            mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
            image[~mask.unsqueeze(1)] = 0  # Apply mask
        return image

    # Original Image
    original_image = data.cpu()

    # Masked Image
    masked_image = patches_to_image(masked_patches.cpu(), mask)

    # Predicted Image
    predicted_patches = outputs.clone().detach().view(B, num_patches_h, num_patches_w, patch_size, patch_size)
    predicted_image = rearrange(predicted_patches, "b h w ph pw -> b 1 (h ph) (w pw)").cpu()

    # Log the first image in the batch
    idx = 0
    wandb.log({
        "Original Image": wandb.Image(original_image[idx].squeeze(0), caption="Original Image"),
        "Masked Image": wandb.Image(masked_image[idx].squeeze(0), caption="Masked Image"),
        "Predicted Image": wandb.Image(predicted_image[idx].squeeze(0), caption="Predicted Image"),
        "Step": step
    })

def train_transformer():
    # Configuration
    config = {
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 128,
        "num_heads": 8,
        "depth": 10,
        "mlp_ratio": 8.0,
        "dropout": 0.1,
        "mask_ratio": 0.3,
        "epochs": 1000,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "save_path": "./best_model_lr_1e-4.pth"
    }

    # Initialize wandb
    wandb.init(project="transformer_vit", config=config)
    cfg = wandb.config

    # Dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Model
    model = VisionTransformer(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        depth=cfg.depth,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        img_size=cfg.img_size
    ).cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Training Loop
    best_loss = float("inf")  # Initialize best loss to infinity
    save_path = cfg.save_path
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            B, C, H, W = data.shape

            # Ensure image dimensions are divisible by patch size
            assert H % cfg.patch_size == 0 and W % cfg.patch_size == 0, "Image dimensions must be divisible by patch size"

            # Rearrange into patches, keeping the channel dimension
            patches = rearrange(data, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=cfg.patch_size, pw=cfg.patch_size)

            # Mask patches
            masked_patches, mask = mask_patches(patches, mask_ratio=cfg.mask_ratio)

            # Forward pass
            outputs = model(masked_patches)  # Predictions for all patches
            loss = criterion(outputs[~mask], patches[~mask])  # Loss on masked patches only

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            step += 1

        wandb.log({"Epoch Loss": total_loss / len(train_loader), "Epoch": epoch + 1})
        if epoch % 2 == 0:
            log_images_to_wandb(data, masked_patches, mask, patches, outputs, cfg.patch_size, step)
        print(f"Epoch {epoch + 1}/{cfg.epochs}, Loss: {total_loss / len(train_loader)}")
        # Save the best model
        epoch_loss = total_loss / len(train_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with loss {best_loss}")

    wandb.finish()

# Run Training
train_transformer()
