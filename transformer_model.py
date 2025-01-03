import torch
import torch.nn as nn
from einops import rearrange

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        # Feed-forward network
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, depth, mlp_ratio=4.0, dropout=0.1, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = nn.Linear(patch_size * patch_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer Layers
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # Prediction Head (to map embeddings back to patch space)
        self.head = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x):
        B, num_patches, patch_dim = x.shape

        # Patch Embedding
        x = self.patch_embed(x)  # Shape: [B, num_patches, embed_dim]
        x += self.positional_embedding

        # Transformer Layers
        for layer in self.transformer:
            x = layer(x)

        # Map back to pixel space
        x = self.head(x)  # Shape: [B, num_patches, patch_dim]
        return x
