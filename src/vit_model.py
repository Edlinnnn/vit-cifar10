"""
vit_model_torch.py
------------------
Vision Transformer (ViT) implemented in PyTorch.
Equivalent architecture to the Keras version.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, embed_dim=64):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        # Conv2D with stride=patch_size extracts and projects patches in one step
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)               # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                     # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)               # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                 embed_dim=64, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        mlp_dim     = int(embed_dim * mlp_ratio)

        self.patch_embed  = PatchEmbedding(image_size, patch_size, embed_dim)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed    = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.embed_drop   = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.embed_drop(x + self.pos_embed)

        x   = self.blocks(x)
        x   = self.norm(x)
        cls_out = x[:, 0]          # CLS token output
        return self.head(cls_out)  # logits (no softmax — CrossEntropyLoss handles it)


if __name__ == "__main__":
    model = ViT()
    dummy = torch.randn(4, 3, 32, 32)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # (4, 10)
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")