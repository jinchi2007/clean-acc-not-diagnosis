"""ViT1D: a small Vision-Transformer adapted to 1-D signals.

Reference
---------
A. Dosovitskiy et al.,
"An Image is Worth 16x16 Words: Transformers for Image Recognition at
Scale", ICLR 2021.  https://arxiv.org/abs/2010.11929

Structure kept from the paper
-----------------------------
Non-overlapping patch embedding (a stride-patch convolution applied to the
signal), learned positional embeddings, a stack of pre-norm Transformer
encoder layers (self-attention + GELU MLP), then the unified head: mean over
patch tokens -> LayerNorm -> linear classifier.

Simplifications (documented for fairness)
-----------------------------------------
* tiny dimensions (dim=64, 4 layers, 4 heads) because the study is about
  lightweight models on 1 M-sample datasets; parameters stay around 0.14M;
* no [CLS] token: mean pooling over tokens is used, matching the unified GAP
  head of the other baselines;
* positional embeddings are built for the requested in_len; if a forward pass
  receives a different number of patches they are linearly interpolated, so
  the same weights work at 1024 and 2048;
* no large-scale pretraining (the study forbids timm/torchvision pretrained
  weights), so this is trained from scratch like every other baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViT1D(nn.Module):
    """Small pre-norm ViT for 1-D signals (Dosovitskiy et al., ICLR 2021)."""

    def __init__(self, num_classes=10, in_len=1024, patch_size=16, dim=64,
                 depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.1, in_ch=1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_classes = num_classes
        self.in_len = in_len
        self.patch_size = patch_size
        self.embed_dim = dim
        self.n_tokens = math.ceil(in_len / patch_size)

        self.patch_embed = nn.Conv1d(in_ch, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        patch = self.patch_size
        if x.shape[-1] % patch:
            x = F.pad(x, (0, patch - x.shape[-1] % patch))
        x = self.patch_embed(x).transpose(1, 2)          # (B, T, C)
        pos = self.pos_embed
        if x.shape[1] != pos.shape[1]:
            pos = F.interpolate(pos.transpose(1, 2), size=x.shape[1],
                                mode="linear", align_corners=False)
            pos = pos.transpose(1, 2)
        x = self.encoder(x + pos)
        return self.head(self.norm(x.mean(dim=1)))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(ViT1D, "vit1d")
