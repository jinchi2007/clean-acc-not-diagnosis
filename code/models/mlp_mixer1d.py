"""MLP-Mixer1D: an all-MLP architecture adapted to 1-D signals.

Reference
---------
I. Tolstikhin et al.,
"MLP-Mixer: An all-MLP Architecture for Vision", NeurIPS 2021.
https://arxiv.org/abs/2105.01601

Structure kept from the paper
-----------------------------
Patch embedding with a stride-patch convolution, then Mixer blocks that
alternate (a) token mixing -- an MLP applied across the token axis and shared
over channels -- and (b) channel mixing -- an MLP applied across channels and
shared over tokens -- with pre-LayerNorm and skip connections. The unified
head is mean-pooling over tokens -> LayerNorm -> linear.

Simplifications (documented for fairness)
-----------------------------------------
* small configuration (dim=128, 5 blocks) to stay lightweight; parameters are
  about 0.7M at in_len=1024 and 0.98M at 2048 (still under the 3M budget and
  independent of num_classes except for the last layer);
* no large-scale pretraining (forbidden by the study protocol: every baseline
  is trained from scratch);
* token-mixing MLPs are sized for the number of patches implied by in_len, so
  a model built for 1024 must be fed 1024-sample windows (a clear ValueError
  is raised otherwise).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixerBlock(nn.Module):
    """Token-mixing MLP followed by a channel-mixing MLP, both pre-norm."""

    def __init__(self, n_tokens, dim, token_hidden=None, channel_hidden=None,
                 dropout=0.0):
        super().__init__()
        token_hidden = token_hidden or 2 * n_tokens
        channel_hidden = channel_hidden or 4 * dim
        self.norm_tokens = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(n_tokens, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, n_tokens),
        )
        self.norm_channels = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # token mixing: (B, T, C) -> (B, C, T) so the MLP spans the token axis
        y = self.norm_tokens(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        x = x + self.drop(y)
        x = x + self.drop(self.channel_mix(self.norm_channels(x)))
        return x


class MLPMixer1D(nn.Module):
    """All-MLP 1-D model (Tolstikhin et al., NeurIPS 2021)."""

    def __init__(self, num_classes=10, in_len=1024, patch_size=16, dim=128,
                 depth=5, token_ratio=2.0, channel_ratio=4.0, dropout=0.1,
                 in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        self.patch_size = patch_size
        self.n_tokens = math.ceil(in_len / patch_size)

        self.patch_embed = nn.Conv1d(in_ch, dim, kernel_size=patch_size,
                                     stride=patch_size)
        token_hidden = int(self.n_tokens * token_ratio)
        channel_hidden = int(dim * channel_ratio)
        self.blocks = nn.Sequential(*[
            MixerBlock(self.n_tokens, dim, token_hidden, channel_hidden,
                       dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        patch = self.patch_size
        length = x.shape[-1]
        if length % patch:
            x = F.pad(x, (0, patch - length % patch))
            length += patch - length % patch
        if length // patch != self.n_tokens:
            raise ValueError(
                f"{type(self).__name__} was built for in_len={self.in_len} "
                f"({self.n_tokens} patches of {patch}), but got a window of "
                f"{length} samples; build a new model for that length")
        x = self.patch_embed(x).transpose(1, 2)          # (B, T, C)
        x = self.blocks(x)
        return self.head(self.norm(x.mean(dim=1)))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(MLPMixer1D, "mlp_mixer1d")
