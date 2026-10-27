"""TICNN: wide-conv + BN backbone used in the TICNN study.

Reference
---------
W. Zhang, C. Li, G. Peng, Y. Chen, Z. Zhang,
"A deep convolutional neural network with new training methods for bearing
fault diagnosis under noisy environment and different working load",
Mechanical Systems and Signal Processing 100 (2018) 439-453.
https://doi.org/10.1016/j.ymssp.2017.07.041

Structure kept from the paper
-----------------------------
* a wide first convolution (kernel 64) as the raw-signal front-end, with BN
  and ReLU after every convolution -- the "wide conv + BN" recipe that TICNN
  inherits and that is its most-copied architectural detail;
* stacked 3x1 conv blocks with BN + ReLU + max-pooling.

Simplifications (documented for fairness)
-----------------------------------------
* TICNN's actual contribution is a *training* recipe (interference / noise
  injection, dropout schedule), not a radically different backbone. That
  recipe belongs in the training script, so this module only implements the
  backbone. Architecture-only comparisons are therefore "WDCNN-like
  backbone" vs the other families, which we state explicitly in the paper;
* the paper's Flatten -> FC head is replaced by the unified GAP + linear head;
* stride 8 and four conv stages are used instead of the paper's stride 16 /
  five stages so that TICNN sits in the same 0.05-0.3M parameter band as the
  other baselines.
"""
from __future__ import annotations

import torch.nn as nn

from .blocks import ConvBNReLU, GAPHead


class TICNN(nn.Module):
    """Wide-conv + BN CNN (Zhang et al., MSSP 2018)."""

    def __init__(self, num_classes=10, in_len=1024,
                 channels=(32, 64, 128, 128), dropout=0.1, in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=64, stride=8, padding=32, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ConvBNReLU(c1, c2, 3), nn.MaxPool1d(2),
            ConvBNReLU(c2, c3, 3), nn.MaxPool1d(2),
            ConvBNReLU(c3, c4, 3), nn.MaxPool1d(2),
        )
        self.head = GAPHead(c4, num_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(TICNN, "ticnn")
