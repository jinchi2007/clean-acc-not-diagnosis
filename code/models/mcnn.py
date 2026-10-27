"""MCNN: multi-scale parallel-convolution CNN.

References
----------
* multi-scale CNN for vibration diagnosis (the MCNN family):
  G. Jiang, H. He, J. Yan, P. Xie,
  "Multiscale Convolutional Neural Networks for Fault Diagnosis of Wind
  Turbine Gearbox", IEEE Transactions on Industrial Electronics 66(4),
  3196-3207, 2019.  https://doi.org/10.1109/TIE.2018.2833033
* the parallel multi-branch idea itself: C. Szegedy et al., "Going Deeper
  with Convolutions" (Inception), CVPR 2015.

Structure kept from the paper family
------------------------------------
Three multi-scale stages. Each stage runs several convolutions with different
kernel sizes (3 / 7 / 15) in parallel on the same input and concatenates
their outputs, so the network sees short, medium and long vibration patterns
at the same depth. Max-pooling between stages; unified GAP + linear head.

Simplifications (documented for fairness)
-----------------------------------------
* channel widths (24/32/48 per branch) are chosen to keep the model in the
  same parameter band as the other baselines (<0.2M here);
* an Inception-style 1x1 bottleneck is not used (the paper family varies
  widely; the essential feature is the parallel multi-scale convolution).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNReLU, GAPHead


class MultiScaleBlock(nn.Module):
    """Parallel convolutions with different kernel sizes, concatenated."""

    def __init__(self, in_ch, out_ch, kernels=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNReLU(in_ch, out_ch, kernel_size=k) for k in kernels])

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class MCNN(nn.Module):
    """Multi-scale parallel-convolution CNN (Jiang et al., IEEE TIE 2019)."""

    def __init__(self, num_classes=10, in_len=1024, kernels=(3, 7, 15),
                 widths=(24, 32, 48), stem=16, dropout=0.1, in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        self.kernels = tuple(kernels)
        c1, c2, c3 = widths
        n_branch = len(kernels)
        self.features = nn.Sequential(
            ConvBNReLU(in_ch, stem, kernel_size=3),
            nn.MaxPool1d(2),
            MultiScaleBlock(stem, c1, kernels), nn.MaxPool1d(2),
            MultiScaleBlock(n_branch * c1, c2, kernels), nn.MaxPool1d(2),
            MultiScaleBlock(n_branch * c2, c3, kernels), nn.MaxPool1d(2),
        )
        self.head = GAPHead(n_branch * c3, num_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(MCNN, "mcnn")
