"""DS-CNN: depthwise-separable CNN for one-dimensional signals.

References
----------
* depthwise separable convolutions: A. G. Howard et al.,
  "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
  Applications", arXiv:1704.04861, 2017.
* the DS-CNN baseline itself (depthwise-separable CNN for keyword spotting,
  the closest 1-D analogue): Y. Zhang, N. Suda, L. Lai, V. Chandra,
  "Hello Edge: Keyword Spotting on Microcontrollers", arXiv:1711.07128, 2017.

Structure kept from the paper
-----------------------------
A regular 5x1 convolution stem, then a stack of depthwise separable blocks
(depthwise 3x1 conv -> BN -> ReLU6 -> pointwise 1x1 conv -> BN -> ReLU6) with
stride-2 blocks interleaved for downsampling, then the unified GAP + linear
head. Separable convolutions cut multiply-adds by roughly 1/k^2 relative to a
dense convolution, which is the point of including this baseline.

Simplifications (documented for fairness)
-----------------------------------------
* no squeeze-and-excitation or hard-swish (the keyword-spotting variant used
  ReLU6, kept here);
* channel widths are chosen to sit in the same parameter band as the other
  baselines (about 0.13M here) so the comparison isolates the convolution
  type rather than capacity.
"""
from __future__ import annotations

import torch.nn as nn

from .blocks import ConvBNReLU, DepthwiseSeparableConv, GAPHead


class DSCNN(nn.Module):
    """Depthwise-separable CNN (Howard 2017 / Zhang 2017, 1-D adaptation)."""

    def __init__(self, num_classes=10, in_len=1024,
                 widths=(32, 64, 64, 128, 128, 256), dropout=0.1, in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        self.stem = ConvBNReLU(in_ch, widths[0], kernel_size=5, stride=2)
        blocks = []
        in_ch = widths[0]
        for i, out_ch in enumerate(widths[1:]):
            stride = 2 if i % 2 == 0 else 1
            blocks.append(DepthwiseSeparableConv(in_ch, out_ch,
                                                 kernel_size=3,
                                                 stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = GAPHead(in_ch, num_classes, dropout)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(DSCNN, "ds_cnn")
