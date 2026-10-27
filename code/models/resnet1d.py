"""ResNet1D: residual CNN adapted to one-dimensional signals.

Reference
---------
K. He, X. Zhang, S. Ren, J. Sun,
"Deep Residual Learning for Image Recognition", CVPR 2016.
https://doi.org/10.1109/CVPR.2016.90
(1-D adaptation widely used as a bearing-diagnosis baseline, e.g. the
open-source benchmark of Zhao et al., ISA Transactions 2020.)

Structure kept from the paper
-----------------------------
Residual learning with identity / 1x1-conv shortcuts: a stem (7x1 conv,
stride 2, then 3x1 max-pool), three stages of two basic blocks each with
channel widths 32 / 64 / 128 and stride-2 transition between stages, then
the unified global-average-pooling + linear head.

Simplifications (documented for fairness)
-----------------------------------------
* 2 blocks per stage (ResNet-18 style) instead of deeper variants;
* the classifier is GAP + linear rather than flatten + FC, as for every
  baseline in this study.
"""
from __future__ import annotations

import torch.nn as nn

from .blocks import ConvBNReLU, GAPHead


class BasicBlock1D(nn.Module):
    """Two 3x1 convolutions plus a (optionally projected) shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(out_ch))
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """ResNet-18-style residual CNN for 1-D signals (He et al., CVPR 2016)."""

    def __init__(self, num_classes=10, in_len=1024, widths=(32, 64, 128),
                 blocks_per_stage=2, dropout=0.1, in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, widths[0], kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages = []
        in_ch = widths[0]
        for stage, out_ch in enumerate(widths):
            stride = 1 if stage == 0 else 2
            stages.append(self._make_stage(in_ch, out_ch, blocks_per_stage,
                                           stride))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = GAPHead(in_ch, num_classes, dropout)

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride):
        blocks = [BasicBlock1D(in_ch, out_ch, stride)]
        blocks += [BasicBlock1D(out_ch, out_ch, 1)
                   for _ in range(n_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(ResNet1D, "resnet1d")
