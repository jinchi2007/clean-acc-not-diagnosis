"""LeNet1D: the classic LeNet-5 adapted to one-dimensional signals.

Reference
---------
Y. LeCun, L. Bottou, Y. Bengio, P. Haffner,
"Gradient-based learning applied to document recognition",
Proceedings of the IEEE 86(11):2278-2324, 1998.
https://doi.org/10.1109/5.726791

Structure kept from the paper
-----------------------------
Two 5x1 convolutions (C1 -> S2 -> C3 -> S4) with pooling between them, i.e.
the canonical "two conv + pool stages" LeNet stack; the classifier is the
unified global-average-pooling + linear head.

Simplifications (documented for fairness)
-----------------------------------------
* the paper uses sigmoid activations and average pooling (with trainable
  S-layers); we use ReLU + 2x max-pooling, which is the standard 1-D
  adaptation and is what modern "LeNet baseline" numbers are reported with;
* the 120 / 84 fully connected layers are replaced by GAP + linear, so the
  parameter count collapses from about 60k to about 3k. We widen the conv
  channels from 6/16 to 16/32 to compensate a little; LeNet1D is therefore
  deliberately the *smallest* capacity baseline in the study.
"""
from __future__ import annotations

import torch.nn as nn

from .blocks import GAPHead


class LeNet1D(nn.Module):
    """LeNet-5 style 1-D CNN (LeCun et al., Proc. IEEE 1998)."""

    def __init__(self, num_classes=10, in_len=1024, channels=(16, 32),
                 dropout=0.0, in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=5, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.head = GAPHead(c2, num_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(LeNet1D, "lenet1d")
