"""WDCNN: wide first-layer CNN for raw vibration signals.

Reference
---------
W. Zhang, G. Peng, C. Li, Y. Chen, Z. Zhang,
"A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and
Domain Adaptation Ability on Raw Vibration Signals",
Sensors 17(2):425, 2017.  https://doi.org/10.3390/s17020425

Structure kept from the paper
-----------------------------
* wide first-layer convolution: 16 filters, kernel 64, stride 16 -- the large
  kernel acts as a band-pass / anti-noise front-end, which is WDCNN's
  defining feature (the paper's headline claim);
* then four 3x1 convolutions (32/64/64/64 channels), each followed by BN,
  ReLU and 2x max-pooling.

Simplifications (documented so the comparison stays honest)
-----------------------------------------------------------
* the paper ends with Flatten -> FC(100) -> FC(num_classes); we replace that
  with global average pooling + linear (the unified head of this study), so
  the parameter count does not depend on in_len;
* dropout defaults to 0.1 instead of the paper's 0.5.
"""
from __future__ import annotations

import torch.nn as nn

from .blocks import ConvBNReLU, GAPHead


class WDCNN(nn.Module):
    """Wide first-layer CNN (Zhang et al., Sensors 2017).

    Parameters
    ----------
    num_classes : number of output classes.
    in_len : input window length (any length works; 1024 / 2048 are used).
    channels : (c1..c5) channel widths of the five conv layers.
    dropout : dropout before the linear classifier.
    in_ch : input channels (default 1; >1 lets a multi-channel front-end
        feed this backbone, see build_stem_model in code.models).
    """

    def __init__(self, num_classes=10, in_len=1024,
                 channels=(16, 32, 64, 64, 64), dropout=0.1, in_ch=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_len = in_len
        c1, c2, c3, c4, c5 = channels
        self.features = nn.Sequential(
            # wide first layer: kernel 64, stride 16 (16 filters)
            nn.Conv1d(in_ch, c1, kernel_size=64, stride=16, padding=32, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ConvBNReLU(c1, c2, 3), nn.MaxPool1d(2),
            ConvBNReLU(c2, c3, 3), nn.MaxPool1d(2),
            ConvBNReLU(c3, c4, 3), nn.MaxPool1d(2),
            ConvBNReLU(c4, c5, 3), nn.MaxPool1d(2),
        )
        self.head = GAPHead(c5, num_classes, dropout)

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    from .blocks import run_smoke

    run_smoke(WDCNN, "wdcnn")
