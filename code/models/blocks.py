"""Reusable 1-D building blocks shared by the eight lightweight baselines.

The module is intentionally tiny and depends only on torch; every model in
code.models is a thin composition of these helpers plus its own
architecture-specific wiring. Keeping the classifier head here guarantees
that every baseline ends with the same "global average pooling + linear"
head, which is required for a fair parameter/latency comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """Conv1d -> BatchNorm1d -> ReLU (bias-free; BN supplies the shift)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=None,
                 groups=1, dilation=1):
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, groups=groups, dilation=dilation,
                      bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style depthwise + pointwise separable convolution block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm1d(in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn_dw(self.dw(x)))
        return self.act(self.bn_pw(self.pw(x)))


class GAPHead(nn.Sequential):
    """Global average pooling + optional dropout + linear classifier.

    This is the unified head of every baseline: it makes the parameter count
    independent of in_len and lets one model be evaluated at several window
    lengths without re-parameterising the classifier.
    """

    def __init__(self, in_ch, num_classes, dropout=0.0):
        layers = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_ch, num_classes))
        super().__init__(*layers)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (mirrors code.common.count_params)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_smoke(model_cls, name, in_lens=(1024, 2048), num_classes=10,
              batch=2, max_params=3_000_000):
    """Tiny CPU smoke test used by python -m code.models.<module>."""
    print(f"[{name}] {model_cls.__name__}")
    for in_len in in_lens:
        model = model_cls(num_classes=num_classes, in_len=in_len).eval()
        n_param = count_parameters(model)
        with torch.no_grad():
            out = model(torch.randn(batch, 1, in_len))
        assert tuple(out.shape) == (batch, num_classes), out.shape
        assert torch.isfinite(out).all(), "non-finite output"
        assert n_param < max_params, f"{name}: {n_param:,} params"
        print(f"  in_len={in_len:<5d} params={n_param:>9,d} "
              f"out={tuple(out.shape)}")
    print("  ok")
