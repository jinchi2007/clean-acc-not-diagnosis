"""Registry of the eight lightweight 1-D baselines.

MODELS maps the fixed short name to its nn.Module subclass; build_model is
the only constructor the training/evaluation code needs::

    from code.models import MODELS, build_model
    model = build_model("wdcnn", num_classes=10, in_len=1024)
    model(torch.randn(4, 1, 1024))        # -> (4, 10)

Every model accepts (B, in_ch, in_len) and returns (B, num_classes). The
default in_ch=1 keeps the original contract; in_ch>1 exists so a learned
multi-channel front-end can feed a backbone (see build_stem_model below).
All of them end with global average pooling + a linear layer (one unified
head) and have fewer than 3M trainable parameters for in_len in {1024, 2048}.
No pretrained weights (timm / torchvision) are used anywhere: every baseline
is trained from scratch on the same data split, required for a fair
comparison.
"""
from __future__ import annotations

import torch.nn as nn

from .blocks import count_parameters
from .ds_cnn import DSCNN
from .lenet1d import LeNet1D
from .mcnn import MCNN
from .mlp_mixer1d import MLPMixer1D
from .resnet1d import ResNet1D
from .ticnn import TICNN
from .vit1d import ViT1D
from .wdcnn import WDCNN

#: name -> nn.Module subclass (insertion order is the canonical model order)
MODELS = {
    "wdcnn": WDCNN,
    "ticnn": TICNN,
    "mcnn": MCNN,
    "resnet1d": ResNet1D,
    "ds_cnn": DSCNN,
    "lenet1d": LeNet1D,
    "vit1d": ViT1D,
    "mlp_mixer1d": MLPMixer1D,
}

MODEL_NAMES = tuple(MODELS)

__all__ = [
    "MODELS",
    "MODEL_NAMES",
    "build_model",
    "StemModel",
    "build_stem_model",
    "count_parameters",
    "WDCNN",
    "TICNN",
    "MCNN",
    "ResNet1D",
    "DSCNN",
    "LeNet1D",
    "ViT1D",
    "MLPMixer1D",
]

#: default kernel / stride / channels of the stem front-end study.
STEM_DEFAULTS = {"k": 64, "s": 16, "c": 1}


def build_model(name: str, num_classes: int = 10, in_len: int = 1024,
                in_ch: int = 1, **kwargs) -> nn.Module:
    """Instantiate a baseline by name.

    Parameters
    ----------
    name : one of MODELS (see MODEL_NAMES).
    num_classes : number of output logits.
    in_len : input window length in samples (1024 / 2048 used in the study).
    in_ch : input channels. Defaults to 1, which reproduces the original
        behaviour; in_ch>1 makes the first convolution of every backbone
        accept that many channels.
    **kwargs : forwarded to the model constructor (e.g. dropout=0.0).
    """
    try:
        cls = MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(MODELS)}") from None
    return cls(num_classes=num_classes, in_len=in_len, in_ch=in_ch, **kwargs)


class StemModel(nn.Module):
    """Learned wide-kernel front-end followed by a baseline backbone.

    The front-end is Conv1d(1, c, k, stride=s) -> BatchNorm1d(c) -> activation
    (bias-free; BN supplies the shift). It maps a raw (B, 1, L) window to
    (B, c, L') with L' = floor((L - k) / s) + 1; the backbone is then built
    for that shorter length and with in_ch=c.

    Rationale: the WDCNN/TICNN families owe much of their noise robustness to
    their wide first convolution (kernel 64, stride 16), which acts as a
    learnable anti-aliasing band-pass filter. This wrapper gives any backbone
    the same front-end; c>1 keeps several filter banks so that the front-end
    is a multi-channel band-pass bank instead of a single lossy channel, which
    is the hypothesis tested by code.models.stem_ablation.
    """

    def __init__(self, front: nn.Module, base: nn.Module, base_name: str,
                 k: int, s: int, c: int, in_len: int, activation: str = "relu"):
        super().__init__()
        self.front = front
        self.base = base
        self.base_name = base_name
        self.k = k
        self.s = s
        self.c = c
        self.in_len = in_len
        self.in_len_stem = (in_len - k) // s + 1
        self.activation = activation

    def forward(self, x):
        return self.base(self.front(x))


def build_stem_model(base_name: str, num_classes: int = 10, in_len: int = 1024,
                     k: int = 64, s: int = 16, c: int = 1,
                     activation: str = "relu") -> StemModel:
    """Wrap a baseline in a learned wide-kernel front-end.

    Parameters
    ----------
    base_name : backbone name, one of MODELS.
    num_classes, in_len : as in build_model (in_len is the *raw* window).
    k, s : stem kernel size and stride (default 64 / 16, as in WDCNN).
    c : number of stem output channels fed to the backbone (default 1).
    activation : 'relu' (default) or 'gelu' after the stem BN.

    The returned module accepts (B, 1, in_len) and returns (B, num_classes);
    .base is the backbone, .front the front-end, .in_len_stem the length the
    backbone sees.
    """
    if base_name not in MODELS:
        raise KeyError(
            f"unknown base model {base_name!r}; available: {sorted(MODELS)}")
    if k < 1 or s < 1:
        raise ValueError(f"k and s must be positive, got k={k}, s={s}")
    if c < 1:
        raise ValueError(f"c must be >= 1, got c={c}")
    if in_len < k:
        raise ValueError(
            f"stem kernel k={k} is longer than the input in_len={in_len}")
    if activation not in ("relu", "gelu"):
        raise ValueError(
            f"activation must be 'relu' or 'gelu', got {activation!r}")

    l_stem = (in_len - k) // s + 1
    act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
    front = nn.Sequential(
        nn.Conv1d(1, c, kernel_size=k, stride=s, bias=False),
        nn.BatchNorm1d(c),
        act,
    )
    base = build_model(base_name, num_classes=num_classes, in_len=l_stem,
                       in_ch=c)
    return StemModel(front, base, base_name, k, s, c, in_len, activation)
