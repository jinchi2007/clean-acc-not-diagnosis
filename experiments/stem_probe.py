"""Lead 探针：wide-kernel 前端的假设检验（不改 models 包，外层包一个 64x16 可学习带限下采样）。
若 resnet1d/ds_cnn/vit1d 加 stem 后噪声精度大幅回升，则「宽核前端」得到支持。
"""
import torch, torch.nn as nn
from code.common import set_seed, get_device
from code.models import build_model

class StemWrap(nn.Module):
    def __init__(self, base_name, num_classes, in_len, k=64, s=16):
        super().__init__()
        torch.manual_seed(0)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 1, k, s, bias=True), nn.BatchNorm1d(1), nn.ReLU(inplace=True))
        self.base = build_model(base_name, num_classes, in_len // s)
    def forward(self, x):
        return self.base(self.stem(x))

def count(m):
    return sum(p.numel() for p in m.parameters())

import code.models as M
print("orig params:", {n: count(build_model(n, 10, 1024)) for n in ("resnet1d", "ds_cnn", "vit1d")})
print("stem params:", {n: count(StemWrap(n, 10, 1024)) for n in ("resnet1d", "ds_cnn", "vit1d")})
print("stem out len:", StemWrap("resnet1d", 10, 1024).stem(torch.zeros(2,1,1024)).shape)

