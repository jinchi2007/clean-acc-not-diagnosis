#!/usr/bin/env python3
"""verify_env.py - one-shot health check for the M5 PyTorch/MPS environment.

Prints package versions, MPS availability, a short 1D-CNN training run on
synthetic bearing-vibration data, a resnet18 forward pass, and a rough
memory/throughput estimate.  Exit code 0 means the environment is usable.

    source setup/env.sh
    python setup/verify_env.py
"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PKGS = [
    "torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn",
    "matplotlib", "seaborn", "jupyterlab", "ipykernel", "tqdm", "PyYAML",
    "timm", "einops", "torchinfo", "captum", "ptflops", "openpyxl",
    "umap-learn",
]


def pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"


def print_versions() -> None:
    print("=" * 68)
    print("ENVIRONMENT")
    print("=" * 68)
    print(f"python           : {platform.python_version()} ({platform.python_implementation()})")
    print(f"executable       : {sys.executable}")
    print(f"platform         : {platform.platform()}")
    print(f"machine          : {platform.machine()}")
    print(f"torch            : {torch.__version__} (git {torch.version.git_version[:8]})")
    print(f"torch mps built  : {torch.backends.mps.is_built()}")
    print("-" * 68)
    for p in PKGS:
        print(f"{p:<17}: {pkg_version(p)}")
    print("-" * 68)


class BearingCNN1D(nn.Module):
    """Small 1D-CNN for raw vibration windows (bearing fault classification)."""

    def __init__(self, n_classes: int = 4, in_ch: int = 1):
        super().__init__()
        def block(cin, cout, k, s):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, s, k // 2, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
        self.features = nn.Sequential(
            block(in_ch, 16, 15, 2),   # 1024 -> 256
            block(16, 32, 7, 2),       # 256 -> 64
            block(32, 64, 5, 2),       # 64 -> 16
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Linear(64 * 4, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def make_bearing_data(n: int = 2048, length: int = 1024, n_classes: int = 4, seed: int = 0):
    """Synthetic bearing vibration: 1x-rev impulse train + resonance + noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 1, length), dtype=np.float32)
    y = rng.integers(0, n_classes, size=n)
    for i in range(n):
        cls = int(y[i])
        # fault characteristic frequency grows with class -> different impulse spacing
        f_char = 0.02 + 0.013 * cls          # cycles / sample
        resonance = 30.0 + 8.0 * cls
        impulses = np.zeros(length, dtype=np.float32)
        pos = (np.arange(0, length, max(1, int(1 / f_char))) + rng.integers(0, 5)) % length
        impulses[pos] = 1.0
        kernel = np.exp(-np.arange(64) / 10.0) * np.cos(
            2 * np.pi * resonance * np.arange(64) / length)
        sig = np.convolve(impulses, kernel, mode="same")
        sig += 0.08 * rng.standard_normal(length)
        x[i, 0] = sig
    x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-6)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def train_1dcnn(device: torch.device, epochs: int = 3, bs: int = 64, n: int = 2048):
    torch.manual_seed(0)
    x, y = make_bearing_data(n=n)
    model = BearingCNN1D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    # warm-up pass: the very first MPS kernel launch compiles Metal shaders
    # (several seconds) and is not representative of steady-state training.
    model.train()
    xb, yb = next(iter(dl))
    opt.zero_grad(set_to_none=True)
    loss_fn(model(xb.to(device)), yb.to(device)).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.synchronize()
    print("  warm-up batch done (first-launch shader compilation excluded)")

    epoch_times, last_loss = [], float("nan")
    peak_alloc = 0
    for ep in range(epochs):
        model.train()
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.numel()
            correct += (out.argmax(1) == yb).sum().item()
            tot += yb.numel()
        if device.type == "mps":
            torch.mps.synchronize()
            peak_alloc = max(peak_alloc, torch.mps.driver_allocated_memory())
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        last_loss = loss_sum / tot
        print(f"  epoch {ep + 1}/{epochs}: {dt * 1000:8.1f} ms  "
              f"({tot / dt:8.0f} samples/s)  loss={last_loss:.4f}  acc={correct / tot:.3f}")
    return epoch_times, last_loss, peak_alloc


def resnet18_forward(device: torch.device, bs: int = 32, iters: int = 10):
    from torchvision.models import resnet18
    model = resnet18(weights=None).to(device).eval()
    x = torch.randn(bs, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(3):                      # warm-up (kernel autotune / lazy init)
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
    per_iter = dt / iters
    return per_iter, per_iter / bs * 1e3  # s/iter, ms/sample


def main() -> int:
    print_versions()

    print("=" * 68)
    print("MPS / DEVICE")
    print("=" * 68)
    mps_avail = torch.backends.mps.is_available()
    print(f"torch.backends.mps.is_built()     : {torch.backends.mps.is_built()}")
    print(f"torch.backends.mps.is_available() : {mps_avail}")
    device = torch.device("mps" if mps_avail else "cpu")
    print(f"selected device                   : {device}")

    print("=" * 68)
    print(f"1D-CNN TRAINING on {device} (synthetic bearing vibration, 2048x1x1024, 3 epochs)")
    print("=" * 68)
    times, loss, peak = train_1dcnn(device)
    mean_ms = sum(times) / len(times) * 1000
    print(f"  mean epoch time : {mean_ms:.1f} ms   final loss: {loss:.4f}")

    print("=" * 68)
    print(f"resnet18 FORWARD on {device} (batch=32, 3x224x224, weights=None)")
    print("=" * 68)
    per_iter, per_sample = resnet18_forward(device)
    print(f"  {per_iter * 1000:.1f} ms / batch   {per_sample:.2f} ms / image   "
          f"({32 / per_iter:.1f} images/s)")

    print("=" * 68)
    print("MEMORY")
    print("=" * 68)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"process peak RSS (ru_maxrss)   : {rss_mb:.0f} MB")
    if mps_avail:
        print(f"mps current_allocated_memory   : {torch.mps.current_allocated_memory() / 2**20:.1f} MB")
        print(f"mps driver_allocated_memory    : {torch.mps.driver_allocated_memory() / 2**20:.1f} MB")
        print(f"mps recommended_max_working_set: {torch.mps.recommended_max_memory() / 2**30:.1f} GB")
    print(f"cpu cores                      : {os.cpu_count()}")

    print("=" * 68)
    ok = mps_avail and all(t > 0 for t in times)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  (mps available={mps_avail})")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
