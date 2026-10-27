"""Shared utilities for the leakage-safe bearing-fault-diagnosis study.

Everything in this module is dependency-free apart from numpy/torch and is the
ONLY place that defines the repo-wide conventions.  Do not duplicate it.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "code")
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

for _d in (DATA_DIR, RAW_DIR, PROC_DIR, RESULTS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

CLASSES_10 = [
    "normal",
    "ball_007", "ball_014", "ball_021",
    "inner_007", "inner_014", "inner_021",
    "outer_007", "outer_014", "outer_021",
]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_noise(x, snr_db: float, kind: str = "gaussian", seed: int = 0):
    """Add noise to (..., L) signals so that SNR = snr_db.

    kind: gaussian | harmonic | impulse | drift
    Accepts numpy arrays or torch tensors, returns the same type.
    """
    is_torch = isinstance(x, torch.Tensor)
    arr = x.detach().cpu().numpy() if is_torch else np.asarray(x)
    dtype = arr.dtype
    arr = arr.astype(np.float64)
    rng = np.random.default_rng(seed)
    p_sig = np.mean(arr ** 2, axis=-1, keepdims=True) + 1e-12
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    L = arr.shape[-1]
    t = np.arange(L, dtype=np.float64)

    if kind == "gaussian":
        n = rng.standard_normal(arr.shape)
    elif kind == "harmonic":
        f = rng.uniform(0.02, 0.40, size=arr.shape[:-1] + (1,))
        n = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2.3 * f * t + 1.0)
    elif kind == "impulse":
        n = np.zeros_like(arr)
        k = max(1, L // 64)
        idx = rng.integers(0, L, size=arr.shape[:-1] + (k,))
        np.put_along_axis(n, idx, 3.0, axis=-1)
    elif kind == "drift":
        f = rng.uniform(0.002, 0.02, size=arr.shape[:-1] + (1,))
        n = np.sin(2 * np.pi * f * t + rng.uniform(0.0, 2 * np.pi))
    else:
        raise ValueError(f"unknown noise kind: {kind}")

    n = n / (np.sqrt(np.mean(n ** 2, axis=-1, keepdims=True)) + 1e-12) * np.sqrt(p_noise)
    out = (arr + n).astype(dtype)
    return torch.from_numpy(out) if is_torch else out


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        print(f"[timer] {self.name}: {self.elapsed:.2f}s")
