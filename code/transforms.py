"""Input representations shared by all experiments.

raw : the z-scored time-domain window, shape (N, 1, L)          -- standard input
ses : squared-envelope-spectrum, shape (N, 1, nfft//2+1)        -- frequency-aligned
      log1p + per-window z-score; phase is discarded on purpose, the amplitude
      modulation at the bearing defect orders is what carries the diagnosis.
"""
from __future__ import annotations

import numpy as np

INPUTS = ("raw", "ses")


def to_ses(x: np.ndarray, nfft: int = 2048) -> np.ndarray:
    """x: (N, 1, L) or (N, L) real windows -> (N, 1, nfft//2+1) log envelope spectrum."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[:, 0, :]
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    h = np.ones(X.shape[-1], dtype=np.float64)
    if X.shape[-1] > 2:
        h[1:-1] = 2.0
    env = np.abs(np.fft.irfft(X * h, n=n, axis=-1))
    e = env - env.mean(axis=-1, keepdims=True)
    S = np.abs(np.fft.rfft(e * np.hanning(n), n=nfft, axis=-1))
    S = np.log1p(S)
    S = (S - S.mean(axis=-1, keepdims=True)) / (S.std(axis=-1, keepdims=True) + 1e-8)
    return S[:, None, :].astype(np.float32)


def apply_input(x: np.ndarray, kind: str = "raw") -> np.ndarray:
    if kind in (None, "raw"):
        return x
    if kind == "ses":
        return to_ses(x)
    raise ValueError(f"unknown input representation: {kind}")
