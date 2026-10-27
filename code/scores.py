"""Decision-layer scores for open-set rejection (E10).

E3/E9 use the softmax maximum as the rejection score.  The score is a free
choice once a network is trained, so this module implements two standard
alternatives that need no extra training and no extra parameters:

    energy      -T * log sum_j exp(z_j / T)      (Liu et al., NeurIPS 2020)
    mahalanobis -min_k Mahalanobis(z; mu_k, Sigma), shared covariance fitted
                on the logit vectors of the training set (logit-space variant
                of Lee et al., NeurIPS 2018)

All functions return "higher means more likely to be a known class", so they
plug into the same auroc(unknown, known) utility used by code.train.
"""
from __future__ import annotations

import numpy as np


def softmax_max(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.max(axis=1)


def energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    r"""Negative free energy $T\log\sum_j e^{z_j/T}$: higher = more familiar.

    (The free energy itself is $E = -T\log\sum_j e^{z_j/T}$, which is lower for
    in-distribution inputs, so the score used here is $-E$.)
    """
    z = logits.astype(np.float64) / T
    m = z.max(axis=1, keepdims=True)
    return T * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


class LogitMahalanobis:
    """Class-conditional Gaussians with a shared covariance in logit space."""

    def __init__(self, logits: np.ndarray, labels: np.ndarray, eps: float = 1e-3):
        f = logits.astype(np.float64)
        self.classes = np.unique(labels)
        self.means = np.stack([f[labels == c].mean(axis=0) for c in self.classes])
        resid = np.concatenate([f[labels == c] - self.means[i] for i, c in enumerate(self.classes)], axis=0)
        cov = np.cov(resid, rowvar=False) + eps * np.eye(f.shape[1])
        self.prec = np.linalg.inv(cov)

    def score(self, logits: np.ndarray) -> np.ndarray:
        f = logits.astype(np.float64)
        d = np.stack([np.einsum("ij,jk,ik->i", f - mu, self.prec, f - mu) for mu in self.means], axis=0)
        return -d.min(axis=0)
