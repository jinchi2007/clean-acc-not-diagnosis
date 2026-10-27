"""FAB -- Frequency-Aligned Baseline (contribution C3 of the paper).

A near-zero-parameter, training-free (well: prototype-only) baseline for bearing
fault diagnosis that is meant to survive cross-machine transfer:

  1. estimate the shaft frequency f_r from the raw amplitude spectrum
  2. select the frequency band with maximal spectral kurtosis (simplified
     fast-kurtogram, 6 log-spaced bands)
  3. compute the squared envelope spectrum (SES) of that band
  4. read SES energy at the bearing characteristic orders
     (BPFO 3.5848, BPFI 5.4152, BSF 4.7135, FTF 0.3983) x {1,2,3} f_r,
     normalised by the SES energy below 400 Hz
  5. classify by nearest class prototype in standardised feature space
  6. reject samples whose distance-to-prototype exceeds a threshold

Nothing is learned except class prototypes and per-feature mean/std, so the
model has ~0 parameters and transfers without fine-tuning.
"""
from __future__ import annotations

import numpy as np

FS_DEFAULT = 12000.0
ORDERS = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 4.7135, "FTF": 0.3983}


# ---------------------------------------------------------------- primitives
def _kurtosis(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    m = s.mean()
    v = s.var()
    if v < 1e-20:
        return 0.0
    return float(((s - m) ** 4).mean() / (v ** 2) - 3.0)


def _bandpass(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mask = ((f >= lo) & (f <= hi)).astype(np.float64)
    return np.fft.irfft(X * mask, n=n, axis=-1)


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    h = np.ones(X.shape[-1], dtype=np.float64)
    if X.shape[-1] > 2:
        h[1:-1] = 2.0
    return np.abs(np.fft.irfft(X * h, n=n, axis=-1))


def _ses(env: np.ndarray, fs: float, nfft: int = 2048):
    n = env.shape[-1]
    e = env - env.mean(axis=-1, keepdims=True)
    w = np.hanning(n)
    S = np.abs(np.fft.rfft(e * w, n=nfft, axis=-1))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return f, S


def estimate_shaft_freq(x: np.ndarray, fs: float = FS_DEFAULT, lo: float = 15.0, hi: float = 70.0):
    """Peak of the raw amplitude spectrum in [lo, hi] Hz -- for CWRU this is 1x RPM."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    X = np.abs(np.fft.rfft(x, axis=-1))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= lo) & (f <= hi)
    fb = f[band]
    Xb = X[..., band]
    return fb[np.argmax(Xb, axis=-1)]


# ---------------------------------------------------------------- features
def extract_features(x: np.ndarray, fs: float = FS_DEFAULT, n_bands: int = 6) -> np.ndarray:
    """x: (N, L) or (N, 1, L) raw windows -> (N, D) FAB feature matrix."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[:, 0, :]
    N, L = x.shape
    fr = estimate_shaft_freq(x, fs)
    edges = np.geomspace(400.0, fs / 2.0 * 0.9, n_bands + 1)
    n_ord = len(ORDERS) * 3
    feats = np.zeros((N, n_ord + 5), dtype=np.float64)

    for i in range(N):
        best_sk, best_sig = -1.0, None
        for b in range(n_bands):
            s = _bandpass(x[i], edges[b], edges[b + 1], fs)
            sk = _kurtosis(s)
            if sk > best_sk:
                best_sk, best_sig = sk, s
        env = _analytic_envelope(best_sig)
        f, S = _ses(env, fs)
        df = f[1] - f[0]
        tot = float(S[f <= 400.0].sum()) + 1e-12

        row = []
        for o in ORDERS.values():
            for k in (1, 2, 3):
                t = o * fr[i] * k
                if t >= f[-1]:
                    row.append(0.0)
                    continue
                m = (f >= t - 2 * df) & (f <= t + 2 * df)
                row.append(float(S[m].sum()) / tot)
        rms = float(np.sqrt((x[i] ** 2).mean()) + 1e-12)
        feats[i, :n_ord] = row
        feats[i, n_ord:] = [best_sk, fr[i], rms, _kurtosis(x[i]), float(np.abs(x[i]).max())]
    return feats


# ---------------------------------------------------------------- classifier
class FAB:
    """Nearest-prototype classifier in standardised FAB feature space."""

    def __init__(self, reject_quantile: float = 0.95, metric: str = "euclidean"):
        self.reject_quantile = reject_quantile
        self.metric = metric
        self.classes_: np.ndarray | None = None
        self.protos_: np.ndarray | None = None
        self.threshold_: float | None = None

    def _z(self, F: np.ndarray) -> np.ndarray:
        return (F - self.mu_) / self.sd_

    def fit(self, F: np.ndarray, y: np.ndarray) -> "FAB":
        F = np.asarray(F, dtype=np.float64)
        y = np.asarray(y)
        self.mu_ = F.mean(axis=0)
        self.sd_ = F.std(axis=0) + 1e-8
        Z = self._z(F)
        self.classes_ = np.unique(y)
        self.protos_ = np.stack([Z[y == c].mean(axis=0) for c in self.classes_])
        d = np.linalg.norm(Z - self.protos_[np.searchsorted(self.classes_, y)], axis=1)
        self.threshold_ = float(np.quantile(d, self.reject_quantile)) if len(d) else 1.0
        return self

    def distances(self, F: np.ndarray) -> np.ndarray:
        Z = self._z(np.asarray(F, dtype=np.float64))
        return np.linalg.norm(Z[:, None, :] - self.protos_[None, :, :], axis=2)

    def predict(self, F: np.ndarray) -> np.ndarray:
        D = self.distances(F)
        return self.classes_[np.argmin(D, axis=1)]

    def predict_open(self, F: np.ndarray, accept_mult: float = 1.0):
        """Returns (pred, accepted_mask, min_dist). Unknown classes are rejected."""
        D = self.distances(F)
        idx = np.argmin(D, axis=1)
        dmin = D[np.arange(len(D)), idx]
        accepted = dmin <= self.threshold_ * accept_mult
        return self.classes_[idx], accepted, dmin


def run_fab(split: str = "safe", overlap: float = 0.5, task: str = "10class",
            seed: int = 0, win: int = 1024, fs: float = FS_DEFAULT) -> dict:
    """End-to-end FAB evaluation on CWRU, matching train_one()'s protocol."""
    from code.data.cwru import build_cwru

    ds = build_cwru(split=split, overlap=overlap, win=win, seed=seed, task=task)
    F_tr = extract_features(ds["x_train"], fs=fs)
    F_te = extract_features(ds["x_test"], fs=fs)
    clf = FAB().fit(F_tr, ds["y_train"])
    pred = clf.predict(F_te)
    acc = float((pred == ds["y_test"]).mean())
    return {
        "split": split,
        "overlap": overlap,
        "task": task,
        "seed": seed,
        "test_acc": acc,
        "n_params": 0,
        "n_train": int(len(F_tr)),
        "n_test": int(len(F_te)),
        "classes": list(map(str, ds["meta"]["classes"])),
    }


if __name__ == "__main__":
    r = run_fab()
    print(r)
