"""Single-run training / evaluation entry point.

    python -m code.train --model wdcnn --split leaky --overlap 0.5 --epochs 20

Every experiment in this repo goes through train_one(cfg) so that the protocol
(seed, split, epochs, eval) is identical across E0-E7.

Extra knobs used by the harder regimes:
  n_per_class      : few-shot -- sample n training windows per class
  holdout_classes  : open-set -- train on the remaining classes only, then
                     measure known-class accuracy and unknown-class rejection
  noise_snrs       : evaluate the same trained model at several SNRs
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from code.common import (
    RESULTS_DIR,
    add_noise,
    count_params,
    get_device,
    save_json,
    set_seed,
)
from code.data.cwru import build_cwru
from code.models import build_model
from code.transforms import apply_input

DEFAULT_CFG = {
    "dataset": "cwru",
    "task": "10class",
    "split": "safe",
    "overlap": 0.5,
    "win": 1024,
    "seed": 0,
    "model": "wdcnn",
    "epochs": 20,
    "batch": 128,
    "lr": 2e-3,
    "wd": 1e-4,
    "device": "mps",
    "input": "raw",
    "n_per_class": None,
    "holdout_classes": None,
    "noise_snr": None,
    "noise_snrs": None,
    "stem": None,
    "train_noise_lo": None,
    "train_noise_hi": None,
    "noise_kind": "gaussian",
    "out": None,
}


# ------------------------------------------------------------------ utilities
def subsample_per_class(x: np.ndarray, y: np.ndarray, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        keep.append(idx)
    idx = np.sort(np.concatenate(keep))
    return x[idx], y[idx]


@torch.no_grad()
def predict_probs(model: nn.Module, x: np.ndarray, device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch])).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg); neg = should-be-rejected, pos = should-be-accepted."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    all_s = np.concatenate([neg, pos])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(len(all_s), dtype=np.float64)
    ranks[order] = np.arange(1, len(all_s) + 1, dtype=np.float64)
    n0, n1 = len(neg), len(pos)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch])).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


@torch.no_grad()
def predict_logits_and_features(model: nn.Module, x: np.ndarray, device, batch: int = 512):
    """Return (logits, penultimate features) for x.

    The penultimate layer is the input of the last nn.Linear module, captured
    with a pre-hook so that no model-specific code is needed.
    """
    last_linear = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("no nn.Linear layer found; cannot extract features")
    feats = []

    def _hook(_module, inputs):
        feats.append(inputs[0].detach().reshape(inputs[0].shape[0], -1).cpu().numpy())

    handle = last_linear.register_forward_pre_hook(_hook)
    model.eval()
    logits = []
    try:
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch])).to(device)
            logits.append(model(xb).cpu().numpy())
    finally:
        handle.remove()
    if not logits:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(logits, axis=0), np.concatenate(feats, axis=0)


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device, batch: int = 512) -> float:
    if len(y) == 0:
        return float("nan")
    probs = predict_probs(model, x, device, batch)
    return float((probs.argmax(1) == y).mean())


# ------------------------------------------------------------------ main entry
def _make_model(cfg, num_classes, in_len):
    """可选宽核前端 stem=(k, s)：把 (B,1,L) 先做可学习带限下采样再送 backbone。"""
    stem = cfg.get("stem")
    if not stem:
        return build_model(cfg["model"], num_classes=num_classes, in_len=in_len)
    if len(stem) == 3:
        from code.models import build_stem_model
        return build_stem_model(cfg["model"], num_classes=num_classes, in_len=in_len,
                                k=int(stem[0]), s=int(stem[1]), c=int(stem[2]))
    k, s = int(stem[0]), int(stem[1])
    front = nn.Sequential(nn.Conv1d(1, 1, k, s), nn.BatchNorm1d(1), nn.ReLU(inplace=True))
    with torch.no_grad():
        n_out = int(front(torch.zeros(1, 1, in_len)).shape[-1])
    base = build_model(cfg["model"], num_classes=num_classes, in_len=n_out)

    class _Stem(nn.Module):
        def __init__(self, f, b):
            super().__init__()
            self.front, self.base = f, b

        def forward(self, x):
            return self.base(self.front(x))

    return _Stem(front, base)


def train_one(cfg: dict) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    t_start = time.time()

    ds = build_cwru(
        split=cfg["split"],
        overlap=cfg["overlap"],
        win=cfg["win"],
        seed=cfg["seed"],
        task=cfg["task"],
    )
    x_train, y_train = ds["x_train"], ds["y_train"]
    x_test, y_test = ds["x_test"], ds["y_test"]
    in_len_raw = ds["x_train"].shape[-1]
    if cfg.get("input", "raw") != "raw":
        x_train = apply_input(x_train, cfg["input"])
        x_test = apply_input(x_test, cfg["input"])
    in_len = x_train.shape[-1]

    if cfg.get("n_per_class"):
        x_train, y_train = subsample_per_class(x_train, y_train, int(cfg["n_per_class"]), cfg["seed"])

    # ---- open-set: restrict training to known classes, keep the rest as unknown
    holdout = cfg.get("holdout_classes")
    known = None
    if holdout:
        holdout = [int(c) for c in holdout]
        known = np.array([c for c in np.unique(y_train) if c not in holdout], dtype=np.int64)
        keep = np.isin(y_train, known)
        x_train, y_train = x_train[keep], y_train[keep]
        y_train = np.searchsorted(known, y_train).astype(np.int64)
        known_test_mask = np.isin(y_test, known)
        x_known, y_known = x_test[known_test_mask], np.searchsorted(known, y_test[known_test_mask])
        x_unknown = x_test[~known_test_mask]
    else:
        x_known, y_known = x_test, y_test
        x_unknown = None

    num_classes = len(known) if known is not None else len(ds["meta"]["classes"])
    model = _make_model(cfg, num_classes, in_len).to(device)
    n_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["epochs"], 1))
    lossf = nn.CrossEntropyLoss()

    xt = torch.from_numpy(np.ascontiguousarray(x_train))
    yt = torch.from_numpy(np.ascontiguousarray(y_train))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=cfg["batch"], shuffle=True, drop_last=False)

    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        loss_sum, correct = 0.0, 0
        for bi, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            if cfg.get("train_noise_lo") is not None:
                lo, hi = float(cfg["train_noise_lo"]), float(cfg["train_noise_hi"])
                # 逐 batch 确定性 RNG：SNR ~ U[lo, hi] 语义不变，但同一
                # (cfg.seed, epoch, batch_idx) 抽到的 SNR 与噪声实现完全一致，
                # 让同一 cfg 重跑逐 batch 可复现（此前用未播种的 default_rng，会抖动）。
                rng = np.random.default_rng(int(cfg["seed"]) * 100003 + ep * 1009 + bi)
                snr = float(rng.uniform(lo, hi))
                xb = torch.from_numpy(
                    add_noise(xb.detach().cpu().numpy(), snr, cfg.get("noise_kind", "gaussian"),
                              seed=int(rng.integers(0, 2**31 - 1)))
                ).to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((out.argmax(dim=1) == yb).sum().item())
        sched.step()
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": loss_sum / max(len(xt), 1),
                "train_acc": correct / max(len(xt), 1),
                "test_acc": evaluate(model, x_known, y_known, device),
            }
        )

    result = {
        "cfg": cfg,
        "data_meta": {k: v for k, v in ds["meta"].items() if k not in ("train_groups", "test_groups")},
        "n_params": n_params,
        "n_train": int(len(x_train)),
        "test_acc": history[-1]["test_acc"],
        "best_test_acc": max(h["test_acc"] for h in history),
        "train_time_s": round(time.time() - t_start, 2),
        "history": history,
    }

    # ---- open-set metrics
    if x_unknown is not None:
        probs_k = predict_probs(model, x_known, device)
        probs_u = predict_probs(model, x_unknown, device)
        conf_k, conf_u = probs_k.max(1), probs_u.max(1)
        thr = float(np.quantile(conf_k, 0.05))
        thr_u95 = float(np.quantile(conf_u, 0.95))
        result["openset"] = {
            "holdout_classes": [int(c) for c in holdout],
            "n_known_test": int(len(x_known)),
            "n_unknown_test": int(len(x_unknown)),
            "auroc": auroc(conf_u, conf_k),
            "tpr_at_95fpr": float((conf_k >= thr_u95).mean()),
            "acc_at_reject_5pct": float(((probs_k.argmax(1) == y_known) & (conf_k >= thr)).mean()),
            "known_conf_mean": float(conf_k.mean()),
            "unknown_conf_mean": float(conf_u.mean()),
            "threshold_q05": thr,
        }

        # ---- decision-layer comparison (E10): same network, three scores
        if cfg.get("score_eval"):
            from code.scores import LogitMahalanobis, energy, softmax_max

            lg_tr = predict_logits(model, x_train, device)
            lg_k = predict_logits(model, x_known, device)
            lg_u = predict_logits(model, x_unknown, device)
            mah = LogitMahalanobis(lg_tr, y_train)
            cand = {
                "softmax": (softmax_max(lg_k), softmax_max(lg_u)),
                "energy": (energy(lg_k), energy(lg_u)),
                "mahalanobis": (mah.score(lg_k), mah.score(lg_u)),
            }
            scores_res = {}
            for name, (sk, su) in cand.items():
                q = float(np.quantile(su, 0.95))
                scores_res[name] = {
                    "auroc": auroc(su, sk),
                    "tpr_at_95fpr": float((sk >= q).mean()),
                }
            # ---- penultimate-feature variant (E11): same estimator, features
            if cfg.get("feature_scores"):
                from code.scores import FeatureMahalanobis

                _, ft_tr = predict_logits_and_features(model, x_train, device)
                _, ft_k = predict_logits_and_features(model, x_known, device)
                _, ft_u = predict_logits_and_features(model, x_unknown, device)
                fm = FeatureMahalanobis(ft_tr, y_train)
                sk, su = fm.score(ft_k), fm.score(ft_u)
                qf = float(np.quantile(su, 0.95))
                scores_res["mahalanobis_feat"] = {
                    "auroc": auroc(su, sk),
                    "tpr_at_95fpr": float((sk >= qf).mean()),
                    "dim": int(ft_tr.shape[1]),
                }
            result["openset_scores"] = scores_res

    # ---- noise sweep
    snrs = cfg.get("noise_snrs")
    if cfg.get("noise_snr") is not None:
        snrs = [float(cfg["noise_snr"])]
    if snrs:
        noise_res = {}
        for s in snrs:
            xn = add_noise(x_known, float(s), cfg.get("noise_kind", "gaussian"), seed=cfg["seed"])
            if cfg.get("input", "raw") != "raw":
                xn = apply_input(xn, cfg["input"])
            noise_res[str(s)] = evaluate(model, xn, y_known, device)
        result["noise"] = {"kind": cfg.get("noise_kind", "gaussian"), "acc_by_snr": noise_res}
        if len(snrs) == 1:
            result["test_acc_noise"] = noise_res[str(float(snrs[0]))]

    if cfg.get("out"):
        save_json(cfg["out"], result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wdcnn")
    ap.add_argument("--split", default="safe", choices=["leaky", "safe", "crossload", "filedisjoint"])
    ap.add_argument("--task", default="10class", choices=["10class", "4class"])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--holdout-classes", default=None, help="comma separated class ids")
    ap.add_argument("--noise-snr", type=float, default=None)
    ap.add_argument("--noise-kind", default="gaussian")
    ap.add_argument("--input", default="raw", choices=["raw", "ses"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = vars(args)
    if cfg["holdout_classes"]:
        cfg["holdout_classes"] = [int(v) for v in cfg["holdout_classes"].split(",")]
    if cfg["out"] is None:
        cfg["out"] = os.path.join(
            RESULTS_DIR,
            f"run_{cfg['model']}_{cfg['split']}_ov{cfg['overlap']}_s{cfg['seed']}.json",
        )
    r = train_one(cfg)
    print(
        f"{cfg['model']:12s} split={cfg['split']:9s} ov={cfg['overlap']:.2f} seed={cfg['seed']} "
        f"acc={r['test_acc']:.4f} params={r['n_params']} time={r['train_time_s']}s"
    )


if __name__ == "__main__":
    main()
