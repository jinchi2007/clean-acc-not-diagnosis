"""E8 -- Paderborn generalization (2-class healthy vs outer ring).

Question: do the CWRU noise-stress conclusions (E2/E4) transfer to a second,
harder rig (64 kHz, 4 operating conditions, 1439 published records)?

Protocol
--------
The 2.7 GB Paderborn cache is read once, stratified-subsampled to
train=30k/class, test=10k/class (seed fixed) and stored as a small npz
(data/processed/paderborn_small_safe.npz); every run after that only touches
the small cache.  Training reuses code.train.train_one unchanged (the CWRU
loader is monkeypatched to return the small Paderborn tensors), so the
protocol (AdamW lr 2e-3, batch 128, 15 epochs, cosine schedule, per-window
z-scored inputs) is identical to E0-E7.

Matrix: models = [wdcnn, ticnn, resnet1d, ds_cnn]
        x regime = {plain, aug (SNR ~ U[0, 15] per batch)}
        x seeds  = {0, 1}
        epochs   = 15, split = safe
Evaluation: clean / 10 dB / 5 dB / 0 dB additive white Gaussian noise.

    python -m experiments.e8_paderborn --seeds 0,1 --epochs 15
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from code.common import RESULTS_DIR, save_json

MODELS = ["wdcnn", "ticnn", "resnet1d", "ds_cnn"]
REGIMES = ["plain", "aug"]
SNRS = [10.0, 5.0, 0.0]
RUNS = Path(RESULTS_DIR) / "runs" / "e8_paderborn"
PROC = Path("data") / "processed"
BIG = PROC / "paderborn_2class_safe_ov0.5_1024.npz"
SMALL = PROC / "paderborn_small_safe.npz"
TRAIN_CAP = 30000
TEST_CAP = 10000


def _subsample_idx(y: np.ndarray, cap: int, rng) -> np.ndarray:
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def load_small(seed: int = 0, rebuild: bool = False) -> dict:
    """Return the small Paderborn dataset (cached in data/processed/)."""
    if SMALL.exists() and not rebuild:
        with np.load(SMALL, allow_pickle=False) as z:
            data = {k: z[k] for k in ("x_train", "y_train", "x_test", "y_test")}
            meta = json.loads(str(z["meta"]))
        print(f"[e8] small cache hit {SMALL.name}: "
              f"train={len(data['y_train'])} test={len(data['y_test'])}", flush=True)
        return data | {"meta": meta}

    print(f"[e8] building small cache from {BIG.name} ...", flush=True)
    t0 = time.time()
    with np.load(BIG, allow_pickle=False) as z:
        big_meta = json.loads(str(z["meta"]))
        y_train, y_test = z["y_train"], z["y_test"]
        rng = np.random.default_rng([seed, 0xE8])
        itr = _subsample_idx(y_train, TRAIN_CAP, rng)
        ite = _subsample_idx(y_test, TEST_CAP, rng)
        data = {
            "x_train": z["x_train"][itr],
            "y_train": y_train[itr],
            "x_test": z["x_test"][ite],
            "y_test": y_test[ite],
        }
    meta = {
        "dataset": "paderborn",
        "split": "safe",
        "source": BIG.name,
        "classes": big_meta["classes"],
        "train_cap_per_class": TRAIN_CAP,
        "test_cap_per_class": TEST_CAP,
        "subsample_seed": seed,
        "n_train": int(len(data["y_train"])),
        "n_test": int(len(data["y_test"])),
        "fs": big_meta.get("fs"),
        "win": big_meta.get("win"),
    }
    data["meta"] = meta
    payload = dict(data)
    payload["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(SMALL, **payload)
    print(f"[e8] wrote {SMALL} ({SMALL.stat().st_size / 1e6:.0f} MB) in "
          f"{time.time() - t0:.1f}s", flush=True)
    return data


def _as_dataset(data: dict) -> dict:
    """Shape expected by code.train.train_one (same keys as build_cwru)."""
    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
        "meta": {
            "dataset": "paderborn",
            "split": "safe",
            "task": "2class",
            "classes": data["meta"]["classes"],
            "win": data["meta"].get("win", 1024),
            "overlap": 0.5,
        },
    }


def run_id(cfg: dict) -> str:
    parts = [cfg["model"], cfg["split"], f"seed{cfg['seed']}", f"ep{cfg['epochs']}"]
    if cfg.get("train_noise_lo") is not None:
        parts.append("aug%g-%g" % (cfg["train_noise_lo"], cfg["train_noise_hi"]))
    else:
        parts.append("plain")
    return "_".join(parts)


def _runs(seeds, epochs):
    for model, regime, seed in itertools.product(MODELS, REGIMES, seeds):
        cfg = {
            "dataset": "paderborn",
            "split": "safe",
            "overlap": 0.5,
            "win": 1024,
            "seed": seed,
            "model": model,
            "epochs": epochs,
            "batch": 128,
            "lr": 2e-3,
            "wd": 1e-4,
            "device": "mps",
            "noise_snrs": list(SNRS),
        }
        if regime == "aug":
            cfg.update(train_noise_lo=0.0, train_noise_hi=15.0)
        yield cfg


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def train_one_paderborn(cfg: dict, ds: dict) -> dict:
    """Same protocol as code.train.train_one, but fed the small Paderborn tensors.

    The only deviation is evaluation cadence: during training the per-epoch
    monitor accuracy is measured on a fixed stratified 2k/class subset, while
    the reported clean accuracy and all noise accuracies are measured on the
    full test set after training.  This saves ~5 min of per-epoch eval on the
    20k-window test set without touching the optimisation protocol.
    """
    import time

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import code.train as T
    from code.common import add_noise, count_params, get_device, set_seed
    from code.models import build_model

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "mps"))
    t_start = time.time()

    x_train, y_train = ds["x_train"], ds["y_train"]
    x_test, y_test = ds["x_test"], ds["y_test"]
    in_len = x_train.shape[-1]

    num_classes = len(ds["meta"]["classes"])
    model = build_model(cfg["model"], num_classes=num_classes, in_len=in_len).to(device)
    n_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["epochs"], 1))
    lossf = nn.CrossEntropyLoss()

    xt = torch.from_numpy(np.ascontiguousarray(x_train))
    yt = torch.from_numpy(np.ascontiguousarray(y_train))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=cfg["batch"], shuffle=True, drop_last=False)

    # fixed stratified monitor subset (2k per class) for the epoch curve
    rng = np.random.default_rng([cfg["seed"], 0xE8])
    midx = _subsample_idx(y_test, 2000, rng)
    x_mon, y_mon = x_test[midx], y_test[midx]

    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg.get("train_noise_lo") is not None:
                lo, hi = float(cfg["train_noise_lo"]), float(cfg["train_noise_hi"])
                snr = float(np.random.default_rng().uniform(lo, hi))
                xb = torch.from_numpy(
                    add_noise(xb.detach().cpu().numpy(), snr, cfg.get("noise_kind", "gaussian"),
                              seed=int(np.random.randint(0, 2**31 - 1)))
                ).to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((out.argmax(dim=1) == yb).sum().item())
        sched.step()
        history.append({"epoch": ep + 1, "train_loss": loss_sum / max(len(xt), 1),
                        "train_acc": correct / max(len(xt), 1),
                        "monitor_test_acc": T.evaluate(model, x_mon, y_mon, device)})

    clean = T.evaluate(model, x_test, y_test, device)
    noise_res = {}
    for s in cfg["noise_snrs"]:
        xn = add_noise(x_test, float(s), cfg.get("noise_kind", "gaussian"), seed=cfg["seed"])
        noise_res[str(s)] = T.evaluate(model, xn, y_test, device)

    return {
        "cfg": cfg,
        "n_params": n_params,
        "n_train": int(len(x_train)),
        "test_acc": clean,
        "best_test_acc": clean,
        "train_time_s": round(time.time() - t_start, 2),
        "history": history,
        "noise": {"kind": cfg.get("noise_kind", "gaussian"), "acc_by_snr": noise_res},
    }


def run_grid(seeds, epochs, data) -> list[dict]:
    """Run every (model, regime, seed) cell; resume from per-run JSON files."""
    ds = _as_dataset(data)
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = list(_runs(seeds, epochs))
    print(f"[e8] {len(todo)} runs", flush=True)
    results = []
    for i, cfg in enumerate(todo, 1):
        rid = run_id(cfg)
        path = RUNS / (rid + ".json")
        if path.exists():
            results.append(json.loads(path.read_text()))
            print(f"[{i}/{len(todo)}] cache hit {rid}", flush=True)
            continue
        t0 = time.time()
        res = train_one_paderborn(cfg, ds)
        res["cfg"] = cfg
        res["run_id"] = rid
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        results.append(res)
        print(f"[{i}/{len(todo)}] {rid} clean={res['test_acc']:.4f} "
              f"10dB={res['noise']['acc_by_snr']['10.0']:.4f} "
              f"5dB={res['noise']['acc_by_snr']['5.0']:.4f} "
              f"0dB={res['noise']['acc_by_snr']['0.0']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return results


def aggregate(results: list[dict], seeds) -> dict:
    table = {}
    for model, regime in itertools.product(MODELS, REGIMES):
        sub = [r for r in results
               if r["cfg"]["model"] == model
               and ("aug" if r["cfg"].get("train_noise_lo") is not None else "plain") == regime]
        if not sub:
            continue
        row = {"n_runs": len(sub), "clean": _mean_std([r["test_acc"] for r in sub])}
        for s in SNRS:
            row[f"snr{int(s)}"] = _mean_std([r["noise"]["acc_by_snr"][str(s)] for r in sub])
        row["params"] = sub[0]["n_params"]
        row["train_time_s"] = float(np.mean([r["train_time_s"] for r in sub]))
        table[f"{model}:{regime}"] = row

    # plain vs aug deltas (mean over seeds), per model and per SNR
    deltas = {}
    for model in MODELS:
        p = table.get(f"{model}:plain")
        a = table.get(f"{model}:aug")
        if not p or not a:
            continue
        d = {"clean": a["clean"][0] - p["clean"][0]}
        for s in SNRS:
            d[f"snr{int(s)}"] = a[f"snr{int(s)}"][0] - p[f"snr{int(s)}"][0]
        deltas[model] = d

    # rank orderings on clean vs 10 dB, and whether they invert
    rank = {}
    for key in ("plain", "aug"):
        cl = sorted([m for m in MODELS if f"{m}:{key}" in table],
                    key=lambda m: -table[f"{m}:{key}"]["clean"][0])
        n10 = sorted([m for m in MODELS if f"{m}:{key}" in table],
                     key=lambda m: -table[f"{m}:{key}"]["snr10"][0])
        rank[key] = {"clean_order": cl, "10dB_order": n10,
                     "inverted": cl != n10,
                     "clean_best": cl[0], "noisy_best": n10[0]}

    # raw per-seed values (required for the write-up: mean±std must be checkable)
    per_seed = {}
    for model, regime in itertools.product(MODELS, REGIMES):
        sub = [r for r in results
               if r["cfg"]["model"] == model
               and ("aug" if r["cfg"].get("train_noise_lo") is not None else "plain") == regime]
        if not sub:
            continue
        rows = []
        for r in sorted(sub, key=lambda r: r["cfg"]["seed"]):
            row = {"seed": r["cfg"]["seed"], "clean": r["test_acc"]}
            for s in SNRS:
                row[f"snr{int(s)}"] = r["noise"]["acc_by_snr"][str(s)]
            row["train_time_s"] = round(float(r["train_time_s"]), 1)
            rows.append(row)
        per_seed[f"{model}:{regime}"] = rows

    return {"experiment": "E8_paderborn", "dataset": "paderborn (2-class healthy vs outer)",
            "split": "safe", "epochs": results[0]["cfg"]["epochs"] if results else None,
            "seeds": list(seeds), "snrs": SNRS, "table": table,
            "delta_aug_minus_plain": deltas, "rank": rank,
            "per_seed": per_seed,
            "n_train_small": table and results[0]["n_train"]}


def render_md(agg: dict) -> str:
    lines = ["# E8 -- Paderborn generalization (2-class healthy vs outer ring)",
             "",
             "Paderborn KAt subset (18 bearings: 6 healthy K001-K006 + 12 artificial "
             "outer-ring damage KA*), 64 kHz, 4 operating conditions, leakage-safe "
             "split (per-record 70/30 with 512-sample guard band).",
             "",
             f"train/test = {agg.get('n_train_small')} / 20000 windows "
             f"(30k / 10k per class stratified subsample of the full safe split), "
             f"{agg['epochs']} epochs, seeds {agg['seeds']}, mean±std over seeds.",
             "",
             "| config | params | clean | 10 dB | 5 dB | 0 dB |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, row in agg["table"].items():
        cells = [f"{row['clean'][0]:.4f}±{row['clean'][1]:.3f}"]
        for s in SNRS:
            v = row[f"snr{int(s)}"]
            cells.append(f"{v[0]:.4f}±{v[1]:.3f}")
        lines.append(f"| {key} | {row['params']} | " + " | ".join(cells) + " |")
    lines += ["", "## aug - plain (delta accuracy)", "",
              "| model | clean | 10 dB | 5 dB | 0 dB |",
              "|---|---:|---:|---:|---:|"]
    for model, d in agg["delta_aug_minus_plain"].items():
        lines.append(f"| {model} | {d['clean']:+.4f} | {d['snr10']:+.4f} | "
                     f"{d['snr5']:+.4f} | {d['snr0']:+.4f} |")
    lines += ["", "## ranking (clean vs 10 dB)", ""]
    for regime, r in agg["rank"].items():
        lines.append(f"- {regime}: clean order {r['clean_order']} vs 10 dB order "
                     f"{r['10dB_order']} -> inverted={r['inverted']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rebuild-small", action="store_true")
    ap.add_argument("--out", default=str(Path(RESULTS_DIR) / "e8_paderborn.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    data = load_small(rebuild=args.rebuild_small)
    results = run_grid(seeds, args.epochs, data)
    agg = aggregate(results, seeds)
    save_json(args.out, agg)
    md = render_md(agg)
    Path(RESULTS_DIR, "e8_paderborn.md").write_text(md)
    print("\n" + md, flush=True)
    print(f"wrote {args.out} and {Path(RESULTS_DIR) / 'e8_paderborn.md'}", flush=True)


if __name__ == "__main__":
    main()
