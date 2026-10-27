"""Ablation: learned wide-kernel front-end (k, s, c) + backbone.

Protocol (identical to code/train.py, so the numbers are comparable):
CWRU 10-class, split='safe', overlap=0.5, win=1024, AdamW lr=2e-3 wd=1e-4,
cosine schedule, batch 128, 15 epochs, seed 0, clean training (no noise
augmentation), then evaluation on the clean test split and on Gaussian-noise
test copies at SNR 10 / 5 / 0 dB (code.common.add_noise, same seed).

The front-end is Conv1d(1, c, k, stride=s) -> BatchNorm1d(c) -> ReLU followed
by the backbone built for the shortened window and in_ch=c; c=0 means "no
front-end" (plain backbone, the control row).

Usage::

    python -m code.models.stem_ablation --part 0 --nparts 2   # two workers
    python -m code.models.stem_ablation --part 1 --nparts 2
    python -m code.models.stem_ablation --merge               # json + md
    python -m code.models.stem_ablation --only resnet1d_k64_c8  # one config

Each worker writes results/stem_ablation_part{part}.json; --merge combines
the parts into results/stem_ablation.json and results/stem_ablation.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from code.common import add_noise, count_params, get_device, save_json, set_seed
from code.data.cwru import build_cwru

from . import build_model, build_stem_model

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "stem_ablation.json"
OUT_MD = RESULTS / "stem_ablation.md"

#: (label, backbone, k, s, c) -- c=0 is the plain backbone control row.
CONFIGS = [
    ("resnet1d_base", "resnet1d", 0, 0, 0),
    ("ds_cnn_base", "ds_cnn", 0, 0, 0),
    ("vit1d_base", "vit1d", 0, 0, 0),
    ("mcnn_base", "mcnn", 0, 0, 0),
    ("resnet1d_k64_c1", "resnet1d", 64, 16, 1),
    ("resnet1d_k64_c4", "resnet1d", 64, 16, 4),
    ("resnet1d_k64_c8", "resnet1d", 64, 16, 8),
    ("resnet1d_k32_c8", "resnet1d", 32, 16, 8),
    ("resnet1d_k128_c8", "resnet1d", 128, 16, 8),
    ("ds_cnn_k64_c4", "ds_cnn", 64, 16, 4),
    ("ds_cnn_k64_c8", "ds_cnn", 64, 16, 8),
    ("vit1d_k64_c8", "vit1d", 64, 16, 8),
    ("mcnn_k64_c8", "mcnn", 64, 16, 8),
]
CONFIG_NAMES = [c[0] for c in CONFIGS]
SNRS = (10.0, 5.0, 0.0)

BASE_CFG = dict(split="safe", overlap=0.5, win=1024, task="10class",
                epochs=15, batch=128, lr=2e-3, wd=1e-4, seed=0)


def make_model(spec, num_classes, in_len):
    _, backbone, k, s, c = spec
    if c == 0:
        return build_model(backbone, num_classes=num_classes, in_len=in_len)
    return build_stem_model(backbone, num_classes=num_classes, in_len=in_len,
                            k=k, s=s, c=c)


@torch.no_grad()
def evaluate(model, x, y, device, batch=512):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / max(len(y), 1)


def train_one(spec, cfg):
    set_seed(cfg["seed"])
    device = get_device("mps")
    t0 = time.time()

    ds = build_cwru(split=cfg["split"], overlap=cfg["overlap"], win=cfg["win"],
                    seed=cfg["seed"], task=cfg["task"])
    x_train, y_train = ds["x_train"], ds["y_train"]
    x_test, y_test = ds["x_test"], ds["y_test"]
    in_len = x_train.shape[-1]
    num_classes = len(ds["meta"]["classes"])

    model = make_model(spec, num_classes, in_len).to(device)
    n_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg["epochs"], 1))
    lossf = nn.CrossEntropyLoss()

    xt = torch.from_numpy(np.ascontiguousarray(x_train))
    yt = torch.from_numpy(np.ascontiguousarray(y_train))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=cfg["batch"],
                        shuffle=True)
    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((out.argmax(1) == yb).sum().item())
        sched.step()
        history.append({"epoch": ep + 1,
                        "train_loss": loss_sum / len(xt),
                        "train_acc": correct / len(xt)})

    result = {
        "label": spec[0],
        "backbone": spec[1],
        "k": spec[2],
        "s": spec[3],
        "c": spec[4],
        "n_params": n_params,
        "clean": evaluate(model, x_test, y_test, device),
        "by_snr": {},
        "n_train": int(len(x_train)),
        "n_test": int(len(y_test)),
        "train_time_s": round(time.time() - t0, 2),
        "final_train_acc": history[-1]["train_acc"],
        "epochs": cfg["epochs"],
        "seed": cfg["seed"],
    }
    for snr in SNRS:
        xn = add_noise(x_test, snr, "gaussian", seed=cfg["seed"])
        result["by_snr"][str(snr)] = evaluate(model, xn, y_test, device)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def run_part(part, nparts, cfg):
    mine = [s for i, s in enumerate(CONFIGS) if i % nparts == part]
    out = []
    for spec in mine:
        t0 = time.time()
        r = train_one(spec, cfg)
        out.append(r)
        print(f"[part {part}] {r['label']:<18s} params={r['n_params']:>8,d} "
              f"clean={r['clean']:.4f} 10dB={r['by_snr']['10.0']:.4f} "
              f"5dB={r['by_snr']['5.0']:.4f} 0dB={r['by_snr']['0.0']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        save_json(RESULTS / f"stem_ablation_part{part}.json",
                  {"configs": CONFIG_NAMES, "rows": out})
    save_json(RESULTS / f"stem_ablation_part{part}.json",
              {"configs": CONFIG_NAMES, "rows": out})
    return out


def merge():
    rows = []
    for p in range(4):
        f = RESULTS / f"stem_ablation_part{p}.json"
        if f.exists():
            rows += json.loads(f.read_text())["rows"]
    order = {name: i for i, name in enumerate(CONFIG_NAMES)}
    rows.sort(key=lambda r: order.get(r["label"], 999))
    if not rows:
        raise SystemExit("no part files found; run --part 0/1 first")

    json_out = {
        "experiment": "stem_ablation",
        "protocol": {**BASE_CFG, "snrs": list(SNRS),
                     "train_noise": None, "note": "clean training, noise only at test"},
        "configs": CONFIG_NAMES,
        "rows": rows,
    }
    save_json(OUT_JSON, json_out)

    by_label = {r["label"]: r for r in rows}

    def base_of(backbone):
        return by_label.get(backbone + "_base")

    lines = []
    lines.append("# Wide-kernel front-end ablation (clean training, noise test)")
    lines.append("")
    lines.append(f"Protocol: CWRU 10-class, split=safe, overlap=0.5, win=1024, "
                 f"AdamW lr={BASE_CFG['lr']} wd={BASE_CFG['wd']}, "
                 f"{BASE_CFG['epochs']} epochs, seed={BASE_CFG['seed']}, "
                 f"batch={BASE_CFG['batch']}, clean training (no augmentation), "
                 f"Gaussian noise at test time.")
    lines.append("")
    lines.append("| config | backbone | stem (k,s,c) | params | clean | 10 dB | 5 dB | 0 dB | "
                 "d clean | d 10 dB |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        stem = "none" if r["c"] == 0 else f"({r['k']},{r['s']},{r['c']})"
        b = base_of(r["backbone"])
        if b is None:
            dc = d10 = float("nan")
        else:
            dc = (r["clean"] - b["clean"]) * 100
            d10 = (r["by_snr"]["10.0"] - b["by_snr"]["10.0"]) * 100
        lines.append(
            f"| {r['label']} | {r['backbone']} | {stem} | {r['n_params']:,} | "
            f"{r['clean']:.4f} | {r['by_snr']['10.0']:.4f} | "
            f"{r['by_snr']['5.0']:.4f} | {r['by_snr']['0.0']:.4f} | "
            f"{dc:+.2f} | {d10:+.2f} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print("\n".join(lines))
    return json_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--nparts", type=int, default=2)
    ap.add_argument("--only", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--epochs", type=int, default=BASE_CFG["epochs"])
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = dict(BASE_CFG, epochs=args.epochs)
    if args.merge:
        merge()
    elif args.only:
        spec = next(s for s in CONFIGS if s[0] == args.only)
        r = train_one(spec, cfg)
        save_json(RESULTS / f"stem_ablation_{args.only}.json", {"rows": [r]})
        print(json.dumps(r, indent=1))
    else:
        run_part(args.part or 0, args.nparts, cfg)


if __name__ == "__main__":
    main()
