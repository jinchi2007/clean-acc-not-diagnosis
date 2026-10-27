"""E6 噪声类型扩展：谐波 / 脉冲 / 漂移 三类非高斯噪声下的精度。

协议与主网格的 E2（clean training + 测试端加噪）一致：split=safe, win=1024, 20 epoch,
干净训练，测试时用 code.common.add_noise(kind=...) 加对应噪声，SNR=10/0 dB。
默认 4 模型 x 3 类型 x 2 seeds = 24 runs，断点续跑（逐 run 一个 json）。

同时把同协议的高斯噪声结果（results/runs/e2_noise_clean，3 seeds）作为参照列，
用于回答"换噪声类型后模型排序是否变化"。

用法::

    python -m experiments.e6_noise_types --seeds 0,1
    python -m experiments.e6_noise_types --models wdcnn,resnet1d --seeds 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.common import RESULTS_DIR
from code.train import train_one

KINDS = ["harmonic", "impulse", "drift"]
SNRS = [10.0, 0.0]
MODELS = ["wdcnn", "resnet1d", "ds_cnn", "vit1d"]
GAUSS_DIR = Path(RESULTS_DIR) / "runs" / "e2_noise_clean"


def _ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def gaussian_reference(models, seeds):
    """同协议（ep20 / safe / 3 seeds）的干净训练 + 高斯噪声参照。"""
    ref = {}
    for model in models:
        d = {"clean": [], "10.0": [], "0.0": []}
        for seed in seeds:
            path = GAUSS_DIR / f"{model}_safe_seed{seed}_ep20.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            d["clean"].append(r["test_acc"])
            for snr in SNRS:
                d[str(snr)].append(r["noise"]["acc_by_snr"][str(snr)])
        if d["clean"]:
            ref[model] = {k: list(_ms(v)) for k, v in d.items() if v}
    return ref


def render_md(table, ref, models):
    lines = ["# E6: 噪声类型扩展（clean training，测试端加噪）", ""]
    lines.append("协议：CWRU 10-class, split=safe, overlap=0.5, win=1024, 20 epoch, "
                 "干净训练；测试端用 add_noise(kind, SNR) 加谐波 / 脉冲 / 漂移噪声，"
                 "SNR = 10 / 0 dB。数值为 seeds 均值±std（std 在只有 1 个 seed 时记 0）。")
    lines.append("")
    lines.append("## 主表：三类非高斯噪声")
    lines.append("")
    lines.append("| model | noise | clean | 10 dB | 0 dB |")
    lines.append("|---|---|---:|---:|---:|")
    for model in models:
        for kind in KINDS:
            e = table.get(model, {}).get(kind)
            if not e:
                continue
            cells = []
            for key in ("clean", "10.0", "0.0"):
                v = e.get(key)
                cells.append("-" if not v else f"{v[0]:.3f}±{v[1]:.3f}")
            lines.append(f"| {model} | {kind} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 高斯参照（同协议，3 seeds）与对照")
    lines.append("")
    lines.append("| model | gaussian clean | gaussian 10 dB | gaussian 0 dB | "
                 "harmonic 10 dB | impulse 10 dB | drift 10 dB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in models:
        g = ref.get(model)
        e = table.get(model, {})
        g10 = "-" if not g else f"{g['10.0'][0]:.3f}±{g['10.0'][1]:.3f}"
        g0 = "-" if not g else f"{g['0.0'][0]:.3f}±{g['0.0'][1]:.3f}"
        gc = "-" if not g else f"{g['clean'][0]:.3f}"
        def cell(kind):
            v = e.get(kind, {}).get("10.0")
            return "-" if not v else f"{v[0]:.3f}"
        lines.append(f"| {model} | {gc} | {g10} | {g0} | {cell('harmonic')} | "
                     f"{cell('impulse')} | {cell('drift')} |")
    lines.append("")

    # 排序对比
    lines.append("## 10 dB 下的模型排序：高斯 vs 三类非高斯")
    lines.append("")
    orders = {}
    if ref and all("10.0" in ref.get(m, {}) for m in models if ref):
        orders["gaussian"] = sorted([m for m in models if m in ref],
                                    key=lambda m: -ref[m]["10.0"][0])
    for kind in KINDS:
        cand = [m for m in models if table.get(m, {}).get(kind, {}).get("10.0")]
        orders[kind] = sorted(cand, key=lambda m: -table[m][kind]["10.0"][0])
    for kind, order in orders.items():
        lines.append(f"- {kind}: " + " > ".join(order))
    lines.append("")
    if "gaussian" in orders:
        base = orders["gaussian"]
        for kind in KINDS:
            if kind not in orders:
                continue
            same = orders[kind] == base
            lines.append(f"- {kind} 与高斯排序{'一致' if same else '不同'}。")
    lines.append("")
    lines.append("> 说明：训练是干净的（无噪声增广），所以只有测试噪声类型在变；"
                 "同一 model/seed 在不同 kind 下会各自重训一次（为断点续跑而逐 run 缓存），"
                 "MPS 非确定性会让 clean 列有千分位差异。")
    lines.append("")
    return "\n".join(lines)


def _agg(table):
    """{model: {kind: {key: [per-seed values]}}} -> {model: {kind: {key: [mean, std]}}}"""
    agg = {}
    for m, kinds in table.items():
        for k, d in kinds.items():
            agg.setdefault(m, {})[k] = {key: list(_ms(v)) for key, v in d.items() if v}
    return agg


def write_outputs(table, ref, models):
    agg = _agg(table)
    payload = {
        "experiment": "e6_noise_types",
        "protocol": dict(split="safe", overlap=0.5, win=1024, epochs=20,
                         train_noise=None, kinds=KINDS, snrs=SNRS, models=models),
        "tables": {m: agg.get(m, {}) for m in models},
        "raw_per_seed": {m: table.get(m, {}) for m in models},
        "gaussian_reference": {m: ref.get(m, {}) for m in models},
    }
    (Path(RESULTS_DIR) / "e6_noise_types.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False))
    md = render_md(agg, ref, models)
    (Path(RESULTS_DIR) / "e6_noise_types.md").write_text(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ref-seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ref_seeds = [int(s) for s in args.ref_seeds.split(",")]
    models = args.models.split(",")

    out_dir = Path(RESULTS_DIR) / "runs" / "e6_noise_types"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for model in models:
        for kind in KINDS:
            for seed in seeds:
                rid = f"{model}_{kind}_seed{seed}"
                path = out_dir / f"{rid}.json"
                if path.exists():
                    res = json.loads(path.read_text())
                else:
                    res = train_one(dict(model=model, split="safe", epochs=args.epochs,
                                         seed=seed, noise_kind=kind,
                                         noise_snrs=list(SNRS)))
                    res["run_id"] = rid
                    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
                d = table.setdefault(model, {}).setdefault(kind, {})
                d.setdefault("clean", []).append(res["test_acc"])
                for snr in SNRS:
                    d.setdefault(str(snr), []).append(
                        res["noise"]["acc_by_snr"][str(snr)])
                write_outputs(table, gaussian_reference(models, ref_seeds), models)
                print(f"{rid} clean={res['test_acc']:.4f} " +
                      " ".join(f"{k}={v:.3f}" for k, v in
                               res["noise"]["acc_by_snr"].items()), flush=True)

    md = write_outputs(table, gaussian_reference(models, ref_seeds), models)
    print(md, flush=True)


if __name__ == "__main__":
    main()
