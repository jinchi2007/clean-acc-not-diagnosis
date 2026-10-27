"""实验网格驱动：可断点续跑，每次运行落一个 JSON，最后聚合出表与图。

用法:
  python -m experiments.grid --grid e4_aug --grid e0_sat
  python -m experiments.grid --grid all --seeds 0,1,2 --epochs 20

网格:
  e0_sat         8 模型 x 4 划分(leaky/safe/crossload/filedisjoint) x seeds，干净条件
  e2_noise_clean 8 模型 x seeds，干净训练，测 5 个 SNR
  e4_aug         8 模型 x seeds，训练时 SNR~U[0,15] 增广，测 5 个 SNR
  e3_openset     8 模型 x 2 个留出类集合 x seeds
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from code.common import RESULTS_DIR, FIG_DIR
from code.train import train_one

MODELS = ["wdcnn", "ticnn", "mcnn", "resnet1d", "ds_cnn", "lenet1d", "vit1d", "mlp_mixer1d"]
SNRS = [15.0, 10.0, 5.0, 0.0, -5.0]
RUNS = Path(RESULTS_DIR) / "runs"


def runs_for(grid, seeds, epochs):
    def base(model, seed, **kw):
        return dict(model=model, split=kw.pop("split", "safe"), overlap=0.5, win=1024,
                    seed=seed, epochs=epochs, **kw)

    if grid == "e0_sat":
        for model in MODELS:
            for split in ("leaky", "safe", "crossload", "filedisjoint"):
                for seed in seeds:
                    yield base(model, seed, split=split)
    elif grid in ("e2_noise_clean", "e4_aug"):
        for model in MODELS:
            for seed in seeds:
                cfg = base(model, seed, noise_snrs=list(SNRS))
                if grid == "e4_aug":
                    cfg.update(train_noise_lo=0.0, train_noise_hi=15.0)
                yield cfg
    elif grid == "e3_openset":
        for model in MODELS:
            for hold in ([3, 6, 9], [1, 4, 7]):
                for seed in seeds:
                    yield base(model, seed, holdout_classes=list(hold))
    else:
        raise ValueError("unknown grid " + grid)


def run_id(cfg):
    parts = [cfg["model"], cfg["split"], "seed%d" % cfg["seed"], "ep%d" % cfg["epochs"]]
    if cfg.get("holdout_classes"):
        parts.append("hold" + "".join(map(str, cfg["holdout_classes"])))
    if cfg.get("train_noise_lo") is not None:
        parts.append("aug%g-%g" % (cfg["train_noise_lo"], cfg["train_noise_hi"]))
    if cfg.get("stem"):
        parts.append("stem" + "x".join(map(str, cfg["stem"])))
    return "_".join(parts)


def run_grid(grid, seeds, epochs):
    out_dir = RUNS / grid
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "a", buffering=1)
    results = []
    todo = list(runs_for(grid, seeds, epochs))
    print("[%s] %d runs" % (grid, len(todo)), flush=True)
    for i, cfg in enumerate(todo, 1):
        rid = run_id(cfg)
        path = out_dir / (rid + ".json")
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        try:
            res = train_one(cfg)
            res["cfg"] = cfg
            res["run_id"] = rid
            path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            results.append(res)
            msg = "%s acc=%.4f t=%.0fs [%d/%d]" % (rid, res["test_acc"], res["train_time_s"], i, len(todo))
        except Exception:
            msg = "FAILED " + rid + "\n" + traceback.format_exc()
        print(msg, flush=True)
        log.write(msg + "\n")
    log.close()
    return results


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def aggregate(grid, results):
    agg = {}
    if grid == "e0_sat":
        for r in results:
            agg.setdefault(r["cfg"]["model"], {}).setdefault(r["cfg"]["split"], []).append(r["test_acc"])
        table = {m: {s: _mean_std(v) for s, v in d.items()} for m, d in agg.items()}
    elif grid in ("e2_noise_clean", "e4_aug"):
        for r in results:
            d = agg.setdefault(r["cfg"]["model"], {})
            d.setdefault("clean", []).append(r["test_acc"])
            for snr, acc in r["noise"]["acc_by_snr"].items():
                d.setdefault("%+.0fdB" % float(snr), []).append(acc)
        table = {m: {k: _mean_std(v) for k, v in d.items()} for m, d in agg.items()}
    elif grid == "e3_openset":
        for r in results:
            d = agg.setdefault(r["cfg"]["model"], {})
            key = "hold" + "".join(map(str, r["cfg"]["holdout_classes"]))
            d.setdefault(key, []).append((r["openset"]["auroc"], r["openset"]["tpr_at_95fpr"], r["test_acc"]))
        table = {}
        for m, d in agg.items():
            table[m] = {}
            for k, v in d.items():
                arr = np.asarray(v, dtype=float)
                table[m][k] = dict(auroc=float(arr[:, 0].mean()), tpr95=float(arr[:, 1].mean()),
                                   known_acc=float(arr[:, 2].mean()))
    else:
        raise ValueError(grid)

    payload = {"grid": grid, "n_runs": len(results), "table": table}
    (Path(RESULTS_DIR) / (grid + ".json")).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    md = render_md(grid, table)
    (Path(RESULTS_DIR) / (grid + ".md")).write_text(md)
    print(md, flush=True)
    try:
        plot(grid, table)
    except Exception as e:
        print("plot failed:", e, flush=True)


def render_md(grid, table):
    if grid == "e0_sat":
        cols = ["leaky", "safe", "crossload", "filedisjoint"]
    elif grid in ("e2_noise_clean", "e4_aug"):
        cols = ["clean"] + ["%+.0fdB" % s for s in SNRS]
    else:
        cols = sorted({k for m in table.values() for k in m})
    head = "| model | " + " | ".join(cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    rows = []
    for m in MODELS:
        if m not in table:
            continue
        cells = []
        for c in cols:
            v = table[m].get(c)
            if v is None:
                cells.append("-")
            elif isinstance(v, dict):
                cells.append("AUROC %.3f / TPR95 %.2f" % (v["auroc"], v["tpr95"]))
            else:
                cells.append("%.4f±%.3f" % (v[0], v[1]))
        rows.append("| " + m + " | " + " | ".join(cells) + " |")
    return "# " + grid + "\n\n" + head + "\n" + sep + "\n" + "\n".join(rows) + "\n"


def plot(grid, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    if grid == "e0_sat":
        splits = ["leaky", "safe", "crossload", "filedisjoint"]
        names = [m for m in MODELS if m in table]
        mat = [[table[m][s][0] for s in splits] for m in names]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(splits)), splits, rotation=20)
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(splits)):
                ax.text(j, i, "%.1f" % (mat[i][j] * 100), ha="center", va="center",
                        color="w" if mat[i][j] < 0.85 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, label="clean test accuracy")
        ax.set_title("E0: saturation across protocols")
    else:
        xs = [0, 15, 10, 5, 0, -5]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for m in MODELS:
            if m not in table:
                continue
            ys = [table[m]["clean"][0]] + [table[m]["%+.0fdB" % s][0] for s in SNRS]
            ax.plot(xs, ys, marker="o", ms=4, label=m)
        ax.set_xlabel("test SNR (dB); 0 = clean")
        ax.set_ylabel("test accuracy")
        ax.set_title("E4: noise-augmented training" if grid == "e4_aug" else "E2: clean training under test noise")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / (grid + ".png"), dpi=160)
    print("figure ->", FIG_DIR / (grid + ".png"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="append", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    grids = args.grid
    if grids == ["all"]:
        grids = ["e0_sat", "e2_noise_clean", "e4_aug", "e3_openset"]
    for g in grids:
        res = run_grid(g, seeds, args.epochs)
        aggregate(g, res)


if __name__ == "__main__":
    main()
