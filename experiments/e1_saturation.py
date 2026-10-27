"""E1 -- saturation check: is CWRU within-dataset already solved by every lightweight model?

Trains all 8 baselines under the three protocols we support:
    leaky      : pooled windows, stratified random 80/20   (what most papers do)
    safe       : per-recording temporal split + guard band (leakage-safe within condition)
    crossload  : train loads {0,1} hp, test loads {2,3} hp  (condition-disjoint)

If every model lands >95% under every protocol the benchmark is saturated and the
paper's argument must move to harder regimes (noise / few-shot / open-set / cross-machine).

    python -m experiments.e1_saturation --epochs 10
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from code.common import RESULTS_DIR, save_json
from code.models import MODELS
from code.train import DEFAULT_CFG, train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--splits", default="leaky,safe,crossload")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--task", default="10class")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e1_saturation.json"))
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m.strip()]
    splits = [s for s in args.splits.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for model, split, seed in itertools.product(models, splits, seeds):
        cfg = {**DEFAULT_CFG, "model": model, "split": split, "seed": seed,
               "epochs": args.epochs, "task": args.task}
        r = train_one(cfg)
        rows.append({"model": model, "split": split, "seed": seed,
                     "test_acc": r["test_acc"], "n_params": r["n_params"],
                     "train_time_s": r["train_time_s"]})
        print(f"{model:12s} {split:10s} s{seed} acc={r['test_acc']:.4f}", flush=True)

    # markdown table
    header = "| model | params | " + " | ".join(splits) + " |"
    sep = "|---|---:|" + "---:|" * len(splits)
    lines = [header, sep]
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        cells = []
        for split in splits:
            accs = [r["test_acc"] for r in sub if r["split"] == split]
            cells.append(f"{np.mean(accs):.4f}" if accs else "-")
        params = sub[0]["n_params"] if sub else 0
        lines.append(f"| {model} | {params} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    save_json(args.out, {"experiment": "E1_saturation", "epochs": args.epochs,
                         "seeds": seeds, "task": args.task, "rows": rows,
                         "table_md": table})
    with open(os.path.join(RESULTS_DIR, "e1_saturation.md"), "w", encoding="utf-8") as f:
        f.write("# E1 saturation check\n\n" + table + "\n")


if __name__ == "__main__":
    main()
