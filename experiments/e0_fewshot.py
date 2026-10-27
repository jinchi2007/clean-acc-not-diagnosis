"""E0b: leakage under a few-shot regime (20 windows per class).

Rationale: Contribution 1 claims leakage is a small effect. At full data it is
<= 0.6 points; the claim also needs a few-shot number, where the same overlap
removes a much larger fraction of the training set.

Protocol is identical to e0_sat except n_per_class=20.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.train import train_one  # noqa: E402
from pathlib import Path

from code.common import RESULTS_DIR  # noqa: E402

RUNS_DIR = str(Path(RESULTS_DIR) / "runs")

MODELS = ["wdcnn", "mcnn", "resnet1d", "lenet1d"]
SPLITS = ["leaky", "safe"]
OUT_JSON = os.path.join(ROOT, "results", "e0_fewshot.json")
OUT_MD = os.path.join(ROOT, "results", "e0_fewshot.md")


def run_id(model, split, seed, n):
    return f"{model}_{split}_seed{seed}_ep20_n{n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-per-class", type=int, default=20)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(RUNS_DIR, exist_ok=True)
    d = os.path.join(RUNS_DIR, "e0_fewshot")
    os.makedirs(d, exist_ok=True)

    rows = []
    total = len(MODELS) * len(SPLITS) * len(seeds)
    i = 0
    for model in MODELS:
        for split in SPLITS:
            for seed in seeds:
                i += 1
                rid = run_id(model, split, seed, args.n_per_class)
                path = os.path.join(d, rid + ".json")
                if os.path.exists(path):
                    with open(path) as fh:
                        rows.append(json.load(fh))
                    print(f"[{i}/{total}] cached {rid}", flush=True)
                    continue
                cfg = dict(
                    dataset="cwru", task="10class", split=split, overlap=0.5,
                    win=1024, seed=seed, model=model, epochs=args.epochs,
                    batch=128, lr=2e-3, wd=1e-4, device="mps", input="raw",
                    n_per_class=args.n_per_class,
                )
                t0 = time.time()
                res = train_one(cfg)
                res["cfg"] = cfg
                with open(path, "w") as fh:
                    json.dump(res, fh)
                print(f"[{i}/{total}] {rid} acc={res['test_acc']:.4f} "
                      f"n_train={res['n_train']} t={time.time()-t0:.0f}s", flush=True)
                rows.append(res)

    # aggregate: mean +- std per model x split
    tab = {}
    for model in MODELS:
        tab[model] = {}
        for split in SPLITS:
            accs = [r["test_acc"] for r in rows
                    if r["cfg"]["model"] == model and r["cfg"]["split"] == split]
            tab[model][split] = dict(
                mean=float(np.mean(accs)), std=float(np.std(accs)), n=len(accs))
    out = dict(n_per_class=args.n_per_class, epochs=args.epochs,
               seeds=seeds, table=tab)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    lines = [f"# E0b: leakage in a few-shot regime ({args.n_per_class}/class)", "",
             "| model | leaky | safe | leaky - safe |", "|---|---:|---:|---:|"]
    for model in MODELS:
        a = tab[model]["leaky"]; b = tab[model]["safe"]
        lines.append(f"| {model} | {a['mean']:.4f}±{a['std']:.3f} | "
                     f"{b['mean']:.4f}±{b['std']:.3f} | {a['mean']-b['mean']:+.4f} |")
    d_means = [tab[m]["leaky"]["mean"] - tab[m]["safe"]["mean"] for m in MODELS]
    lines += ["", f"mean leaky - safe gap: {np.mean(d_means):+.4f} points "
                  f"({np.mean(d_means)*100:.1f} accuracy points)", ""]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
