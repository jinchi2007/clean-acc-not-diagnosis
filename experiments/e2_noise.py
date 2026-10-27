"""E2 -- noise stress test: does clean accuracy predict robustness?

Every model is trained once on clean data (leakage-safe split) and then evaluated
on the same test set corrupted by additive white Gaussian noise at several SNRs.
This is the deployment-relevant question: in a real plant the sensor is noisy.

    python -m experiments.e2_noise --seeds 0,1,2 --epochs 15
"""
from __future__ import annotations

import argparse
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.common import FIG_DIR, RESULTS_DIR, save_json
from code.models import MODELS
from code.train import DEFAULT_CFG, train_one

SNRS = [15.0, 10.0, 5.0, 0.0, -5.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--split", default="safe")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e2_noise.json"))
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    for model, seed in itertools.product(models, seeds):
        cfg = {**DEFAULT_CFG, "model": model, "split": args.split, "seed": seed,
               "epochs": args.epochs, "noise_snrs": SNRS}
        r = train_one(cfg)
        row = {"model": model, "seed": seed, "clean": r["test_acc"],
               "by_snr": {k: v for k, v in r["noise"]["acc_by_snr"].items()},
               "n_params": r["n_params"]}
        rows.append(row)
        print(f"{model:12s} seed={seed} clean={r['test_acc']:.4f} " +
              " ".join(f"{k}dB={v:.3f}" for k, v in row["by_snr"].items()), flush=True)

    xs = SNRS
    table_models = models
    lines = ["| model | params | clean | " + " | ".join(f"{int(s)} dB" for s in xs) + " |",
             "|---|---:|---:|" + "---:|" * len(xs)]
    for m in table_models:
        sub = [r for r in rows if r["model"] == m]
        clean = np.mean([r["clean"] for r in sub])
        cells = [f"{np.mean([r['by_snr'][str(s)] for r in sub]):.3f}" for s in xs]
        lines.append(f"| {m} | {sub[0]['n_params']} | {clean:.4f} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    save_json(args.out, {"experiment": "E2_noise", "split": args.split,
                         "epochs": args.epochs, "seeds": seeds, "snrs": SNRS, "rows": rows,
                         "table_md": table})
    with open(os.path.join(RESULTS_DIR, "e2_noise.md"), "w", encoding="utf-8") as f:
        f.write("# E2 noise stress test (trained clean)\n\n" + table + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    ax = axes[0]
    for m in table_models:
        sub = [r for r in rows if r["model"] == m]
        means = [np.mean([r["by_snr"][str(s)] for r in sub]) for s in xs]
        ax.plot([20] + list(xs), [np.mean([r["clean"] for r in sub])] + means, "o-", label=m, linewidth=1.4)
    ax.set_xlabel("test SNR (dB), 20 = clean")
    ax.set_ylabel("accuracy")
    ax.set_xlim(-6, 21)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("E2: trained clean, tested noisy", fontsize=10)

    ax = axes[1]
    clean = [np.mean([r["clean"] for r in rows if r["model"] == m]) for m in table_models]
    at10 = [np.mean([r["by_snr"]["10.0"] for r in rows if r["model"] == m]) for m in table_models]
    order = np.argsort(clean)
    ax.scatter(np.array(clean)[order], np.array(at10)[order], c="#d62728", zorder=3)
    for i, idx in enumerate(order):
        ax.annotate(table_models[idx], (clean[idx], at10[idx]), fontsize=7,
                    xytext=(4, -3), textcoords="offset points")
    ax.plot([0.3, 1.0], [0.3, 1.0], "--", color="gray", linewidth=1)
    ax.set_xlabel("clean accuracy")
    ax.set_ylabel("accuracy at 10 dB SNR")
    ax.set_title("clean vs noisy accuracy (rank inversion)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, "e2_noise.png")
    fig.savefig(png)
    with open(os.path.join(FIG_DIR, "e2_noise.png"), "rb"):
        pass
    print(f"saved figure -> {png}")


if __name__ == "__main__":
    main()
