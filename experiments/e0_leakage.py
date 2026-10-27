"""E0 -- leakage audit: how much accuracy is bought by splitting overlapping windows at random?

For each window overlap r in {0, .5, .75, .9} we train the same model twice:
  * split="leaky": all windows pooled, stratified random 80/20 split
  * split="safe" : per-recording temporal split with a guard band (no shared samples)

If the headline claim of the paper holds, leaky accuracy should rise (and stay
high) with r while safe accuracy stays flat / drops.

    python -m experiments.e0_leakage --seeds 0,1 --epochs 15
"""
from __future__ import annotations

import argparse
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.common import FIG_DIR, RESULTS_DIR, save_json
from code.train import DEFAULT_CFG, train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wdcnn")
    ap.add_argument("--task", default="10class")
    ap.add_argument("--overlaps", default="0,0.5,0.75,0.9")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e0_leakage.json"))
    args = ap.parse_args()

    overlaps = [float(v) for v in args.overlaps.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]

    rows = []
    for split in ("leaky", "safe"):
        for ov in overlaps:
            for seed in seeds:
                cfg = {
                    **DEFAULT_CFG,
                    "model": args.model,
                    "task": args.task,
                    "split": split,
                    "overlap": ov,
                    "seed": seed,
                    "epochs": args.epochs,
                }
                r = train_one(cfg)
                rows.append(
                    {
                        "split": split,
                        "overlap": ov,
                        "seed": seed,
                        "test_acc": r["test_acc"],
                        "best_test_acc": r["best_test_acc"],
                        "n_train": r["data_meta"]["n_train"],
                        "n_test": r["data_meta"]["n_test"],
                        "train_time_s": r["train_time_s"],
                    }
                )
                print(f"  {split:6s} ov={ov:.2f} seed={seed} acc={r['test_acc']:.4f}", flush=True)

    summary = {}
    for split in ("leaky", "safe"):
        for ov in overlaps:
            accs = [r["test_acc"] for r in rows if r["split"] == split and r["overlap"] == ov]
            summary[f"{split}@{ov}"] = {
                "mean": sum(accs) / len(accs),
                "std": st.pstdev(accs) if len(accs) > 1 else 0.0,
                "n": len(accs),
                "accs": accs,
            }

    # table
    lines = [
        "| overlap | leaky acc | safe acc | gap (leaky - safe) |",
        "|---|---|---|---|",
    ]
    for ov in overlaps:
        lk = summary[f"leaky@{ov}"]["mean"]
        sf = summary[f"safe@{ov}"]["mean"]
        lines.append(f"| {ov:.2f} | {lk:.4f} | {sf:.4f} | **{lk - sf:+.4f}** |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    payload = {
        "experiment": "E0_leakage_audit",
        "model": args.model,
        "task": args.task,
        "epochs": args.epochs,
        "seeds": seeds,
        "summary": summary,
        "runs": rows,
    }
    save_json(args.out, payload)

    md = os.path.join(RESULTS_DIR, "e0_leakage.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# E0 leakage audit\n\n")
        f.write(f"model={args.model}, task={args.task}, epochs={args.epochs}, seeds={seeds}\n\n")
        f.write(table + "\n")

    # figure
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    for split, color in (("leaky", "#d62728"), ("safe", "#1f77b4")):
        means = [summary[f"{split}@{ov}"]["mean"] for ov in overlaps]
        stds = [summary[f"{split}@{ov}"]["std"] for ov in overlaps]
        ax.plot(overlaps, means, "o-", color=color, label=f"{split} split", linewidth=1.8)
        ax.fill_between(overlaps, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.18)
        ax.set_xlabel("window overlap ratio $r$")
        ax.set_ylabel(f"test accuracy ({args.task})")
        ax.set_title("E0: window-level random split leaks overlapping windows", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend()
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    png = os.path.join(FIG_DIR, "e0_leakage.png")
    fig.savefig(png)
    print(f"saved figure -> {png}")


if __name__ == "__main__":
    main()
