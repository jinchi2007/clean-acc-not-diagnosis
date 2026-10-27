"""E9: does noise augmentation also improve open-set rejection?

E3 measured rejection for clean-trained models only, which the manuscript lists as a
limitation. This grid trains with the same per-batch augmentation as E4 (SNR ~ U[0, 15])
and measures the same three quantities -- closed-set accuracy, accuracy at 10/0 dB, and
open-set AUROC / TPR at 5% FPR on two held-out class configurations -- so the two axes
can be compared on the same runs.

    python -m experiments.e9_openset_aug --seeds 0,1 --epochs 20

Resumable: every run writes results/runs/e9_openset_aug/<run_id>.json, and the
aggregate is rebuilt from those files.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from code.common import RESULTS_DIR
from code.train import train_one

MODELS = ["wdcnn", "ticnn", "mcnn", "resnet1d", "ds_cnn", "lenet1d", "vit1d", "mlp_mixer1d"]
HOLDS = [[3, 6, 9], [1, 4, 7]]
SNRS = [10.0, 0.0]
RUNS = Path(RESULTS_DIR) / "runs" / "e9_openset_aug"


def cfg_for(model, hold, seed, epochs):
    return dict(model=model, split="safe", overlap=0.5, win=1024, seed=seed,
                epochs=epochs, holdout_classes=list(hold),
                train_noise_lo=0.0, train_noise_hi=15.0,
                noise_snrs=list(SNRS))


def run_id(cfg):
    return "%s_safe_seed%d_ep%d_hold%s_aug0-15" % (
        cfg["model"], cfg["seed"], cfg["epochs"], "".join(map(str, cfg["holdout_classes"])))


def run(seeds, epochs):
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [cfg_for(m, h, s, epochs) for m in MODELS for h in HOLDS for s in seeds]
    results = []
    for i, cfg in enumerate(todo, 1):
        path = RUNS / (run_id(cfg) + ".json")
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        try:
            r = train_one(cfg)
            r["cfg"] = cfg
            r["run_id"] = run_id(cfg)
            path.write_text(json.dumps(r, indent=1, ensure_ascii=False))
            results.append(r)
            msg = "%s acc=%.4f auroc=%.3f t=%.0fs [%d/%d]" % (
                run_id(cfg), r["test_acc"], r["openset"]["auroc"], r["train_time_s"], i, len(todo))
        except Exception:
            msg = "FAILED " + run_id(cfg) + "\n" + traceback.format_exc()
        print(msg, flush=True)
    return results


def mean_std(v):
    a = np.asarray(v, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def aggregate(results):
    table = {}
    for r in results:
        m = r["cfg"]["model"]
        key = "hold" + "".join(map(str, r["cfg"]["holdout_classes"]))
        d = table.setdefault(m, {})
        d.setdefault(key, []).append((r["openset"]["auroc"], r["openset"]["tpr_at_95fpr"],
                                      r["test_acc"],
                                      r["noise"]["acc_by_snr"]["10.0"],
                                      r["noise"]["acc_by_snr"]["0.0"]))
    out = {}
    for m, d in table.items():
        row = {}
        aurocs, tprs = [], []
        for key, vals in d.items():
            arr = np.asarray(vals, dtype=float)
            row[key] = dict(auroc=float(arr[:, 0].mean()), tpr95=float(arr[:, 1].mean()),
                            known_acc=float(arr[:, 2].mean()))
            aurocs.append(float(arr[:, 0].mean()))
            tprs.append(float(arr[:, 1].mean()))
        allv = np.concatenate([np.asarray(v, dtype=float) for v in d.values()])
        row["mean"] = dict(auroc=float(allv[:, 0].mean()), tpr95=float(allv[:, 1].mean()),
                           acc10=float(allv[:, 3].mean()), acc0=float(allv[:, 4].mean()))
        out[m] = row

    # paired delta against clean-trained E3, where available
    e3_path = Path(RESULTS_DIR) / "e3_openset.json"
    deltas = {}
    if e3_path.exists():
        e3 = json.loads(e3_path.read_text())["table"]
        for m, row in out.items():
            if m not in e3:
                continue
            clean_auroc = 0.5 * (e3[m]["hold369"]["auroc"] + e3[m]["hold147"]["auroc"])
            clean_tpr = 0.5 * (e3[m]["hold369"]["tpr95"] + e3[m]["hold147"]["tpr95"])
            deltas[m] = dict(d_auroc=row["mean"]["auroc"] - clean_auroc,
                             d_tpr=row["mean"]["tpr95"] - clean_tpr,
                             clean_auroc=clean_auroc, clean_tpr=clean_tpr)
    return out, deltas


def paired_stats(results):
    """Paired comparison against clean-trained E3 on the same (model, hold, seed)."""
    e3_dir = Path(RESULTS_DIR) / "runs" / "e3_openset"
    if not e3_dir.exists():
        return {}
    clean = {}
    for p in e3_dir.glob("*.json"):
        r = json.loads(p.read_text())
        cfg = r.get("cfg", {})
        if "holdout_classes" not in cfg or cfg.get("train_noise_lo") is not None:
            continue
        key = (cfg["model"], cfg["seed"], tuple(cfg["holdout_classes"]))
        clean[key] = (r["openset"]["auroc"], r["openset"]["tpr_at_95fpr"])
    pairs = []
    for r in results:
        cfg = r["cfg"]
        key = (cfg["model"], cfg["seed"], tuple(cfg["holdout_classes"]))
        if key not in clean:
            continue
        pairs.append((clean[key][0], r["openset"]["auroc"],
                      clean[key][1], r["openset"]["tpr_at_95fpr"]))
    if len(pairs) < 3:
        return {"n_pairs": len(pairs)}
    from scipy.stats import wilcoxon
    arr = np.asarray(pairs, dtype=float)
    out = {"n_pairs": len(pairs), "metrics": {}}
    ps = []
    for name, col in (("auroc", 0), ("tpr95", 2)):
        a, b = arr[:, col], arr[:, col + 1]
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:
            stat, p = float("nan"), 1.0
        out["metrics"][name] = {"clean_trained": float(a.mean()), "augmented": float(b.mean()),
                                "delta": float(b.mean() - a.mean()),
                                "n_improved": int((b > a).sum()), "p": float(p)}
        ps.append(p)
    order = np.argsort(ps)
    m = len(ps)
    for rank, idx in enumerate(order):
        name = list(out["metrics"])[idx]
        out["metrics"][name]["p_holm"] = float(min(1.0, ps[idx] * (m - rank)))
    return out


def render_md(table, deltas, stats=None):
    head = ["| model | AUROC 3,6,9 | AUROC 1,4,7 | mean AUROC | TPR95 3,6,9 | TPR95 1,4,7 | mean TPR95 | 10 dB | 0 dB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for m in MODELS:
        if m not in table:
            continue
        r = table[m]
        rows.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            m, r["hold369"]["auroc"], r["hold147"]["auroc"], r["mean"]["auroc"],
            r["hold369"]["tpr95"], r["hold147"]["tpr95"], r["mean"]["tpr95"],
            r["mean"]["acc10"], r["mean"]["acc0"]))
    lines = ["# E9: open-set rejection under noise-augmented training", "",
             "Training: per-batch augmentation SNR ~ U[0, 15] dB (same as E4).",
             "Evaluation: two held-out class configurations, softmax-max confidence.", ""]
    lines += head + rows
    if deltas:
        lines += ["", "## Delta vs clean-trained (E3), paired on the same configurations", "",
                  "| model | clean-trained AUROC | augmented AUROC | delta | clean-trained TPR95 | augmented TPR95 | delta |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m in MODELS:
            if m not in deltas:
                continue
            d = deltas[m]
            lines.append("| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f |" % (
                m, d["clean_auroc"], table[m]["mean"]["auroc"], d["d_auroc"],
                d["clean_tpr"], table[m]["mean"]["tpr95"], d["d_tpr"]))
        dm = np.asarray([[d["d_auroc"], d["d_tpr"]] for d in deltas.values()], dtype=float)
        lines += ["", "Mean delta over %d models: AUROC %+.4f, TPR@5%%FPR %+.4f." % (
            len(dm), dm[:, 0].mean(), dm[:, 1].mean())]
    if stats and "metrics" in stats:
        lines += ["", "## Paired Wilcoxon over %d (model, holdout, seed) pairs" % stats["n_pairs"], "",
                  "| metric | clean-trained | augmented | delta | improved | p | p_Holm |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, d in stats["metrics"].items():
            lines.append("| %s | %.3f | %.3f | %+.3f | %d/%d | %.2g | %.2g |" % (
                name, d["clean_trained"], d["augmented"], d["delta"],
                d["n_improved"], stats["n_pairs"], d["p"], d["p_holm"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    results = run(seeds, args.epochs)
    table, deltas = aggregate(results)
    stats = paired_stats(results)
    payload = {"experiment": "e9_openset_aug", "n_runs": len(results),
               "paired_vs_e3": stats,
               "protocol": {"split": "safe", "overlap": 0.5, "win": 1024,
                            "train_noise": [0.0, 15.0], "snrs": SNRS},
               "table": table, "deltas_vs_clean_trained": deltas}
    (Path(RESULTS_DIR) / "e9_openset_aug.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    md = render_md(table, deltas, stats)
    (Path(RESULTS_DIR) / "e9_openset_aug.md").write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
