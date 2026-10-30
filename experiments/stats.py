"""Statistics for the paper: tidy run table + paired tests with Holm correction.

Reads every per-run json under results/runs/<grid>/ and produces
results/stats.md (tables) and results/stats.json (raw numbers).

Usage: python -m experiments.stats
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from code.common import RESULTS_DIR, save_json

SNRS = ["15.0", "10.0", "5.0", "0.0", "-5.0"]
TRACKED = ["10.0", "5.0", "0.0"]


def load_runs():
    rows = []
    for f in sorted(Path(RESULTS_DIR, "runs").glob("*/*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        cfg = d.get("cfg", {})
        noise = d.get("noise") or {}
        by_snr = noise.get("acc_by_snr") or {}
        rows.append(
            dict(
                grid=f.parent.name,
                run_id=d.get("run_id", f.stem),
                model=cfg.get("model", "?"),
                seed=cfg.get("seed", 0),
                split=cfg.get("split", "?"),
                epochs=cfg.get("epochs"),
                stem=tuple(cfg["stem"]) if cfg.get("stem") else None,
                aug=cfg.get("train_noise_lo") is not None,
                holdout=cfg.get("holdout_classes"),
                clean=d.get("test_acc"),
                n_params=d.get("n_params"),
                train_time_s=d.get("train_time_s"),
                acc={k: v for k, v in by_snr.items()},
                openset=d.get("openset"),
            )
        )
    return rows


def mean_std(values):
    v = [x for x in values if x is not None]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def condition(row):
    """Human-readable condition label."""
    if row["stem"]:
        k, s, c = (list(row["stem"]) + [1, 1])[:3]
        tag = f"stem k{k} c{c}"
    else:
        tag = "plain"
    return tag + (" +aug" if row["aug"] else "")


def paired(a_map, b_map):
    """Pairs on shared keys; returns (n, mean_a, mean_b, delta, p)."""
    keys = sorted(set(a_map) & set(b_map))
    a = np.array([a_map[k] for k in keys], dtype=float)
    b = np.array([b_map[k] for k in keys], dtype=float)
    if len(keys) < 4:
        return len(keys), a.mean() if len(a) else np.nan, b.mean() if len(b) else np.nan, np.nan, np.nan
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(b, a)
        p = float(stat.pvalue)
    except Exception:
        p = float("nan")
    return len(keys), float(a.mean()), float(b.mean()), float(b.mean() - a.mean()), p


def holm(pvals):
    """Holm-Bonferroni; returns adjusted p in the same order."""
    idx = [i for i, p in enumerate(pvals) if p == p and p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = list(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def main():
    rows = load_runs()
    out = {"n_runs": len(rows), "tables": {}}
    lines = ["# Statistical summary", "", f"Runs scanned: {len(rows)}", ""]

    # ---- T1: clean vs noisy per model, clean-trained only (no aug, no stem)
    clean_trained = [r for r in rows if not r["aug"] and not r["stem"] and r["split"] in ("safe", "leaky", "crossload", "filedisjoint")]
    if clean_trained:
        lines += ["## T1 clean-trained: clean / 10 dB / 5 dB / 0 dB (mean±std over seeds)", "",
                  "| model | split | clean | 10 dB | 5 dB | 0 dB | n_seeds |", "|---|---|---:|---:|---:|---:|---:|"]
        t1 = {}
        for model in sorted({r["model"] for r in clean_trained}):
            for split in sorted({r["split"] for r in clean_trained if r["model"] == model}):
                sel = [r for r in clean_trained if r["model"] == model and r["split"] == split]
                c_m, c_s = mean_std([r["clean"] for r in sel])
                cells = []
                for snr in TRACKED:
                    m, s = mean_std([r["acc"].get(snr) for r in sel])
                    cells.append(f"{m:.4f}±{s:.4f}" if m == m else "—")
                lines.append(f"| {model} | {split} | {c_m:.4f}±{c_s:.4f} | " + " | ".join(cells) + f" | {len(sel)} |")
                t1[f"{model}|{split}"] = dict(clean=c_m, clean_std=c_s,
                                              **{s: mean_std([r["acc"].get(s) for r in sel])[0] for s in TRACKED})
        out["tables"]["T1_clean_trained"] = t1
        lines.append("")

    # ---- T2: paired tests aug vs clean-trained (paired on model x seed)
    cmp_defs = []
    for grid, label in [("e4_aug", "aug"), ("e2_noise_clean", "clean-trained")]:
        pass
    base = {}
    for r in rows:
        if r["split"] != "safe" or r["stem"]:
            continue
        # only the clean-vs-augmentation pair grids feed the paired tests
        if r["grid"] not in ("e2_noise_clean", "e4_aug"):
            continue
        if r["aug"]:
            base.setdefault("aug", {})[(r["model"], r["seed"])] = r
        else:
            base.setdefault("clean", {})[(r["model"], r["seed"])] = r
    if "aug" in base and "clean" in base:
        lines += ["## T2 paired Wilcoxon: noise augmentation vs clean training (pairs = model x seed)", "",
                  "| metric | n pairs | clean-trained | +aug | Δ | p (Wilcoxon) |", "|---|---:|---:|---:|---:|---:|"]
        t2 = {}
        tests = []
        for metric, getter in [("clean", lambda r: r["clean"])] + [(f"{s} dB", (lambda s: (lambda r: r["acc"].get(s)))(s)) for s in TRACKED]:
            a = {k: getter(r) for k, r in base["clean"].items()}
            b = {k: getter(r) for k, r in base["aug"].items()}
            a = {k: v for k, v in a.items() if v is not None}
            b = {k: v for k, v in b.items() if v is not None}
            n, ma, mb, delta, p = paired(a, b)
            tests.append((metric, n, ma, mb, delta, p))
        ps = holm([t[5] for t in tests])
        for (metric, n, ma, mb, delta, p), padj in zip(tests, ps):
            lines.append(f"| {metric} | {n} | {ma:.4f} | {mb:.4f} | {delta:+.4f} | {p:.4g} (adj {padj:.4g}) |")
            t2[metric] = dict(n=n, clean_trained=ma, aug=mb, delta=delta, p=p, p_holm=padj)
        out["tables"]["T2_aug_vs_clean"] = t2
        lines.append("")

    # ---- T3: openset summary
    os_rows = [r for r in rows if r["openset"]]
    if os_rows:
        lines += ["## T3 open-set rejection (unknown = held-out fault classes)", "",
                  "| model | holdout | known acc | AUROC | TPR@95TNR | acc@5% reject |", "|---|---|---:|---:|---:|---:|"]
        t3 = {}
        for model in sorted({r["model"] for r in os_rows}):
            for ho in sorted({str(r["holdout"]) for r in os_rows if r["model"] == model}):
                sel = [r for r in os_rows if r["model"] == model and str(r["holdout"]) == ho]
                g = lambda k: mean_std([r["openset"].get(k) for r in sel])[0]
                lines.append(f"| {model} | {ho} | {g('known_acc'):.4f} | {g('auroc'):.4f} | {g('tpr_at_95fpr'):.4f} | {g('acc_at_reject_5pct'):.4f} |")
                t3[f"{model}|{ho}"] = {k: g(k) for k in ("known_acc", "auroc", "tpr_at_95fpr", "acc_at_reject_5pct")}
        out["tables"]["T3_openset"] = t3
        lines.append("")

    save_json(Path(RESULTS_DIR, "stats.json"), out)
    Path(RESULTS_DIR, "stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote results/stats.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
