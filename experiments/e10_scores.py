"""E10: does the decision layer buy back the rejection that augmentation costs?

E9 showed that per-batch noise augmentation, which repairs closed-set robustness,
lowers open-set AUROC from 0.90 to 0.80 when the rejection score is the softmax
maximum.  A network's rejection behaviour is partly a property of the score, not
of the features, so this grid retrains the same 8 models x 2 holdout
configurations x 2 seeds x {clean, augmented} and compares three scores that cost
no extra training and no extra parameters:

    softmax      max_j softmax(z)_j                 (baseline, E3/E9)
    energy       -T log sum_j exp(z_j / T)          (Liu et al., 2020)
    mahalanobis  -min_k (z - mu_k)^T Sigma^-1 (z - mu_k)   (logit space)

    python -m experiments.e10_scores --seeds 0,1 --epochs 20

Resumable: results/runs/e10_scores/<run_id>.json per run; the aggregate is
rebuilt from those files.
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
SCORES = ["softmax", "energy", "mahalanobis"]
MODES = ["clean", "aug"]
SNRS = [10.0, 0.0]
RUNS = Path(RESULTS_DIR) / "runs" / "e10_scores"


def cfg_for(model, hold, seed, mode, epochs):
    cfg = dict(model=model, split="safe", overlap=0.5, win=1024, seed=seed,
               epochs=epochs, holdout_classes=list(hold), noise_snrs=list(SNRS),
               score_eval=True)
    if mode == "aug":
        cfg["train_noise_lo"] = 0.0
        cfg["train_noise_hi"] = 15.0
    return cfg


def run_id(cfg):
    mode = "aug0-15" if cfg.get("train_noise_lo") is not None else "clean"
    return "%s_safe_seed%d_ep%d_hold%s_%s" % (
        cfg["model"], cfg["seed"], cfg["epochs"],
        "".join(map(str, cfg["holdout_classes"])), mode)


def run(seeds, epochs):
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [cfg_for(m, h, s, mode, epochs) for m in MODELS for h in HOLDS for s in seeds for mode in MODES]
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
            sc = r.get("openset_scores", {})
            msg = "%s auroc(soft/energy/maha)=%.3f/%.3f/%.3f t=%.0fs [%d/%d]" % (
                run_id(cfg), sc.get("softmax", {}).get("auroc", float("nan")),
                sc.get("energy", {}).get("auroc", float("nan")),
                sc.get("mahalanobis", {}).get("auroc", float("nan")),
                r["train_time_s"], i, len(todo))
        except Exception:
            msg = "FAILED " + run_id(cfg) + "\n" + traceback.format_exc()
        print(msg, flush=True)
    return results


def _mode(cfg):
    return "aug" if cfg.get("train_noise_lo") is not None else "clean"


def aggregate(results):
    table = {}
    for r in results:
        m, mode = r["cfg"]["model"], _mode(r["cfg"])
        d = table.setdefault(m, {}).setdefault(mode, {s: [] for s in SCORES})
        sc = r.get("openset_scores", {})
        for s in SCORES:
            if s in sc:
                d[s].append((sc[s]["auroc"], sc[s]["tpr_at_95fpr"]))
    out = {}
    for m, modes in table.items():
        out[m] = {}
        for mode, scores in modes.items():
            row = {}
            for s, vals in scores.items():
                a = np.asarray(vals, dtype=float)
                if len(a):
                    row[s] = dict(auroc=float(a[:, 0].mean()), tpr95=float(a[:, 1].mean()), n=int(len(a)))
            acc = [r["test_acc"] for r in results
                   if r["cfg"]["model"] == m and _mode(r["cfg"]) == mode]
            row["n_runs"] = len(acc)
            row["known_acc"] = float(np.mean(acc)) if acc else float("nan")
            out[m][mode] = row
    return out


def paired(results):
    """Paired tests over (model, holdout, seed), one row per training mode."""
    key = lambda r: (r["cfg"]["model"], r["cfg"]["seed"], tuple(r["cfg"]["holdout_classes"]))
    by_mode = {"clean": {}, "aug": {}}
    for r in results:
        by_mode[_mode(r["cfg"])][key(r)] = r
    out = {}
    from scipy.stats import wilcoxon

    def holm(pvals):
        order = np.argsort(pvals)
        m = len(pvals)
        adj = [0.0] * m
        for rank, idx in enumerate(order):
            adj[idx] = min(1.0, pvals[idx] * (m - rank))
        return adj

    for mode in MODES:
        rows = by_mode[mode]
        cands = {}
        # score choice, paired within the same training mode
        for s in SCORES[1:]:
            pairs = [(rows[k]["openset_scores"]["softmax"]["auroc"],
                      rows[k]["openset_scores"][s]["auroc"],
                      rows[k]["openset_scores"]["softmax"]["tpr_at_95fpr"],
                      rows[k]["openset_scores"][s]["tpr_at_95fpr"])
                     for k in rows if "openset_scores" in rows[k]]
            if len(pairs) < 3:
                continue
            arr = np.asarray(pairs, dtype=float)
            entry = {"n_pairs": len(pairs), "metrics": {}}
            ps = []
            for name, col in (("auroc", 0), ("tpr95", 2)):
                a, b = arr[:, col], arr[:, col + 1]
                stat, p = wilcoxon(a, b)
                entry["metrics"][name] = {"softmax": float(a.mean()), s: float(b.mean()),
                                          "delta": float(b.mean() - a.mean()),
                                          "n_improved": int((b > a).sum()), "p": float(p)}
                ps.append(p)
            for name, p_adj in zip(entry["metrics"], holm(ps)):
                entry["metrics"][name]["p_holm"] = float(p_adj)
            cands["%s_vs_softmax" % s] = entry
        out[mode] = cands

    # augmentation cost, per score, paired clean vs aug
    costs = {}
    for s in SCORES:
        pairs = []
        for k, rc in by_mode["clean"].items():
            if k not in by_mode["aug"]:
                continue
            ra = by_mode["aug"][k]
            if "openset_scores" not in rc or "openset_scores" not in ra:
                continue
            pairs.append((rc["openset_scores"][s]["auroc"], ra["openset_scores"][s]["auroc"],
                          rc["openset_scores"][s]["tpr_at_95fpr"], ra["openset_scores"][s]["tpr_at_95fpr"]))
        if len(pairs) < 3:
            continue
        arr = np.asarray(pairs, dtype=float)
        entry = {"n_pairs": len(pairs), "metrics": {}}
        ps = []
        for name, col in (("auroc", 0), ("tpr95", 2)):
            a, b = arr[:, col], arr[:, col + 1]
            stat, p = wilcoxon(a, b)
            entry["metrics"][name] = {"clean": float(a.mean()), "aug": float(b.mean()),
                                      "delta": float(b.mean() - a.mean()),
                                      "n_improved": int((b < a).sum()), "p": float(p)}
            ps.append(p)
        for name, p_adj in zip(entry["metrics"], holm(ps)):
            entry["metrics"][name]["p_holm"] = float(p_adj)
        costs[s] = entry
    return {"score_comparison": out, "augmentation_cost_by_score": costs}


def render_md(table, stats):
    lines = ["# E10: decision-layer scores for open-set rejection", "",
             "Same networks as E3/E9 (8 models, two held-out class sets, two seeds).",
             "Scores: softmax maximum (baseline), negative free energy, and negative",
             "Mahalanobis distance to the class-conditional Gaussians in logit space.", "",
             "## Mean AUROC / TPR at 5% FPR, clean-trained", "",
             "| model | softmax AUROC | energy AUROC | Mahalanobis AUROC | softmax TPR95 | energy TPR95 | Mahalanobis TPR95 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        if m not in table or "clean" not in table[m]:
            continue
        r = table[m]["clean"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            m, r["softmax"]["auroc"], r["energy"]["auroc"], r["mahalanobis"]["auroc"],
            r["softmax"]["tpr95"], r["energy"]["tpr95"], r["mahalanobis"]["tpr95"]))
    lines += ["", "## Mean AUROC / TPR at 5% FPR, augmented (SNR ~ U[0, 15] dB)", "",
              "| model | softmax AUROC | energy AUROC | Mahalanobis AUROC | softmax TPR95 | energy TPR95 | Mahalanobis TPR95 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        if m not in table or "aug" not in table[m]:
            continue
        r = table[m]["aug"]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            m, r["softmax"]["auroc"], r["energy"]["auroc"], r["mahalanobis"]["auroc"],
            r["softmax"]["tpr95"], r["energy"]["tpr95"], r["mahalanobis"]["tpr95"]))
    for mode in MODES:
        comp = stats["score_comparison"].get(mode, {})
        if not comp:
            continue
        lines += ["", "## %s: score choice (paired over model x holdout x seed)" % mode, "",
                  "| comparison | metric | softmax | other | delta | improved | p | p_Holm |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, entry in comp.items():
            for metric, d in entry["metrics"].items():
                other = [k for k in d if k not in ("softmax", "delta", "n_improved", "p", "p_holm")][0]
                lines.append("| %s | %s | %.3f | %.3f | %+.3f | %d/%d | %.2g | %.2g |" % (
                    name, metric, d["softmax"], d[other], d["delta"], d["n_improved"],
                    entry["n_pairs"], d["p"], d["p_holm"]))
    costs = stats["augmentation_cost_by_score"]
    if costs:
        lines += ["", "## Cost of augmentation on rejection, per score", "",
                  "| score | clean AUROC | aug AUROC | delta | clean TPR95 | aug TPR95 | delta |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for s, entry in costs.items():
            a, t = entry["metrics"]["auroc"], entry["metrics"]["tpr95"]
            lines.append("| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f |" % (
                s, a["clean"], a["aug"], a["delta"], t["clean"], t["aug"], t["delta"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    results = run(seeds, args.epochs)
    table = aggregate(results)
    stats = paired(results)
    payload = {"experiment": "e10_scores", "n_runs": len(results),
               "protocol": {"split": "safe", "overlap": 0.5, "win": 1024, "snrs": SNRS,
                            "scores": SCORES, "modes": MODES},
               "table": table, "stats": stats}
    (Path(RESULTS_DIR) / "e10_scores.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    md = render_md(table, stats)
    (Path(RESULTS_DIR) / "e10_scores.md").write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
