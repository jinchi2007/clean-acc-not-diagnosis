"""E11: is the logit-space rejection result of E10 an artefact of using logits?

E10 showed that the same trained network rejects unknown faults far better when
the score is a distance in logit space than when it is the softmax maximum
(mean AUROC 0.899 -> 0.953 clean-trained).  A natural objection is that the
gain comes from a particular coordinate system.  This grid retrains the same
8 models x 2 holdout sets x 2 seeds x {clean, augmented} and compares three
distance-like read-outs computed from the *same* forward pass:

    softmax            max_j softmax(z)_j                       (baseline)
    mahalanobis        negative min class distance in logit space (E10)
    mahalanobis_feat   negative min class distance in the input of the last
                       linear layer, i.e. penultimate features

Feature-space Gaussians are fitted on the training set with a shared
covariance, exactly like the logit variant, so the only difference is where
the distance is measured.

    python -m experiments.e11_penultimate --seeds 0,1 --epochs 20

Resumable: results/runs/e11_penultimate/<run_id>.json per run.
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
SCORES = ["softmax", "mahalanobis", "mahalanobis_feat"]
MODES = ["clean", "aug"]
SNRS = [10.0, 0.0]
RUNS = Path(RESULTS_DIR) / "runs" / "e11_penultimate"


def cfg_for(model, hold, seed, mode, epochs):
    cfg = dict(model=model, split="safe", overlap=0.5, win=1024, seed=seed,
               epochs=epochs, holdout_classes=list(hold), noise_snrs=list(SNRS),
               score_eval=True, feature_scores=True)
    if mode == "aug":
        cfg["train_noise_lo"] = 0.0
        cfg["train_noise_hi"] = 15.0
    return cfg


def run_id(cfg):
    mode = "aug0-15" if cfg.get("train_noise_lo") is not None else "clean"
    return "%s_safe_seed%d_ep%d_hold%s_%s" % (
        cfg["model"], cfg["seed"], cfg["epochs"],
        "".join(map(str, cfg["holdout_classes"])), mode)


def _mode(cfg):
    return "aug" if cfg.get("train_noise_lo") is not None else "clean"


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
            msg = "%s auroc(soft/logit/feat)=%.3f/%.3f/%.3f t=%.0fs [%d/%d]" % (
                run_id(cfg), sc.get("softmax", {}).get("auroc", float("nan")),
                sc.get("mahalanobis", {}).get("auroc", float("nan")),
                sc.get("mahalanobis_feat", {}).get("auroc", float("nan")),
                r["train_time_s"], i, len(todo))
        except Exception:
            msg = "FAILED " + run_id(cfg) + "\n" + traceback.format_exc()
        print(msg, flush=True)
    return results


def aggregate(results):
    table = {}
    for r in results:
        m, mode = r["cfg"]["model"], _mode(r["cfg"])
        d = table.setdefault(m, {}).setdefault(mode, {s: [] for s in SCORES})
        sc = r.get("openset_scores", {})
        for s in SCORES:
            if s in sc:
                d[s].append((sc[s]["auroc"], sc[s]["tpr_at_95fpr"]))
        if "mahalanobis_feat" in sc and "dim" in sc["mahalanobis_feat"]:
            d.setdefault("dim", []).append(sc["mahalanobis_feat"]["dim"])
    out = {}
    for m, modes in table.items():
        out[m] = {}
        for mode, scores in modes.items():
            row = {}
            for s, vals in scores.items():
                if s == "dim":
                    continue
                a = np.asarray(vals, dtype=float)
                if len(a):
                    row[s] = dict(auroc=float(a[:, 0].mean()), tpr95=float(a[:, 1].mean()), n=int(len(a)))
            acc = [r["test_acc"] for r in results if r["cfg"]["model"] == m and _mode(r["cfg"]) == mode]
            row["n_runs"] = len(acc)
            row["known_acc"] = float(np.mean(acc)) if acc else float("nan")
            dims = table[m][mode].get("dim", [])
            row["feat_dim"] = int(dims[0]) if dims else None
            out[m][mode] = row
    return out


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - rank))
    return adj


def paired(results):
    from scipy.stats import wilcoxon

    key = lambda r: (r["cfg"]["model"], r["cfg"]["seed"], tuple(r["cfg"]["holdout_classes"]))
    by_mode = {"clean": {}, "aug": {}}
    for r in results:
        by_mode[_mode(r["cfg"])][key(r)] = r

    def compare(rows, ref, other):
        pairs = [(rows[k]["openset_scores"][ref]["auroc"], rows[k]["openset_scores"][other]["auroc"],
                  rows[k]["openset_scores"][ref]["tpr_at_95fpr"], rows[k]["openset_scores"][other]["tpr_at_95fpr"])
                 for k in rows
                 if "openset_scores" in rows[k]
                 and ref in rows[k]["openset_scores"] and other in rows[k]["openset_scores"]]
        if len(pairs) < 3:
            return None
        arr = np.asarray(pairs, dtype=float)
        entry = {"n_pairs": len(pairs), "ref": ref, "other": other, "metrics": {}}
        ps = []
        for name, col in (("auroc", 0), ("tpr95", 2)):
            a, b = arr[:, col], arr[:, col + 1]
            _stat, p = wilcoxon(a, b)
            entry["metrics"][name] = {ref: float(a.mean()), other: float(b.mean()),
                                      "delta": float(b.mean() - a.mean()),
                                      "n_improved": int((b > a).sum()), "p": float(p)}
            ps.append(p)
        for name, p_adj in zip(entry["metrics"], holm(ps)):
            entry["metrics"][name]["p_holm"] = float(p_adj)
        return entry

    out = {}
    for mode in MODES:
        rows = by_mode[mode]
        out[mode] = {}
        e = compare(rows, "softmax", "mahalanobis_feat")
        if e:
            out[mode]["mahalanobis_feat_vs_softmax"] = e
        e = compare(rows, "mahalanobis", "mahalanobis_feat")
        if e:
            out[mode]["mahalanobis_feat_vs_logit"] = e

    costs = {}
    for s in SCORES:
        pairs = []
        for k, rc in by_mode["clean"].items():
            if k not in by_mode["aug"]:
                continue
            ra = by_mode["aug"][k]
            if "openset_scores" not in rc or "openset_scores" not in ra:
                continue
            if s not in rc["openset_scores"] or s not in ra["openset_scores"]:
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
            _stat, p = wilcoxon(a, b)
            entry["metrics"][name] = {"clean": float(a.mean()), "aug": float(b.mean()),
                                      "delta": float(b.mean() - a.mean()),
                                      "n_improved": int((b < a).sum()), "p": float(p)}
            ps.append(p)
        for name, p_adj in zip(entry["metrics"], holm(ps)):
            entry["metrics"][name]["p_holm"] = float(p_adj)
        costs[s] = entry
    return {"score_comparison": out, "augmentation_cost_by_score": costs}


def render_md(table, stats):
    lines = ["# E11: logit-space vs penultimate-feature rejection scores", "",
             "Same grid as E10 (8 models, two held-out class sets, two seeds, clean and",
             "augmented training). All scores use one forward pass; the only difference",
             "is where the class-conditional distance is measured.", ""]
    for mode in MODES:
        lines += ["## %s-trained: mean AUROC / TPR at 5%% FPR" % mode, "",
                  "| model | softmax AUROC | logit-Maha AUROC | feat-Maha AUROC | softmax TPR95 | logit-Maha TPR95 | feat-Maha TPR95 | feat dim |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in MODELS:
            if m not in table or mode not in table[m]:
                continue
            r = table[m][mode]
            dim = "" if r.get("feat_dim") is None else str(r["feat_dim"])
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s |" % (
                m, r["softmax"]["auroc"], r["mahalanobis"]["auroc"], r["mahalanobis_feat"]["auroc"],
                r["softmax"]["tpr95"], r["mahalanobis"]["tpr95"], r["mahalanobis_feat"]["tpr95"], dim))
        # mean row
        keys = [m for m in MODELS if m in table and mode in table[m]]
        if keys:
            mean = lambda s, f: float(np.mean([table[m][mode][s][f] for m in keys]))
            lines.append("| **mean** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | |" % (
                mean("softmax", "auroc"), mean("mahalanobis", "auroc"), mean("mahalanobis_feat", "auroc"),
                mean("softmax", "tpr95"), mean("mahalanobis", "tpr95"), mean("mahalanobis_feat", "tpr95")))
        lines.append("")
    for mode in MODES:
        comp = stats["score_comparison"].get(mode, {})
        if not comp:
            continue
        lines += ["## %s: paired comparisons (model x holdout x seed, Holm-corrected)" % mode, "",
                  "| comparison | metric | reference | other | delta | improved | p | p_Holm |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, entry in comp.items():
            ref, other = entry["ref"], entry["other"]
            for metric, d in entry["metrics"].items():
                lines.append("| %s | %s | %.3f | %.3f | %+.3f | %d/%d | %.2g | %.2g |" % (
                    name, metric, d[ref], d[other], d["delta"], d["n_improved"],
                    entry["n_pairs"], d["p"], d["p_holm"]))
        lines.append("")
    costs = stats["augmentation_cost_by_score"]
    if costs:
        lines += ["## Cost of augmentation on rejection, per score", "",
                  "| score | clean AUROC | aug AUROC | delta | clean TPR95 | aug TPR95 | delta | p_Holm (AUROC) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s, entry in costs.items():
            a, t = entry["metrics"]["auroc"], entry["metrics"]["tpr95"]
            lines.append("| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f | %.2g |" % (
                s, a["clean"], a["aug"], a["delta"], t["clean"], t["aug"], t["delta"], a["p_holm"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = run(seeds, args.epochs)
    table = aggregate(results)
    stats = paired(results)
    out = {"table": table, "stats": stats,
           "meta": {"seeds": seeds, "epochs": args.epochs, "n_runs": len(results),
                    "models": MODELS, "holds": HOLDS}}
    (Path(RESULTS_DIR) / "e11_penultimate.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    md = render_md(table, stats)
    (Path(RESULTS_DIR) / "e11_penultimate.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
