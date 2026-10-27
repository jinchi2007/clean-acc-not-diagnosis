"""Build the saturation table (8 models x 4 splits) from results/runs/*.

Only clean-training, full-data, 10-class runs are used; per-seed accuracies are averaged
within a seed first (several grids may cover the same (model, split, seed)).
"""
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["wdcnn", "ticnn", "mcnn", "resnet1d", "ds_cnn", "lenet1d", "vit1d", "mlp_mixer1d"]
SPLITS = ["leaky", "safe", "crossload", "filedisjoint"]
NAME = {"wdcnn":"WDCNN","ticnn":"TICNN","mcnn":"MCNN","resnet1d":"ResNet1D","ds_cnn":"DS-CNN",
        "lenet1d":"LeNet1D","vit1d":"ViT1D","mlp_mixer1d":"MLP-Mixer1D"}

def usable(cfg, r):
    if cfg.get("holdout_classes"): return False
    if cfg.get("noise_snr") is not None: return False      # single fixed test SNR = subset eval
    if cfg.get("train_noise_lo") is not None or cfg.get("train_noise_hi") is not None: return False
    if cfg.get("stem"): return False
    meta = r.get("data_meta") or {}
    if meta.get("dataset", "cwru") != "cwru": return False
    if meta.get("task") not in (None, "10class"): return False
    if meta.get("n_per_class"): return False
    if int(cfg.get("epochs", 0)) < 15: return False
    return True

GRIDS = ("e0_sat", "e2_noise_clean")

def collect():
    acc = {}
    for path in glob.glob(os.path.join(ROOT, "results", "runs", "*", "*.json")):
        if os.path.basename(os.path.dirname(path)) not in GRIDS:
            continue
        try:
            r = json.load(open(path))
        except Exception:
            continue
        cfg = r.get("cfg") or {}
        if "test_acc" not in r or "model" not in cfg or "split" not in cfg:
            continue
        if not usable(cfg, r):
            continue
        acc.setdefault((cfg["model"], cfg["split"]), {}).setdefault(int(cfg.get("seed", 0)), []).append(float(r["test_acc"]))
    out = {}
    for key, seeds in acc.items():
        vals = [sum(v) / len(v) for v in seeds.values()]
        out[key] = (sum(vals) / len(vals), len(vals))
    return out

def main():
    acc = collect()
    rows = []
    for m in MODELS:
        cells = [acc.get((m, s), (float("nan"), 0)) for s in SPLITS]
        rows.append((m, cells))
    print("| model | " + " | ".join(SPLITS) + " |")
    print("|---" * (len(SPLITS) + 1) + "|")
    for m, cells in rows:
        print(f"| {NAME[m]} | " + " | ".join(f"{v:.4f} (n={n})" if n else "---" for v, n in cells) + " |")
    print()
    print("LATEX")
    for m, cells in rows:
        if any(n == 0 for _, n in cells):
            print(f"% {NAME[m]} incomplete: " + ", ".join(f"{s}:{n}" for s, (_, n) in zip(SPLITS, cells)))
            continue
        vals = [v for v, _ in cells]
        best, worst = max(vals), min(vals)
        tex = ["\\best{%s}" % f"{v:.4f}" if v == best else ("\\emph{%s}" % f"{v:.4f}" if v == worst else f"{v:.4f}") for v in vals]
        print(f"{NAME[m]:12s} & " + " & ".join(tex) + " \\\\")
    print()
    for j, s in enumerate(SPLITS):
        vals = [cells[j][0] for _, cells in rows if cells[j][1]]
        if vals:
            print(f"{s:12s} n={len(vals)} mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} spread={100*(max(vals)-min(vals)):.1f}pt")

if __name__ == "__main__":
    main()
