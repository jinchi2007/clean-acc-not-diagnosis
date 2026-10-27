"""Publication figures for the paper. Reads results/runs/<grid>/*.json.

    python -m experiments.figures

Outputs vector (PDF) + raster (PNG) into figures/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.common import FIG_DIR, RESULTS_DIR

RUNS = Path(RESULTS_DIR) / "runs"
FIG = Path(FIG_DIR)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.markersize": 3.5,
})

SNRS = [15.0, 10.0, 5.0, 0.0, -5.0]
MODELS = ["wdcnn", "ticnn", "mcnn", "resnet1d", "ds_cnn", "lenet1d", "vit1d", "mlp_mixer1d"]
NICE = {"wdcnn": "WDCNN", "ticnn": "TICNN", "mcnn": "MCNN", "resnet1d": "ResNet1D",
        "ds_cnn": "DS-CNN", "lenet1d": "LeNet1D", "vit1d": "ViT1D", "mlp_mixer1d": "MLP-Mixer1D"}


def load(grid):
    out = []
    for f in sorted((RUNS / grid).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        out.append(d)
    return out


def curves(grid):
    """{model: (clean, [acc@snr...], std arrays)}"""
    agg = defaultdict(lambda: defaultdict(list))
    for d in load(grid):
        cfg = d.get("cfg", {})
        m = cfg.get("model")
        if not m:
            continue
        agg[m]["clean"].append(d["test_acc"])
        for s in SNRS:
            v = (d.get("noise") or {}).get("acc_by_snr", {}).get(str(s))
            if v is not None:
                agg[m][str(s)].append(v)
    res = {}
    for m, dd in agg.items():
        clean = np.array(dd["clean"], float)
        means = np.array([np.mean(dd[str(s)]) if dd[str(s)] else np.nan for s in SNRS])
        stds = np.array([np.std(dd[str(s)], ddof=1) if len(dd[str(s)]) > 1 else 0.0 for s in SNRS])
        res[m] = dict(clean=clean.mean(), clean_std=clean.std(ddof=1) if len(clean) > 1 else 0.0,
                      mean=means, std=stds, n=len(clean))
    return res


def fig_noise():
    e2, e4 = curves("e2_noise_clean"), curves("e4_aug")
    if not e2:
        print("skip fig2 (no e2 runs)")
        return
    xs = [20] + SNRS[::-1]  # clean, -5, 0, 5, 10, 15
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, data, title in ((axes[0], e2, "(a) trained clean"), (axes[1], e4, "(b) noise-augmented training")):
        for m in MODELS:
            if m not in data:
                continue
            d = data[m]
            ys = [d["clean"]] + list(d["mean"][::-1])
            sd = [d["clean_std"]] + list(d["std"][::-1])
            ax.plot(xs, ys, marker="o", label=NICE[m], linewidth=1.2)
            ax.fill_between(xs, np.array(ys) - sd, np.array(ys) + sd, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("test SNR (dB); 20 = clean")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(xs[::-1])
    axes[0].set_ylabel("test accuracy")
    axes[1].legend(ncol=2, loc="lower right", framealpha=0.9)
    fig.savefig(FIG / "fig2_noise.pdf")
    fig.savefig(FIG / "fig2_noise.png")
    plt.close(fig)
    print("fig2_noise")


def fig_rank():
    e2 = curves("e2_noise_clean")
    if not e2:
        return
    offsets = {"wdcnn": (-46, -4), "ticnn": (-40, -10), "resnet1d": (6, -8),
               "ds_cnn": (6, 4), "mcnn": (6, 4), "vit1d": (6, 6),
               "mlp_mixer1d": (6, -8), "lenet1d": (6, 4)}
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for m, d in e2.items():
        ax.scatter(d["clean"], d["mean"][1], s=22, label=NICE[m], zorder=3)
        ax.annotate(NICE[m], (d["clean"], d["mean"][1]), fontsize=5.5,
                    xytext=offsets.get(m, (4, 2)), textcoords="offset points")
    ax.set_xlabel("clean test accuracy")
    ax.set_ylabel("test accuracy at 10 dB SNR")
    ax.set_xlim(0.94, 1.005)
    ax.set_ylim(0.35, 1.05)
    fig.savefig(FIG / "fig3_rank_inversion.pdf")
    fig.savefig(FIG / "fig3_rank_inversion.png")
    plt.close(fig)
    print("fig3_rank_inversion")


def fig_aug_gain():
    e2, e4 = curves("e2_noise_clean"), curves("e4_aug")
    if not e2 or not e4:
        return
    models = [m for m in MODELS if m in e2 and m in e4]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    a = [e2[m]["mean"][1] for m in models]
    b = [e4[m]["mean"][1] for m in models]
    ax.bar(x - w / 2, a, w, label="trained clean", color="#9ecae1")
    ax.bar(x + w / 2, b, w, label="+ noise aug.", color="#3182bd")
    ax.set_xticks(x, [NICE[m] for m in models], rotation=20, ha="right")
    ax.set_ylabel("accuracy @ 10 dB")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.savefig(FIG / "fig4_aug_gain.pdf")
    fig.savefig(FIG / "fig4_aug_gain.png")
    plt.close(fig)
    print("fig4_aug_gain")


def fig_stem():
    p = Path(RESULTS_DIR) / "stem_ablation.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    rows = data.get("rows", [])
    if not rows:
        print("skip fig5 (no rows)")
        return
    base = {}
    for r in rows:
        if not r.get("c"):
            base[r["backbone"]] = r
    pts = []
    for r in rows:
        if not r.get("c"):
            continue
        b = base.get(r["backbone"])
        if not b:
            continue
        dc = round((r["clean"] - b["clean"]) * 100, 2)
        d10 = round((r["by_snr"]["10.0"] - b["by_snr"]["10.0"]) * 100, 2)
        pts.append((r["label"], dc, d10))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for name, dc, d10 in pts:
        ax.scatter(dc, d10, s=26, zorder=3)
        ax.annotate(name.replace("_", " "), (dc, d10), fontsize=5,
                    xytext=(3, 2), textcoords="offset points")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Δ clean accuracy (points)")
    ax.set_ylabel("Δ accuracy @ 10 dB (points)")
    fig.savefig(FIG / "fig5_stem_tradeoff.pdf")
    fig.savefig(FIG / "fig5_stem_tradeoff.png")
    plt.close(fig)
    print("fig5_stem_tradeoff")


def fig_saturation():
    runs = load("e0_sat")
    if not runs:
        return
    splits = ["leaky", "safe", "crossload", "filedisjoint"]
    agg = defaultdict(lambda: defaultdict(list))
    for d in runs:
        cfg = d.get("cfg", {})
        agg[cfg.get("model")][cfg.get("split")].append(d["test_acc"])
    models = [m for m in MODELS if m in agg]
    if not models:
        return
    mat = np.full((len(models), len(splits)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(splits):
            if agg[m].get(s):
                mat[i, j] = np.mean(agg[m][s])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(mat, vmin=0.85, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(splits)), splits, rotation=20, ha="right")
    ax.set_yticks(range(len(models)), [NICE[m] for m in models])
    for i in range(len(models)):
        for j in range(len(splits)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]*100:.1f}", ha="center", va="center",
                        color="w" if mat[i, j] < 0.95 else "k", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="clean accuracy", shrink=0.85)
    fig.savefig(FIG / "fig1_saturation.pdf")
    fig.savefig(FIG / "fig1_saturation.png")
    plt.close(fig)
    print("fig1_saturation")


if __name__ == "__main__":
    fig_noise()
    fig_rank()
    fig_aug_gain()
    fig_stem()
    fig_saturation()