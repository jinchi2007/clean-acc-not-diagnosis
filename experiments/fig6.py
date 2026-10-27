"""Figure 6: robustness vs rejectability (scatter, 8 models)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = {"wdcnn":"WDCNN","ticnn":"TICNN","mcnn":"MCNN","resnet1d":"ResNet1D","ds_cnn":"DS-CNN",
        "lenet1d":"LeNet1D","vit1d":"ViT1D","mlp_mixer1d":"MLP-Mixer1D"}

e2 = json.load(open(os.path.join(ROOT, "results", "e2_noise_clean.json")))["table"]
e3 = json.load(open(os.path.join(ROOT, "results", "e3_openset.json")))["table"]

pts = []
for k in e2:
    acc10 = e2[k]["+10dB"][0]
    auroc = 0.5 * (e3[k]["hold369"]["auroc"] + e3[k]["hold147"]["auroc"])
    pts.append((NAME[k], acc10, auroc))

xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
rho, p = spearmanr(xs, ys)
print(f"n={len(xs)} Spearman rho={rho:.3f} p={p:.4f}")
fig, ax = plt.subplots(figsize=(4.2, 3.3), dpi=200)
ax.scatter(xs, ys, s=46, c="#c0392b", zorder=3)
off = {"WDCNN": (-0.120, -0.006), "TICNN": (0.008, 0.012), "MCNN": (0.008, 0.006),
       "ResNet1D": (0.010, -0.030), "DS-CNN": (-0.010, 0.022), "LeNet1D": (0.010, -0.006),
       "ViT1D": (-0.100, 0.014), "MLP-Mixer1D": (0.008, -0.024)}
for n, x, y in pts:
    dx, dy = off.get(n, (0.008, 0.006))
    ax.annotate(n, (x, y), xytext=(x + dx, y + dy), fontsize=7.5)
b, a = np.polyfit(xs, ys, 1)
xx = np.linspace(min(xs), max(xs), 20)
ax.plot(xx, a + b * xx, ls="--", lw=1.1, color="#555555", zorder=1)
ax.text(0.03, 0.05, f"Spearman $\\rho = {rho:.2f}$ ($p = {p:.2f}$)", transform=ax.transAxes, fontsize=8)
ax.set_xlabel("test accuracy at 10 dB SNR (clean-trained)")
ax.set_ylabel("mean open-set AUROC")
ax.set_ylim(0.72, 1.03)
ax.set_xlim(0.38, 1.03)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig6_openset.{ext}"))
print("wrote figures/fig6_openset.pdf/png")
for n, x, y in sorted(pts, key=lambda t: -t[1]):
    print(f"  {n:12s} 10dB={x:.4f} AUROC={y:.4f}")
