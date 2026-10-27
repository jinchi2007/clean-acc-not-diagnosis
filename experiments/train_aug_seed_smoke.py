"""Smoke: 验证 train.py 的噪声增广 RNG 修复后，同一 cfg 重跑逐 batch 完全可复现。

两次运行同 seed、固定 SNR=10 dB（min-noise 场景），比较 CPU 上 20 epoch 的
history 与测试 10 dB 精度；修复前两次会差 1-8 个点。
"""
import json
from pathlib import Path

from code.common import RESULTS_DIR
from code.train import train_one

CFG = dict(model="wdcnn", split="safe", epochs=20, seed=0, device="cpu",
           batch=128, lr=2e-3, wd=1e-4,
           train_noise_lo=10.0, train_noise_hi=10.0, noise_kind="gaussian",
           noise_snrs=[10.0])

runs = [train_one(dict(CFG)) for _ in range(2)]
same = [r["history"] for r in runs]
ident = same[0] == same[1]
summary = {
    "cfg": {k: v for k, v in CFG.items()},
    "device": "cpu",
    "identical_history": ident,
    "final_test_acc": [r["test_acc"] for r in runs],
    "test_acc_10db": [r["noise"]["acc_by_snr"]["10.0"] for r in runs],
    "train_loss_last": [r["history"][-1]["train_loss"] for r in runs],
}
Path(RESULTS_DIR, "train_aug_seed_smoke.json").write_text(
    json.dumps(summary, indent=1, ensure_ascii=False))
print(json.dumps(summary, indent=1, ensure_ascii=False))
