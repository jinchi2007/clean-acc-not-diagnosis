"""E7 效率实测：参数量 + MPS 推理延迟（batch=1 与 batch=256）+ 吞吐。

输出 results/e7_efficiency.json 与 results/e7_efficiency.md：
每个模型的参数量、单样本延迟（batch=1）、batch=256 的批量延迟与单样本摊薄延迟、吞吐。
延迟为 warmup 后的连续前向平均；MPS 用 synchronize 保证计时包含 GPU 计算。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from code.common import RESULTS_DIR, count_params, get_device
from code.models import MODELS, build_model

IN_LEN = 1024
NUM_CLASSES = 10
REPS_B1 = 30
REPS_B256 = 10
WARMUP = 5


def bench(model, device, batch, reps=REPS_B1, warmup=WARMUP):
    x = torch.randn(batch, 1, IN_LEN, device=device)
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - t0) / reps


def render_md(payload):
    lines = ["# E7: 效率（参数量 / MPS 延迟 / 吞吐）", ""]
    lines.append(
        "环境：Apple M5 / torch %s / device=%s / in_len=%d / batch 1 与 256 / "
        "warmup %d 次后连续前向 B1=%d、B256=%d 次取平均（MPS synchronize）。"
        % (payload["torch"], payload["device"], payload["in_len"],
           payload["warmup"], payload["reps_b1"], payload["reps_b256"]))
    lines.append("")
    lines.append("| model | params | b1 ms | b1 ms/sample | b256 ms | b256 ms/sample | "
                 "b256 samples/s | b1 samples/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in payload["rows"]:
        lines.append(
            "| %s | %s | %.3f | %.3f | %.2f | %.3f | %.1f | %.1f |"
            % (r["model"], format(r["n_params"], ","), r["lat_b1_ms"],
               r["lat_b1_per_sample_ms"], r["lat_b256_ms"],
               r["lat_b256_per_sample_ms"], r["thr_b256"], r["thr_b1"]))
    lines.append("")
    lines.append("> 注：延迟是 wall-clock，受机器负载影响（本项目主网格与该测量共享 MPS 时"
                 "会偏大 2-3 倍）；同表内相对比较可靠。batch=1 时小模型主要由调度开销决定，"
                 "参数量的差异不直接等于延迟差异。")
    lines.append("")
    return "\n".join(lines)


def main():
    device = get_device("mps")
    rows = []
    for name in MODELS:
        m = build_model(name, NUM_CLASSES, IN_LEN)
        t1 = bench(m, device, 1)
        t256 = bench(m, device, 256, reps=REPS_B256)
        rows.append(dict(
            model=name,
            n_params=count_params(m),
            lat_b1_ms=round(t1 * 1e3, 3),
            lat_b1_per_sample_ms=round(t1 * 1e3, 3),
            thr_b1=round(1.0 / t1, 1),
            lat_b256_ms=round(t256 * 1e3, 2),
            lat_b256_per_sample_ms=round(t256 * 1e3 / 256.0, 4),
            thr_b256=round(256.0 / t256, 1),
        ))
        print(rows[-1], flush=True)

    payload = dict(
        experiment="e7_efficiency",
        device=str(device),
        torch=torch.__version__,
        in_len=IN_LEN,
        num_classes=NUM_CLASSES,
        batch_sizes=[1, 256],
        reps_b1=REPS_B1,
        reps_b256=REPS_B256,
        warmup=WARMUP,
        rows=rows,
    )
    out_json = Path(RESULTS_DIR) / "e7_efficiency.json"
    out_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    md = render_md(payload)
    (Path(RESULTS_DIR) / "e7_efficiency.md").write_text(md)
    print(md)
    print("saved", out_json)


if __name__ == "__main__":
    main()
